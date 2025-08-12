import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.sampler import BatchSampler, RandomSampler
from datasets.coco_style_dataset import CocoStyleDataset, CocoStyleDatasetTeaching
from models.backbone import DINOv2Backbone, Joiner
from models.position_encoding import PositionEmbeddingSine, PositionEmbeddingLearned
from models.deformable_detr import DeformableDETR
from models.criterion import SetCriterion
from models.transformer import Transformer
from models.matcher import HungarianMatcher
from datasets.augmentations import weak_aug, strong_aug, base_trans


def build_sampler(args, dataset, split):
    if split == 'train':
        if args.distributed:
            sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
        else:
            sampler = RandomSampler(dataset)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    else:
        if args.distributed:
            sampler = DistributedSampler(dataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler, args.eval_batch_size, drop_last=False)
    return batch_sampler


def build_dataloader(args, dataset_name, domain, split, trans):
    dataset = CocoStyleDataset(root_dir=args.data_root,
                               dataset_name=dataset_name,
                               domain=domain,
                               split=split,
                               transforms=trans)
    batch_sampler = build_sampler(args, dataset, split)
    data_loader = DataLoader(dataset=dataset,
                             batch_sampler=batch_sampler,
                             collate_fn=CocoStyleDataset.collate_fn,
                             num_workers=args.num_workers)
    return data_loader


def build_dataloader_teaching(args, dataset_name, domain, split):
    dataset = CocoStyleDatasetTeaching(root_dir=args.data_root,
                                       dataset_name=dataset_name,
                                       domain=domain,
                                       split=split,
                                       weak_aug=weak_aug(args.img_size),
                                       strong_aug=strong_aug(),
                                       final_trans=base_trans())
    batch_sampler = build_sampler(args, dataset, split)
    data_loader = DataLoader(dataset=dataset,
                             batch_sampler=batch_sampler,
                             collate_fn=CocoStyleDatasetTeaching.collate_fn_teaching,
                             num_workers=args.num_workers)
    return data_loader


def build_model(args, device):
    backbone = build_backbone(args)

    args.num_feature_levels = len(args.projector_scale)

    transformer = Transformer(
        d_model=args.hidden_dim,
        sa_nhead=args.sa_nheads,
        ca_nhead=args.ca_nheads,
        num_queries=args.num_queries,
        dropout=args.dropout,
        dim_feedforward=args.feedforward_dim,
        num_decoder_layers=args.num_decoder_layers,
        return_intermediate_dec=True,
        group_detr=args.group_detr,
        two_stage=args.two_stage,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        lite_refpoint_refine=True,
        decoder_norm_type='LN',
        bbox_reparam=True,
    )

    model = DeformableDETR(
        args,
        backbone,
        transformer,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        group_detr=args.group_detr,
        two_stage=True,
        lite_refpoint_refine=True,
        bbox_reparam=True,
    )
    model.to(device)
    return model


def build_criterion(args, device, box_loss=True):
    matcher = HungarianMatcher()
    weight_dict = {}
    weight_dict['loss_ce'] = args.coef_class
    weight_dict['loss_bbox'] = args.coef_boxes if box_loss else 0 
    weight_dict['loss_giou'] = args.coef_giou if box_loss else 0 
    aux_weight_dict = {}
    for i in range(args.num_decoder_layers - 1):
        aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
    aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)
        
    weight_dict['loss_mae'] = args.coef_mae
    weight_dict['loss_domain'] = args.coef_domain

    losses = ['labels', 'boxes', 'cardinality']

    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(args.num_classes, 
                             matcher, 
                             weight_dict, 
                             losses, 
                             focal_alpha=args.focal_alpha,
                             group_detr=args.group_detr,
                             alpha_dt=args.alpha_dt,
                             gamma_dt=args.gamma_dt,
                             max_dt=args.max_dt,
                             device=device
                             )
    criterion.to(device)

    return criterion


def build_backbone(args):
    position_embedding = PositionEmbeddingSine(args.hidden_dim // 2, normalize=True)
    if 'dinov2' in args.backbone:
        backbone = DINOv2Backbone(args, peft=True)
    else:     
        raise NotImplementedError(f"Backbone {args.backbone} is not implemented.")
    model = Joiner(backbone, position_embedding)
  
    return model


def build_optimizer(args, model, enable_mae=False):
    params_backbone = [param for name, param in model.named_parameters()
                       if 'backbone' in name]
    params_linear_proj = [param for name, param in model.named_parameters()
                          if 'reference_points' in name or 'sampling_offsets' in name]
    params = [param for name, param in model.named_parameters()
              if 'backbone' not in name and 'reference_points' not in name and 'sampling_offsets' not in name]
    param_dicts = [
        {'params': params, 'lr': args.lr},
        {'params': params_backbone, 'lr': 0.0 if enable_mae else args.lr_backbone},
        {'params': params_linear_proj, 'lr': args.lr_linear_proj},
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer


def build_teacher(args, student_model, device):
    teacher_model = build_model(args, device)
    state_dict, student_state_dict = teacher_model.state_dict(), student_model.state_dict()
    for key, value in state_dict.items():
        state_dict[key] = student_state_dict[key].clone().detach()
    teacher_model.load_state_dict(state_dict)
    return teacher_model
