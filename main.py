import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
import argparse
import random
import copy
from pathlib import Path
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

from engine import *
from build_modules import *
from datasets.augmentations import train_trans, val_trans, strong_trans
from utils import get_rank, init_distributed_mode, resume_and_load, resume_and_load_MAE, save_ckpt, selective_reinitialize

# torch.autograd.set_detect_anomaly(True) 

def get_args_parser(parser):
    # Model Settings
    parser.add_argument('--backbone', default='facebook/dinov2-large', type=str)
    parser.add_argument('--pos_encoding', default='sine', type=str)
    parser.add_argument('--num_classes', default=4, type=int)
    parser.add_argument('--num_queries', default=300, type=int)
    parser.add_argument('--num_feature_levels', default=4, type=int)
    parser.add_argument('--with_box_refine', default=True, type=bool)
    parser.add_argument('--two_stage', default=True, type=bool)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--sa_nheads', default=8, type=int)
    parser.add_argument('--ca_nheads', default=16, type=int)
    parser.add_argument('--num_decoder_layers', default=6, type=int)
    parser.add_argument('--feedforward_dim', default=1024, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--dec_n_points', default=4, type=int)
    # Optimization hyperparameters
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--eval_batch_size', default=16, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj', default=2e-5, type=float)
    parser.add_argument('--sgd', default=False, type=bool)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--epoch_lr_drop', default=40, type=int)
    # Loss coefficients
    parser.add_argument('--teach_box_loss', default=False, type=bool)
    parser.add_argument('--coef_class', default=2.0, type=float)
    parser.add_argument('--coef_boxes', default=5.0, type=float)
    parser.add_argument('--coef_giou', default=2.0, type=float)
    parser.add_argument('--coef_target', default=1.0, type=float)
    parser.add_argument('--coef_domain', default=1.0, type=float)
    parser.add_argument('--coef_mae', default=1.0, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--alpha_ema', default=0.9996, type=float)
    # Dataset parameters
    parser.add_argument('--data_root', default='/input0', type=str)
    parser.add_argument('--source_dataset', default='xView', type=str)
    parser.add_argument('--target_dataset', default='DOTA', type=str)
    parser.add_argument('--img_size', default=700, type=int)
    # Retraining parameters
    parser.add_argument('--epoch_retrain', default=40, type=int)
    parser.add_argument('--retrain_modules', default=["dinov2"], type=str, nargs="+")
    # MAE parameters
    parser.add_argument('--epoch_mae_decay', default=10, type=float)
    # Dynamic threshold (DT) parameters
    parser.add_argument('--threshold', default=0.3, type=float)
    parser.add_argument('--alpha_dt', default=0.5, type=float)
    parser.add_argument('--gamma_dt', default=0.9, type=float)
    parser.add_argument('--max_dt', default=0.45, type=float)
    # mode settings
    parser.add_argument("--mode", default="single_domain", type=str,
                        help="'single_domain' for single domain training, "
                             "'cross_domain_mae' for cross domain training with mae, "
                             "'teaching' for teaching process, " 
                             "'eval' for evaluation only.")
    # Other settings
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--output_dir', default='./output', type=str)
    parser.add_argument('--random_seed', default=8008, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--print_freq', default=20, type=int)
    parser.add_argument('--flush', default=True, type=bool)
    parser.add_argument("--resume", default="", type=str)
    # ===================================================================
    """
    所有自己添加的参数都在这
    """
    parser.add_argument('--group_detr', default=13, type=int)
    parser.add_argument('--R_LoRA', default=16, type=int)
    # parser.add_argument('--feature_extraction_layers', default=[2, 5, 8, 11], type=int, nargs='+')
    parser.add_argument('--projector_scale', default=[1.0], type=float, nargs='+')
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--mae_depth', default=8, type=int)
    parser.add_argument('--mae_nheads', default=16, type=int)

    # --- loss ---

    # ===================================================================


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_loss(epoch, prefix, total_loss, loss_dict):
    writer.add_scalar(prefix + '/total_loss', total_loss, epoch)
    for k, v in loss_dict.items():
        writer.add_scalar(prefix + '/' + k, v, epoch)


def write_ap50(epoch, prefix, m_ap, ap_per_class, idx_to_class):
    writer.add_scalar(prefix + '/mAP50', m_ap, epoch)
    for idx, num in zip(idx_to_class.keys(), ap_per_class):
        writer.add_scalar(prefix + '/AP50_%s' % (idx_to_class[idx]['name']), num, epoch)


def single_domain_training(model, device):
    # Record the start time
    start_time = time.time()
    # Build dataloaders
    train_loader = build_dataloader(args, args.source_dataset, 'source', 'train', train_trans())
    val_loader = build_dataloader(args, args.target_dataset, 'target', 'val', val_trans(args.img_size))
    idx_to_class = val_loader.dataset.coco.cats
    # Prepare model for optimization
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    criterion = build_criterion(args, device)
    optimizer = build_optimizer(args, model)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.epoch_lr_drop)
    # Record some args
    with open(output_dir / 'args.txt', 'w') as f:
        json.dump(vars(args), f, indent=4)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    with open(output_dir / 'args.txt', 'a') as f:
        f.write(f'\nnumber of params: {n_parameters}\n')
    for n, p in model.named_parameters():
        with open(output_dir / 'args.txt', 'a') as f:
            f.write(f'{n}\n' if p.requires_grad else f'*{n}\n')
    # Record the best mAP
    ap50_best = -1.0
    for epoch in range(args.epoch):
        # Set the epoch for the sampler
        if args.distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        # Train for one epoch
        loss_train, loss_train_dict = train_one_epoch_standard(
            model=model,
            criterion=criterion,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            clip_max_norm=args.clip_max_norm,
            print_freq=args.print_freq,
            flush=args.flush
        )
        write_loss(epoch, 'single_domain', loss_train, loss_train_dict)
        lr_scheduler.step()
        # Evaluate
        ap50_per_class, loss_val = evaluate(
            model=model,
            criterion=criterion,
            data_loader_val=val_loader,
            device=device,
            print_freq=args.print_freq,
            flush=args.flush
        )
        criterion.clear_positive_logits()
        # Save the best checkpoint
        map50 = np.asarray([ap for ap in ap50_per_class if ap > -0.001]).mean().tolist()
        if map50 > ap50_best:
            ap50_best = map50
            ap50_per_class_best = ap50_per_class
            save_ckpt(model, output_dir/'model_best.pth', args.distributed)
        if epoch == args.epoch - 1:
            save_ckpt(model, output_dir/'model_last.pth', args.distributed)
        # Write the evaluation results to tensorboard
        write_ap50(epoch, 'single_domain', map50, ap50_per_class, idx_to_class)
    # Record the end time
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Single-domain training finished. Time cost: ' + total_time_str +
          ' . Best mAP50: ' + str(ap50_best), flush=args.flush)
    with open(output_dir / 'ap_best.txt', 'w') as f:
        f.write(f'Best mAP50:{ap50_best:.4f}\n')
        for idx in idx_to_class.keys():
            f.write(f'{idx_to_class[idx]["name"]:<15} {ap50_per_class_best[idx]:.4f}\n')


def cross_domain_mae(model, device):
    start_time = time.time()
    # Build dataloaders
    source_loader = build_dataloader(args, args.source_dataset, 'source', 'train', strong_trans(args.img_size))
    target_loader = build_dataloader(args, args.target_dataset, 'target', 'train', strong_trans(args.img_size))
    val_loader = build_dataloader(args, args.target_dataset, 'target', 'val', val_trans(args.img_size))
    idx_to_class = val_loader.dataset.coco.cats
    # Build MAE branch
    assert args.coef_mae != 0
    model.build_MAEDecoder(args.img_size, device, 
                           depth=args.mae_depth, 
                           num_heads=args.mae_nheads
                           )
    # Prepare model for optimization
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    criterion, criterion_mae = build_criterion(args, device), build_criterion(args, device)
    optimizer = build_optimizer(args, model)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.epoch_lr_drop)
    # Record some args
    with open(output_dir / 'args.txt', 'w') as f:
        json.dump(vars(args), f, indent=4)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    with open(output_dir / 'args.txt', 'a') as f:
        f.write(f'\nnumber of params: {n_parameters}\n')
    for n, p in model.named_parameters():
        with open(output_dir / 'args.txt', 'a') as f:
            f.write(f'{n}\n' if p.requires_grad else f'*{n}\n')
    # Record the best mAP
    ap50_best = -1.0
    for epoch in range(args.epoch):
        # Set the epoch for the sampler
        if args.distributed and hasattr(source_loader.sampler, 'set_epoch'):
            source_loader.sampler.set_epoch(epoch)
            target_loader.sampler.set_epoch(epoch)
        # Train for one epoch
        loss_train, loss_train_dict = train_one_epoch_with_mae(
            model=model,
            criterion=criterion,
            criterion_mae=criterion_mae,
            source_loader=source_loader,
            target_loader=target_loader,
            coef_target=args.coef_target,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            clip_max_norm=args.clip_max_norm,
            print_freq=args.print_freq,
            flush=args.flush
        )
        write_loss(epoch, 'cross_domain_mae', loss_train, loss_train_dict)
        lr_scheduler.step()
        # Evaluate
        ap50_per_class, loss_val = evaluate(
            model=model,
            criterion=criterion,
            data_loader_val=val_loader,
            device=device,
            print_freq=args.print_freq,
            flush=args.flush
        )
        criterion.clear_positive_logits()
        # Save the best checkpoint
        map50 = np.asarray([ap for ap in ap50_per_class if ap > -0.0001]).mean().tolist()
        if map50 > ap50_best:
            ap50_best = map50
            ap50_per_class_best = ap50_per_class
            save_ckpt(model, output_dir/'model_best.pth', args.distributed)
        if epoch == args.epoch - 1:
            save_ckpt(model, output_dir/'model_last.pth', args.distributed)
        # Write the evaluation results to tensorboard
        write_ap50(epoch, 'cross_domain_mae', map50, ap50_per_class, idx_to_class)
    # Record the end time
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Cross-domain MAE training finished. Time cost: ' + total_time_str +
          ' . Best mAP50: ' + str(ap50_best), flush=args.flush)
    with open(output_dir / 'ap_best.txt', 'w') as f:
        f.write(f'Best mAP50     {ap50_best:.4f}\n')
        for idx in idx_to_class.keys():
            f.write(f'{idx_to_class[idx]["name"]:<15} {ap50_per_class_best[idx]:.4f}\n')


# Teaching
def teaching(model_stu, device):
    start_time = time.time()
    # Build dataloaders
    source_loader = build_dataloader(args, args.source_dataset, 'source', 'train', strong_trans(args.img_size))
    target_loader = build_dataloader_teaching(args, args.target_dataset, 'target', 'train')
    val_loader = build_dataloader(args, args.target_dataset, 'target', 'val', val_trans(args.img_size))
    idx_to_class = val_loader.dataset.coco.cats
    # Build teacher model
    model_tch = build_teacher(args, model_stu, device)
    # Build discriminators
    if args.coef_domain > 0:
        model_stu.build_discriminator(device)
    # Build MAE branch
    assert args.coef_mae != 0
    model_stu.build_MAEDecoder(args.img_size, device, 
                               depth=args.mae_depth,
                               num_heads=args.mae_nheads
                               )
    if args.resume != "":
        model_stu.MAEDecoder = resume_and_load_MAE(model_stu.MAEDecoder, args.resume, device)
    # Prepare model for optimization
    if args.distributed:
        model_stu = DistributedDataParallel(model_stu, device_ids=[args.gpu], find_unused_parameters=True)
        model_tch = DistributedDataParallel(model_tch, device_ids=[args.gpu], find_unused_parameters=True)
    criterion = build_criterion(args, device)
    criterion_pseudo = build_criterion(args, device, box_loss=args.teach_box_loss)
    optimizer = build_optimizer(args, model_stu)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.epoch_lr_drop)
    # Reinitialize checkpoint for selective retraining
    reinit_ckpt = copy.deepcopy(model_tch.state_dict())
    # Record model
    with open(output_dir / 'args.txt', 'w') as f:
        json.dump(vars(args), f, indent=4)
    for n, p in model_stu.named_parameters():
        with open(output_dir / 'args.txt', 'a') as f:
            f.write(f'{n}\n' if p.requires_grad else f'--{n}\n')
    for n, p in model_tch.named_parameters():
        with open(output_dir / 'args.txt', 'a') as f:
            f.write(f'{n}\n' if p.requires_grad else f'--{n}\n')
    # Initialize thresholds
    thresholds = [args.threshold] * args.num_classes
    # Record the best mAP
    ap50_best = -1.0
    for epoch in range(args.epoch):
        # Set the epoch for the sampler
        if args.distributed and hasattr(source_loader.sampler, 'set_epoch'):
            source_loader.sampler.set_epoch(epoch)
            target_loader.sampler.set_epoch(epoch)
        loss_train, loss_source_dict, loss_target_dict = train_one_epoch_teaching(
            student_model=model_stu,
            teacher_model=model_tch,
            criterion=criterion,
            criterion_pseudo=criterion_pseudo,
            source_loader=source_loader,
            target_loader=target_loader,
            optimizer=optimizer,
            thresholds=thresholds,
            coef_target=args.coef_target,
            alpha_ema=args.alpha_ema,
            device=device,
            epoch=epoch,
            enable_mae=(epoch < args.epoch_mae_decay),
            clip_max_norm=args.clip_max_norm,
            print_freq=args.print_freq,
            flush=args.flush
        )
        # Renew thresholds
        thresholds = criterion.dynamic_threshold(thresholds)
        criterion.clear_positive_logits()
        # Write the losses to tensorboard
        write_loss(epoch, 'teaching_source', loss_train, loss_source_dict)
        write_loss(epoch, 'teaching_target', loss_train, loss_target_dict)
        lr_scheduler.step()
        # Selective Retraining
        if (epoch + 1) % args.epoch_retrain == 0 and epoch != args.epoch - 1:
            model_stu = selective_reinitialize(model_stu, reinit_ckpt, args.retrain_modules)
        # Evaluate teacher and student model
        ap50_per_class_teacher, loss_val_teacher = evaluate(
            model=model_tch,
            criterion=criterion,
            data_loader_val=val_loader,
            device=device,
            print_freq=args.print_freq,
            flush=args.flush
        )
        ap50_per_class_student, loss_val_student = evaluate(
            model=model_stu,
            criterion=criterion,
            data_loader_val=val_loader,
            device=device,
            print_freq=args.print_freq,
            flush=args.flush
        )
        # Renew thresholds
        thresholds = criterion.dynamic_threshold(thresholds)
        criterion.clear_positive_logits()
        criterion_pseudo.clear_positive_logits()
        # Save the best checkpoint
        map50_tch = np.asarray([ap for ap in ap50_per_class_teacher if ap > -0.001]).mean().tolist()
        map50_stu = np.asarray([ap for ap in ap50_per_class_student if ap > -0.001]).mean().tolist()
        write_ap50(epoch, 'teaching_teacher', map50_tch, ap50_per_class_teacher, idx_to_class)
        write_ap50(epoch, 'teaching_student', map50_stu, ap50_per_class_student, idx_to_class)
        if max(map50_tch, map50_stu) > ap50_best:
            # ap50_best = max(map50_tch, map50_stu)
            if map50_tch > map50_stu:
                ap50_best = map50_tch
                ap50_per_class_best = ap50_per_class_teacher
            else:
                ap50_best = map50_stu
                ap50_per_class_best = ap50_per_class_student
            save_ckpt(model_tch if map50_tch > map50_stu else model_stu, output_dir/'model_best.pth', args.distributed)
        if epoch == args.epoch - 1:
            save_ckpt(model_tch, output_dir/'model_last_tch.pth', args.distributed)
            save_ckpt(model_stu, output_dir/'model_last_stu.pth', args.distributed)
        # break
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Teaching finished. Time cost: ' + total_time_str + ' . Best mAP50: ' + str(ap50_best), flush=args.flush)
    with open(output_dir / 'ap_best.txt', 'w') as f:
        f.write(f'Best mAP50:{ap50_best:.4f}\n')
        for idx in idx_to_class.keys():
            f.write(f'{idx_to_class[idx]["name"]:<15} {ap50_per_class_best[idx]:.4f}\n')


# Evaluate only
def eval_only(model, device):
    if args.distributed:
        Warning('Evaluation with distributed mode may cause error in output result labels.')
    criterion = build_criterion(args, device)
    # Eval source or target dataset
    val_loader = build_dataloader(args, args.target_dataset, 'target', 'val', val_trans(args.img_size))
    idx_to_class = val_loader.dataset.coco.cats
    ap50_per_class, epoch_loss_val, coco_data = evaluate(
        model=model,
        criterion=criterion,
        data_loader_val=val_loader,
        output_result_labels=True,
        visualization_path=output_dir/'visualization' if args.visualize else None,
        device=device,
        print_freq=args.print_freq,
        flush=args.flush
    )
    ap_50 = np.asarray([ap for ap in ap50_per_class if ap > -0.001]).mean().tolist()
    print('Evaluation finished. mAPs: ' + str(ap50_per_class) + '. Evaluation loss: ' + str(epoch_loss_val))
    output_file = output_dir/'evaluation_result_labels.json'
    print("Writing evaluation result labels to " + str(output_file))
    with open(output_file, 'w', encoding='utf-8') as fp:
        json.dump(coco_data, fp)
    with open(output_dir / 'ap_eval.txt', 'w') as f:
        f.write(f'mAP50:{ap_50:.4f}\n')
        for idx in idx_to_class.keys():
            f.write(f'{idx_to_class[idx]["name"]:<15} {ap50_per_class[idx]:.4f}\n')


def main():
    # Initialize distributed mode
    init_distributed_mode(args)
    # Set random seed
    if args.random_seed is None:
        args.random_seed = random.randint(1, 10000)
    set_random_seed(args.random_seed + get_rank())
    # Print args
    print('-------------------------------------', flush=args.flush)
    print('Logs will be written to ' + str(logs_dir))
    print('Checkpoints will be saved to ' + str(output_dir))
    print('-------------------------------------', flush=args.flush)
    for key, value in args.__dict__.items():
        print(key, value, flush=args.flush)
    # Build model
    device = torch.device(args.device)
    model = build_model(args, device)
    if args.resume != "":
        model = resume_and_load(model, args.resume, device)
    # Training or evaluation
    print('-------------------------------------', flush=args.flush)
    if args.mode == "single_domain":
        single_domain_training(model, device)
    elif args.mode == "cross_domain_mae":
        cross_domain_mae(model, device)
    elif args.mode == "teaching":
        teaching(model, device)
    elif args.mode == "eval":
        eval_only(model, device)
    else:
        raise ValueError('Invalid mode: ' + args.mode)


if __name__ == '__main__':
    # Parse arguments
    parser_main = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    get_args_parser(parser_main)
    args = parser_main.parse_args()
    # Set output directory
    output_dir = Path(args.output_dir)
    logs_dir = output_dir/'data_logs'
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(logs_dir))
    # Call main function
    main()
