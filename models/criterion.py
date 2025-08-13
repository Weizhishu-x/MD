import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.boxes import nms

import math
import numpy as np

from utils import box_utils as box_ops
from utils.misc_utils import accuracy
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized
from models.segmentation import sigmoid_focal_loss

import copy

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25,
                 group_detr=1, alpha_dt=0.5, gamma_dt=0.9, max_dt=0.45, ia_bce_loss=True, device='cuda'):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
            alpha_dt: 均值对新阈值的贡献程度
            gamma_dt: 历史阈值的权重
            max_dt: 阈值的上限
        """
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.group_detr = group_detr
        self.logits_sum = [torch.zeros(1, dtype=torch.float, device=device) for _ in range(num_classes)]
        self.logits_count = [torch.zeros(1, dtype=torch.int, device=device) for _ in range(num_classes)]
        self.alpha_dt = alpha_dt
        self.gamma_dt = gamma_dt
        self.max_dt = max_dt
        self.ia_bce_loss = ia_bce_loss


    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        if self.ia_bce_loss:  # Improved Adaptive Binary Cross-Entropy Loss
        
            alpha = self.focal_alpha
            gamma = 2 
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            iou_targets=torch.diag(box_ops.box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                box_ops.box_cxcywh_to_xyxy(target_boxes))[0])
            pos_ious = iou_targets.clone().detach()
            prob = src_logits.sigmoid()
            #init positive weights and negative weights
            pos_weights = torch.zeros_like(src_logits)
            neg_weights =  prob ** gamma

            pos_ind=[id for id in idx]
            pos_ind.append(target_classes_o)

            t = prob[pos_ind].pow(alpha) * pos_ious.pow(1 - alpha)
            t = torch.clamp(t, 0.01).detach()

            pos_weights[pos_ind] = t.to(pos_weights.dtype)
            neg_weights[pos_ind] = 1 - t.to(neg_weights.dtype)
            # a reformulation of the standard loss_ce = - pos_weights * prob.log() - neg_weights * (1 - prob).log()
            # with a focus on statistical stability by using fused logsigmoid
            loss_ce = neg_weights * src_logits - F.logsigmoid(src_logits) * (pos_weights + neg_weights)
            loss_ce = loss_ce.sum() / num_boxes
        else:
            target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                        dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o

            target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                                dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

            target_classes_onehot = target_classes_onehot[:,:,:-1]
            loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses
    
    def loss_domains(self, outputs_da, domain_label):
        domain_label = torch.tensor(domain_label, dtype=torch.long, device=self.device)
        # encoder
        loss_domain_enc = 0 
        for idx, domain_pred_enc in enumerate(outputs_da['domain_enc_all']):
            batch_size, _, h, w  = domain_pred_enc.shape   # (batch_size, 2, h, w)
            domain_label_enc = domain_label.expand(batch_size, h, w)
            loss_domain_enc += nn.CrossEntropyLoss()(domain_pred_enc, domain_label_enc)
        loss_domain_enc /= (idx + 1)
        # decoder
        domain_pred_dec = outputs_da['domain_dec_all'].permute(0, 2, 1)  # (batch_size, num_queries, 2) -> (batch_size, 2, num_queries)
        batch_size, _, num_queries = domain_pred_dec.shape
        domain_label_dec = domain_label.expand(batch_size, num_queries)
        loss_domain_dec = nn.CrossEntropyLoss()(domain_pred_dec, domain_label_dec)

        loss_dict = {
            'loss_domain_enc': loss_domain_enc,
            'loss_domain_dec': loss_domain_dec,
        }
        loss = loss_domain_enc + loss_domain_dec
        return loss
    
    # def loss_mae(self, outputs_mae):
    #     F_reconstructed = outputs_mae['mae_reconstructed']
    #     F_teacher = outputs_mae['mae_teacher']
    #     feats_layers = len(F_teacher)
    #     loss = 0.0
    #     for i in range(feats_layers):
    #         loss += F.mse_loss(F_reconstructed[i], F_teacher[i])
    #     loss /= feats_layers
    #     return loss
        
    def loss_mae(self, outputs_mae):
        """
        计算 MAE 重建损失（仅在被 mask 的位置），
        基于 decoder 已还原顺序后的重建特征。

        Args:
            outputs_mae: dict，包含：
                - 'mae_reconstructed': list of B x C x H x W（decoder输出）
                - 'mae_teacher': list of B x C x H x W（原始特征）
                - 'mask': B x N（1=mask, 0=visible）

        Returns:
            loss: scalar
        """
        F_reconstructed = outputs_mae['mae_reconstructed']
        F_teacher = outputs_mae['mae_teacher']
        mask = outputs_mae['mask'].bool()  # B x N

        loss = 0.0
        num_layers = len(F_teacher)

        for i in range(num_layers):
            B, C, H, W = F_teacher[i].shape
            N = H * W

            # 展平：B x C x H x W -> B x N x C
            teacher = F_teacher[i].flatten(2).transpose(1, 2)       # B x N x C
            pred = F_reconstructed[i].flatten(2).transpose(1, 2)    # B x N x C

            # 取出 mask 位置
            pred_masked = pred[mask]         # (M_mask, C)
            teacher_masked = teacher[mask]   # (M_mask, C)

            loss += F.mse_loss(pred_masked, teacher_masked, reduction='mean')

        return loss / num_layers
    
    # def loss_mae(self, outputs_mae, mask):
    #     """
    #     只在被 mask 掉的位置上计算 MAE 损失，使用 F.mse_loss

    #     Args:
    #         outputs_mae: 包含 'mae_reconstructed' 和 'mae_teacher' 两个 list（每层特征）
    #         mask: Tensor of shape (B, N)，1 表示被 mask 的 patch
    #     """
    #     F_reconstructed = outputs_mae['mae_reconstructed']  # list of [B, C, H, W]
    #     F_teacher = outputs_mae['mae_teacher']              # list of [B, C, H, W]
        
    #     loss = 0.0
    #     feats_layers = len(F_teacher)

    #     for i in range(feats_layers):
    #         # Flatten spatial feature maps to patch sequence: [B, C, H, W] -> [B, N, C]
    #         pred = F_reconstructed[i].flatten(2).transpose(1, 2)  # (B, N, C)
    #         gt   = F_teacher[i].flatten(2).transpose(1, 2)        # (B, N, C)

    #         # mask: (B, N), 1 表示被 mask
    #         mask_expanded = mask.unsqueeze(-1).expand_as(pred)    # (B, N, C)

    #         # 仅选取被 mask 的 patch
    #         pred_masked = pred[mask_expanded].view(-1, pred.size(-1))  # (num_masked, C)
    #         gt_masked   = gt[mask_expanded].view(-1, gt.size(-1))      # (num_masked, C)

    #         # 使用 F.mse_loss 计算被 mask 的 patch 上的平均损失
    #         loss += F.mse_loss(pred_masked, gt_masked, reduction='mean')

    #     loss /= feats_layers
    #     return loss
            
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def record_positive_logits(self, logits, indices):
        idx = self._get_src_permutation_idx(indices)
        labels = logits[idx].argmax(dim=1)
        pos_logits = logits[idx].max(dim=1).values
        for label, logit in zip(labels, pos_logits):
            self.logits_sum[label] += logit.detach()
            self.logits_count[label] += 1

    def dynamic_threshold(self, thresholds):
        if torch.distributed.is_initialized():
            for s in self.logits_sum:
                torch.distributed.all_reduce(s)
            for n in self.logits_count:
                torch.distributed.all_reduce(n)
        logits_means = [s.item() / n.item() if n > 0 else 0.0 for s, n in zip(self.logits_sum, self.logits_count)]
        assert len(logits_means) == len(thresholds)
        new_thresholds = [self.gamma_dt * threshold + (1 - self.gamma_dt) * self.alpha_dt * math.sqrt(mean)
                          for threshold, mean in zip(thresholds, logits_means)]
        new_thresholds = [max(min(threshold, self.max_dt), 0.25) for threshold in new_thresholds]
        print('New Dynamic Thresholds: ', new_thresholds)
        return new_thresholds

    def clear_positive_logits(self):
        self.logits_sum = [torch.zeros(1, dtype=torch.float, device=self.device) for _ in range(self.num_classes)]
        self.logits_count = [torch.zeros(1, dtype=torch.int, device=self.device) for _ in range(self.num_classes)]
    
    def forward(self, outputs, targets=None, domain_label=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        group_detr = self.group_detr if self.training else 1
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets) if targets is not None else 0
        num_boxes = num_boxes * group_detr
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=outputs['pred_logits'].device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        
        if targets is not None:
            outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
            # Retrieve the matching between the outputs of the last layer and the targets
            indices = self.matcher(outputs_without_aux, targets, group_detr=group_detr)
            # Record positive logits
            self.record_positive_logits(outputs['pred_logits'].sigmoid(), indices)
            # Compute all the requested losses
            for loss in self.losses:
                kwargs = {}
                losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

            # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
            if 'aux_outputs' in outputs:
                for i, aux_outputs in enumerate(outputs['aux_outputs']):
                    indices = self.matcher(aux_outputs, targets, group_detr=group_detr)
                    for loss in self.losses:
                        if loss == 'masks':
                            # Intermediate masks losses are too costly to compute, we ignore them.
                            continue
                        kwargs = {}
                        if loss == 'labels':
                            # Logging is enabled only for the last layer
                            kwargs['log'] = False
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

            if 'enc_outputs' in outputs:
                enc_outputs = outputs['enc_outputs']
                bin_targets = copy.deepcopy(targets)
                for bt in bin_targets:
                    bt['labels'] = torch.zeros_like(bt['labels'])
                indices = self.matcher(enc_outputs, bin_targets, group_detr=group_detr)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'da_output' in outputs:
            losses['loss_domain'] = self.loss_domains(outputs['da_output'], domain_label)
            
        if 'mae_output' in outputs:
            losses['loss_mae'] = self.loss_mae(outputs['mae_output'])

        return losses


@torch.no_grad()
def post_process(pred_logits, pred_boxes, image_sizes, topk=100):
    assert len(pred_logits) == len(image_sizes)
    assert image_sizes.shape[1] == 2
    prob = pred_logits.sigmoid()
    prob = prob.view(pred_logits.shape[0], -1)
    topk_values, topk_indexes = torch.topk(prob, topk, dim=1)
    topk_boxes = torch.div(topk_indexes, pred_logits.shape[2], rounding_mode='trunc')
    labels = topk_indexes % pred_logits.shape[2]
    boxes = box_ops.box_cxcywh_to_xyxy(pred_boxes)
    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
    # From relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = image_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]
    results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(topk_values, labels, boxes)]
    return results


def get_pseudo_labels(pred_logits, pred_boxes, thresholds, nms_threshold=0.7):
    probs = pred_logits.sigmoid()
    scores_batch, labels_batch = torch.max(probs, dim=-1)
    pseudo_labels = []
    thresholds_tensor = torch.tensor(thresholds, device=pred_logits.device)
    for scores, labels, pred_box in zip(scores_batch, labels_batch, pred_boxes):
        larger_idx = torch.gt(scores, thresholds_tensor[labels]).nonzero()[:, 0]
        scores, labels, boxes = scores[larger_idx], labels[larger_idx], pred_box[larger_idx, :]
        nms_idx = nms(box_ops.box_cxcywh_to_xyxy(boxes), scores, iou_threshold=nms_threshold)
        scores, labels, boxes = scores[nms_idx], labels[nms_idx], boxes[nms_idx, :]
        pseudo_labels.append({'scores': scores, 'labels': labels, 'boxes': boxes})
    return pseudo_labels

