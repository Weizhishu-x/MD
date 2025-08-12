import os
import time
import datetime
import json
import torch
from torch.utils.data import DataLoader

from datasets.coco_style_dataset import DataPreFetcher
from datasets.coco_eval import CocoEvaluator
from pathlib import Path

from models.criterion import post_process, get_pseudo_labels
from utils.distributed_utils import is_main_process
from utils.box_utils import box_cxcywh_to_xyxy, convert_to_xywh
from collections import defaultdict
from typing import List
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from collections import defaultdict



def train_one_epoch_standard(model: torch.nn.Module,
                             criterion: torch.nn.Module,
                             data_loader: DataLoader,
                             optimizer: torch.optim.Optimizer,
                             device: torch.device,
                             epoch: int,
                             clip_max_norm: float = 0.0,
                             print_freq: int = 20,
                             flush: bool = True):
    start_time = time.time()
    model.train()
    criterion.train()
    fetcher = DataPreFetcher(data_loader, device=device)
    samples, annotations = fetcher.next()
    # Training statistics
    epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)
    epoch_loss_dict = defaultdict(float)
    for i in range(len(data_loader)):
        # Forward
        out = model(samples)
        # Loss
        loss_dict = criterion(out, annotations)
        weight_dict = criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        # Record loss
        epoch_loss += loss.detach()
        for k, v in loss_dict.items():
            epoch_loss_dict[k] += v.detach().cpu().item()
        # Data pre-fetch
        samples, annotations = fetcher.next()
        # Log
        if is_main_process() and (i + 1) % print_freq == 0:
            print('Training epoch ' + str(epoch) + ' : [ ' + str(i + 1) + '/' + str(len(data_loader)) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)
    # Final process of training statistic
    epoch_loss /= len(data_loader)
    for k, v in epoch_loss_dict.items():
        epoch_loss_dict[k] /= len(data_loader)
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Training epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
          ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()), flush=flush)
    return epoch_loss, epoch_loss_dict


def train_one_epoch_with_mae(model: torch.nn.Module,
                             criterion: torch.nn.Module,
                             criterion_mae: torch.nn.Module,
                             source_loader: DataLoader,
                             target_loader: DataLoader,
                             coef_target: float,
                             optimizer: torch.optim.Optimizer,
                             device: torch.device,
                             epoch: int,
                             clip_max_norm: float = 0.0,
                             print_freq: int = 20,
                             flush: bool = True):
    start_time = time.time()
    model.train()
    criterion.train()
    criterion_mae.train()
    weight_dict = criterion.weight_dict
    source_fetcher = DataPreFetcher(source_loader, device=device)
    target_fetcher = DataPreFetcher(target_loader, device=device)
    source_samples, source_annotations = source_fetcher.next()
    target_samples, _ = target_fetcher.next()
    # Training statistics
    epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)
    epoch_loss_dict = defaultdict(float)
    total_iters = min(len(source_loader), len(target_loader))
    for i in range(total_iters):
        optimizer.zero_grad()
        
        # Source forward and backward
        out = model(source_samples)
        loss_dict = criterion(out, source_annotations)
        loss_det = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_det.backward()
        
        # Target forward and backward
        out_mae = model(target_samples, enable_mae=True, mask_ratio=0.75)
        loss_dict_mae = criterion_mae(out_mae)
        loss_mae = sum(loss_dict_mae[k] * weight_dict[k] for k in loss_dict_mae.keys() if k in weight_dict)
        (loss_mae * coef_target).backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        
        # Record loss
        loss = loss_det.detach() + loss_mae.detach()  * coef_target
        loss_dict['loss_mae'] = loss_dict_mae['loss_mae']
        epoch_loss += loss
        for k, v in loss_dict.items():
            epoch_loss_dict[k] += v.detach().cpu().item()
        # Data pre-fetch
        source_samples, source_annotations = source_fetcher.next()
        target_samples, _ = target_fetcher.next()
        # Log
        if is_main_process() and (i + 1) % print_freq == 0:
            print('Cross-domain MAE training epoch ' + str(epoch) + ' : [ ' + str(i + 1) + '/' +
                  str(total_iters) + ' ] ' + 'total loss: ' + str(loss.cpu().numpy()), flush=flush)
    # Final process of training statistic
    epoch_loss /= total_iters
    for k, v in epoch_loss_dict.items():
        epoch_loss_dict[k] /= total_iters
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Cross-domain MAE training epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
          ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()), flush=flush)
    return epoch_loss, epoch_loss_dict


def train_one_epoch_with_mae_bk(model: torch.nn.Module,
                             criterion: torch.nn.Module,
                             criterion_mae: torch.nn.Module,
                             source_loader: DataLoader,
                             target_loader: DataLoader,
                             coef_target: float,
                             optimizer: torch.optim.Optimizer,
                             device: torch.device,
                             epoch: int,
                             clip_max_norm: float = 0.0,
                             print_freq: int = 20,
                             flush: bool = True):
    start_time = time.time()
    model.train()
    criterion.train()
    criterion_mae.train()
    source_fetcher = DataPreFetcher(source_loader, device=device)
    target_fetcher = DataPreFetcher(target_loader, device=device)
    source_samples, source_annotations = source_fetcher.next()
    target_samples, _ = target_fetcher.next()
    # Training statistics
    epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)
    epoch_loss_dict = defaultdict(float)
    total_iters = min(len(source_loader), len(target_loader))
    for i in range(total_iters):
        # Source forward
        out = model(source_samples)
        # Target forward
        # mask_ratio = 0.7 + 0.02 * epoch if epoch <= 10 else 0.9
        out_mae = model(target_samples, enable_mae=True, mask_ratio=0.75)
        # Loss
        loss_dict = criterion(out, source_annotations)
        loss_dict_mae = criterion_mae(out_mae)
        weight_dict = criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_mae = sum(loss_dict_mae[k] * weight_dict[k] for k in loss_dict_mae.keys() if k in weight_dict)
        loss += loss_mae * coef_target 
        loss_dict['loss_mae'] = loss_dict_mae['loss_mae']
        # Backward
        optimizer.zero_grad()
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        # Record loss
        epoch_loss += loss.detach()
        for k, v in loss_dict.items():
            epoch_loss_dict[k] += v.detach().cpu().item()
        # Data pre-fetch
        source_samples, source_annotations = source_fetcher.next()
        target_samples, _ = target_fetcher.next()
        # Log
        if is_main_process() and (i + 1) % print_freq == 0:
            print('Cross-domain MAE training epoch ' + str(epoch) + ' : [ ' + str(i + 1) + '/' +
                  str(total_iters) + ' ] ' + 'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)
    # Final process of training statistic
    epoch_loss /= total_iters
    for k, v in epoch_loss_dict.items():
        epoch_loss_dict[k] /= total_iters
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Cross-domain MAE training epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
          ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()), flush=flush)
    return epoch_loss, epoch_loss_dict


def train_one_epoch_teaching(student_model: torch.nn.Module,
                             teacher_model: torch.nn.Module,
                             criterion: torch.nn.Module,
                             criterion_pseudo: torch.nn.Module,
                             source_loader: DataLoader,
                             target_loader: DataLoader,
                             optimizer: torch.optim.Optimizer,
                             thresholds: List[float],
                             coef_target: float,
                             alpha_ema: float,
                             device: torch.device,
                             epoch: int,
                             enable_mae: bool = False,
                             clip_max_norm: float = 0.0,
                             print_freq: int = 20,
                             flush: bool = True):
    start_time = time.time()
    student_model.train()
    teacher_model.train()
    criterion.train()
    criterion_pseudo.train()
    source_weight_dict = criterion.weight_dict
    target_weight_dict = criterion_pseudo.weight_dict
    source_fetcher = DataPreFetcher(source_loader, device=device)
    target_fetcher = DataPreFetcher(target_loader, device=device)
    source_sammples, source_annotations = source_fetcher.next()
    target_images, _ = target_fetcher.next()
    target_teacher_images, target_student_images = target_images[0], target_images[1]
    # Record epoch losses
    epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)
    # Training data statistics
    epoch_source_loss_dict = defaultdict(float)
    epoch_target_loss_dict = defaultdict(float)
    total_iters = min(len(source_loader), len(target_loader))
    for i in range(total_iters):
        optimizer.zero_grad()
        # Source forward
        source_out = student_model(source_sammples)
        source_loss_dict = criterion(source_out, source_annotations, domain_label=0)
        source_loss = sum(source_loss_dict[k] * source_weight_dict[k] for k in source_loss_dict.keys() if k in source_weight_dict)
        source_loss.backward()
        # Target teacher forward
        with torch.no_grad():
            teacher_out = teacher_model(target_teacher_images)
            pseudo_labels = get_pseudo_labels(teacher_out['pred_logits'], teacher_out['pred_boxes'], thresholds)
        # Target student forward
        target_student_out = student_model(target_student_images, enable_mae)
        target_loss_dict = criterion_pseudo(target_student_out, pseudo_labels, domain_label=1)
        target_loss = sum(target_loss_dict[k] * target_weight_dict[k] for k in target_loss_dict.keys() if k in target_weight_dict)        
        (coef_target * target_loss).backward()
        # Backward
        loss = source_loss.detach() + coef_target * target_loss.detach()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), clip_max_norm)
        optimizer.step()
        # Record epoch losses
        epoch_loss += loss.detach()
        # update loss_dict
        for k, v in source_loss_dict.items():
            epoch_source_loss_dict[k] += v.detach().cpu().item()
        for k, v in target_loss_dict.items():
            epoch_target_loss_dict[k] += v.detach().cpu().item()
        # EMA update teacher
        with torch.no_grad():
            state_dict, student_state_dict = teacher_model.state_dict(), student_model.state_dict()
            for key, value in state_dict.items():
                state_dict[key] = alpha_ema * value + (1 - alpha_ema) * student_state_dict[key].detach()
            teacher_model.load_state_dict(state_dict)
        # Data pre-fetch
        source_sammples, source_annotations = source_fetcher.next()
        target_images, _ = target_fetcher.next()
        if target_images is not None:
            target_teacher_images, target_student_images = target_images[0], target_images[1]
        # Log
        if is_main_process() and (i + 1) % print_freq == 0:
            print('Teaching epoch ' + str(epoch) + ' : [ ' + str(i + 1) + '/' + str(total_iters) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)
        # break
    # Final process of loss dict
    epoch_loss /= total_iters
    for k, v in epoch_source_loss_dict.items():
        epoch_source_loss_dict[k] /= total_iters
    for k, v in epoch_target_loss_dict.items():
        epoch_target_loss_dict[k] /= total_iters
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Teaching epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
          ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()), flush=flush)
    return epoch_loss, epoch_source_loss_dict, epoch_target_loss_dict


def train_one_epoch_teaching_bk(student_model: torch.nn.Module,
                             teacher_model: torch.nn.Module,
                             criterion: torch.nn.Module,
                             criterion_pseudo: torch.nn.Module,
                             source_loader: DataLoader,
                             target_loader: DataLoader,
                             optimizer: torch.optim.Optimizer,
                             thresholds: List[float],
                             coef_target: float,
                             alpha_ema: float,
                             device: torch.device,
                             epoch: int,
                             enable_mae: bool = False,
                             clip_max_norm: float = 0.0,
                             print_freq: int = 20,
                             flush: bool = True):
    start_time = time.time()
    student_model.train()
    teacher_model.train()
    criterion.train()
    criterion_pseudo.train()
    source_fetcher = DataPreFetcher(source_loader, device=device)
    target_fetcher = DataPreFetcher(target_loader, device=device)
    source_sammples, source_annotations = source_fetcher.next()
    target_images, _ = target_fetcher.next()
    target_teacher_images, target_student_images = target_images[0], target_images[1]
    # Record epoch losses
    epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)
    # Training data statistics
    epoch_source_loss_dict = defaultdict(float)
    epoch_target_loss_dict = defaultdict(float)
    total_iters = min(len(source_loader), len(target_loader))
    for i in range(total_iters):
        # Source forward
        source_out = student_model(source_sammples)
        source_loss_dict = criterion(source_out, source_annotations, domain_label=0)
        source_weight_dict = criterion.weight_dict
        source_loss = sum(source_loss_dict[k] * source_weight_dict[k] for k in source_loss_dict.keys() if k in source_weight_dict)
        # Target teacher forward
        with torch.no_grad():
            teacher_out = teacher_model(target_teacher_images)
            pseudo_labels = get_pseudo_labels(teacher_out['pred_logits'], teacher_out['pred_boxes'], thresholds)
        # Target student forward
        target_student_out = student_model(target_student_images, enable_mae)
        target_loss_dict = criterion_pseudo(target_student_out, pseudo_labels, domain_label=1)
        target_weight_dict = criterion_pseudo.weight_dict
        target_loss = sum(target_loss_dict[k] * target_weight_dict[k] for k in target_loss_dict.keys() if k in target_weight_dict)        
        # Backward
        optimizer.zero_grad()
        loss = source_loss + coef_target * target_loss
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), clip_max_norm)
        optimizer.step()
        # Record epoch losses
        epoch_loss += loss.detach()
        # update loss_dict
        for k, v in source_loss_dict.items():
            epoch_source_loss_dict[k] += v.detach().cpu().item()
        for k, v in target_loss_dict.items():
            epoch_target_loss_dict[k] += v.detach().cpu().item()
        # EMA update teacher
        with torch.no_grad():
            state_dict, student_state_dict = teacher_model.state_dict(), student_model.state_dict()
            for key, value in state_dict.items():
                state_dict[key] = alpha_ema * value + (1 - alpha_ema) * student_state_dict[key].detach()
            teacher_model.load_state_dict(state_dict)
        # Data pre-fetch
        source_sammples, source_annotations = source_fetcher.next()
        target_images, _ = target_fetcher.next()
        if target_images is not None:
            target_teacher_images, target_student_images = target_images[0], target_images[1]
        # Log
        if is_main_process() and (i + 1) % print_freq == 0:
            print('Teaching epoch ' + str(epoch) + ' : [ ' + str(i + 1) + '/' + str(total_iters) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)
        # break
    # Final process of loss dict
    epoch_loss /= total_iters
    for k, v in epoch_source_loss_dict.items():
        epoch_source_loss_dict[k] /= total_iters
    for k, v in epoch_target_loss_dict.items():
        epoch_target_loss_dict[k] /= total_iters
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Teaching epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
          ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()), flush=flush)
    return epoch_loss, epoch_source_loss_dict, epoch_target_loss_dict


@torch.no_grad()
def evaluate(model: torch.nn.Module,
             criterion: torch.nn.Module,
             data_loader_val: DataLoader,
             device: torch.device,
             print_freq: int,
             output_result_labels: bool = False,
             visualization_path: str = None,
             flush: bool = False):
    start_time = time.time()
    model.eval()
    criterion.eval()
    if hasattr(data_loader_val.dataset, 'coco') or hasattr(data_loader_val.dataset, 'anno_file'):
        evaluator = CocoEvaluator(data_loader_val.dataset.coco)
        coco_data = json.load(open(data_loader_val.dataset.anno_file, 'r'))
        # dataset_annotations = [[] for _ in range(len(coco_data['images']) + 1)]
        dataset_annotations = defaultdict(list)
    else:
        raise ValueError('Unsupported dataset type.')
    epoch_loss = 0.0
    for i, (images, annotations) in enumerate(data_loader_val):
        # To CUDA
        images = images.to(device)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        # Forward
        out = model(images)
        logits_all, boxes_all = out['pred_logits'], out['pred_boxes']
        # Get pseudo labels
        if output_result_labels:
            results = get_pseudo_labels(logits_all, boxes_all, [0.4 for _ in range(9)])
            for anno, res in zip(annotations, results):
                image_id = anno['image_id'].item()
                orig_image_size = anno['orig_size']
                img_h, img_w = orig_image_size.unbind(0)
                scale_fct = torch.stack([img_w, img_h, img_w, img_h])
                converted_boxes = convert_to_xywh(box_cxcywh_to_xyxy(res['boxes'] * scale_fct))
                converted_boxes = converted_boxes.detach().cpu().numpy().tolist()
                for label, box in zip(res['labels'].detach().cpu().numpy().tolist(), converted_boxes):
                    pseudo_anno = {
                        'id': 0,
                        'image_id': image_id,
                        'category_id': label,
                        'iscrowd': 0,
                        'area': box[-2] * box[-1],
                        'bbox': box
                    }
                    dataset_annotations[image_id].append(pseudo_anno)
            # Visualize pseudo labels
            if visualization_path:
                if i == 0: 
                    base_ds = data_loader_val.dataset.coco
                    image_root = data_loader_val.dataset.root
                    cat_ids = base_ds.getCatIds()
                    cats = base_ds.loadCats(cat_ids)
                    class_names = {cat['id']: cat['name'] for cat in cats}
                    os.makedirs(visualization_path, exist_ok=True)
                    print(f"可视化结果将保存到: {visualization_path}")

                for img_idx in range(len(images.tensors)):
                    image_id = annotations[img_idx]['image_id'].item()
                    file_name = base_ds.loadImgs(image_id)[0]['file_name']
                    image_path = Path(image_root) / file_name              
                    save_visualization_image(
                        image_path=str(image_path),
                        gt_anno=annotations[img_idx],
                        pred_anno=results[img_idx],
                        class_names=class_names,
                        save_dir=visualization_path,
                        device=device
                    )
        # Loss
        loss_dict = criterion(out, annotations)
        weight_dict = criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        epoch_loss += loss
        if is_main_process() and (i + 1) % print_freq == 0:
            print('Evaluation : [ ' + str(i + 1) + '/' + str(len(data_loader_val)) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)
        # mAP
        orig_image_sizes = torch.stack([anno['orig_size'] for anno in annotations], dim=0)
        results = post_process(logits_all, boxes_all, orig_image_sizes, 100)
        results = {anno['image_id'].item(): res for anno, res in zip(annotations, results)}
        evaluator.update(results)
        
        # break
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    aps = evaluator.summarize()
    epoch_loss /= len(data_loader_val)
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Evaluation finished. Time cost: ' + total_time_str, flush=flush)
    # Save results
    if output_result_labels:
        dataset_annotations_return = []
        id_cnt = 0
        for image_anno in dataset_annotations.values():
            for box_anno in image_anno:
                box_anno['id'] = id_cnt
                id_cnt += 1
                dataset_annotations_return.append(box_anno)
        coco_data['annotations'] = dataset_annotations_return
        return aps, epoch_loss / len(data_loader_val), coco_data
    return aps, epoch_loss / len(data_loader_val)


def set_dropout_rate(model: torch.nn.Module, dropout_rate: float):
    """递归地设置模型中所有Dropout层的p值。"""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = dropout_rate


def save_visualization_image(image_path: str,
                             gt_anno: dict,
                             pred_anno: dict,
                             class_names: dict,
                             save_dir: str,
                             device: torch.device,
                             iou_threshold=0.5
                             ):
    """
    通过分类展示 TP, FP, FN 来优化密集场景的可视化。
    """
    original_image = Image.open(image_path).convert('RGB')
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    orig_h, orig_w = gt_anno['orig_size'].cpu().numpy()
    
    fig, ax = plt.subplots(1, figsize=(16, 12))
    ax.imshow(original_image)
    ax.axis('off')

    # --- 1. 坐标转换 ---
    scale_fct = torch.tensor([orig_w, orig_h, orig_w, orig_h], device=device)
    gt_boxes = (box_cxcywh_to_xyxy(gt_anno['boxes']) * scale_fct).cpu().numpy()
    gt_labels = gt_anno['labels'].cpu().numpy()
    pred_boxes = (box_cxcywh_to_xyxy(pred_anno['boxes']) * scale_fct).cpu().numpy()
    pred_labels = pred_anno['labels'].cpu().numpy()
    pred_scores = pred_anno['scores'].cpu().numpy()

    # --- 2. 匹配GT和预测 ---
    gt_matched = [False] * len(gt_boxes)
    pred_matched = [False] * len(pred_boxes)

    for i, p_box in enumerate(pred_boxes):
        best_iou = 0
        best_gt_idx = -1
        for j, g_box in enumerate(gt_boxes):
            # 必须是同类别且未被匹配过的GT
            if pred_labels[i] == gt_labels[j] and not gt_matched[j]:
                iou = calculate_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
        
        if best_iou > iou_threshold:
            gt_matched[best_gt_idx] = True
            pred_matched[i] = True

    # --- 3. 绘制结果 ---
    # a. 绘制正确检测 (TP) - 绿色实线
    for i, p_box in enumerate(pred_boxes):
        if pred_matched[i]:
            class_name = class_names.get(pred_labels[i], "N/A")
            xmin, ymin, xmax, ymax = p_box
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=1.5, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin - 2, f'{class_name} ({pred_scores[i]:.2f})',
                    bbox=dict(facecolor='lime', alpha=0.7, pad=1), color='black', fontsize=7)

    # b. 绘制漏报 (FN) - 黄色虚线
    for i, g_box in enumerate(gt_boxes):
        if not gt_matched[i]:
            class_name = class_names.get(gt_labels[i], "N/A")
            xmin, ymin, xmax, ymax = g_box
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=1.5, edgecolor='yellow', facecolor='none', linestyle='--')
            ax.add_patch(rect)
            ax.text(xmin, ymin - 2, f'GT: {class_name}',
                    bbox=dict(facecolor='yellow', alpha=0.7, pad=1), color='black', fontsize=7)

    # c. 绘制误报 (FP) - 红色虚线
    for i, p_box in enumerate(pred_boxes):
        if not pred_matched[i]:
            class_name = class_names.get(pred_labels[i], "N/A")
            xmin, ymin, xmax, ymax = p_box
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=1.5, edgecolor='red', facecolor='none', linestyle='--')
            ax.add_patch(rect)
            ax.text(xmin, ymin - 2, f'Pred: {class_name} ({pred_scores[i]:.2f})',
                    bbox=dict(facecolor='red', alpha=0.7, pad=1), color='white', fontsize=7)
    
    # ... (添加图例) ...
    save_path = os.path.join(save_dir, f"{image_name}.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=200)
    plt.close(fig)


# def save_visualization_image(image_path: str,
#                              gt_anno: dict,
#                              pred_anno: dict,
#                              class_names: dict,
#                              save_dir: str,
#                              device: torch.device,
#                              iou_threshold=0.5
#                              ):
#     """
#     - 颜色表示类别。
#     - 线条样式表示检测状态 (TP, FP, FN)。
#     - 在右下角添加图例。
#     """
#     original_image = Image.open(image_path).convert('RGB')
#     image_name = os.path.splitext(os.path.basename(image_path))[0]
#     orig_h, orig_w = gt_anno['orig_size'].cpu().numpy()

#     fig, ax = plt.subplots(1, figsize=(16, 12))
#     ax.imshow(original_image)
#     ax.axis('off')

#     # --- 0. 获取类别颜色映射 ---
#     class_colors = get_class_colors(class_names)

#     # --- 1. 坐标转换 ---
#     scale_fct = torch.tensor([orig_w, orig_h, orig_w, orig_h], device=device)
#     gt_boxes = (box_cxcywh_to_xyxy(gt_anno['boxes']) * scale_fct).cpu().numpy()
#     gt_labels = gt_anno['labels'].cpu().numpy()
#     pred_boxes = (box_cxcywh_to_xyxy(pred_anno['boxes']) * scale_fct).cpu().numpy()
#     pred_labels = pred_anno['labels'].cpu().numpy()
#     pred_scores = pred_anno['scores'].cpu().numpy()

#     # --- 2. 匹配GT和预测 ---
#     pred_to_gt_map = {}
#     matched_gt_indices = set()
#     sorted_pred_indices = sorted(range(len(pred_scores)), key=lambda i: pred_scores[i], reverse=True)

#     for i in sorted_pred_indices:
#         p_box, p_label = pred_boxes[i], pred_labels[i]
#         best_iou, best_gt_idx = 0, -1
#         for j, (g_box, g_label) in enumerate(zip(gt_boxes, gt_labels)):
#             if p_label == g_label and j not in matched_gt_indices:
#                 iou = calculate_iou(p_box, g_box)
#                 if iou > best_iou:
#                     best_iou, best_gt_idx = iou, j
#         if best_iou > iou_threshold:
#             pred_to_gt_map[i] = best_gt_idx
#             matched_gt_indices.add(best_gt_idx)

#     # --- 3. 绘制结果 ---
#     # 定义线条样式
#     styles = {'TP': '-', 'FP': '--', 'FN': ':'}

#     # a. 绘制 TP 和 FP
#     for i, p_box in enumerate(pred_boxes):
#         p_label = pred_labels[i]
#         class_name = class_names.get(p_label, "N/A")
#         color = class_colors.get(p_label, 'gray')
#         score = pred_scores[i]
#         xmin, ymin, xmax, ymax = p_box

#         if i in pred_to_gt_map:
#             status = 'TP'
#             label_text = f'{class_name} ({score:.2f})'
#         else:
#             status = 'FP'
#             label_text = f'{class_name} ({score:.2f})'

#         rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
#                                  linewidth=1, edgecolor=color,
#                                  facecolor='none', linestyle=styles[status])
#         ax.add_patch(rect)
#         ax.text(xmin, ymin - 3, label_text,
#                 color='white',
#                 bbox=dict(facecolor=color, alpha=0.8, pad=1, edgecolor='none'),
#                 fontsize=8, weight='bold')

#     # b. 绘制 FN
#     for i, g_box in enumerate(gt_boxes):
#         if i not in matched_gt_indices:
#             g_label = gt_labels[i]
#             class_name = class_names.get(g_label, "N/A")
#             color = class_colors.get(g_label, 'gray')
#             xmin, ymin, xmax, ymax = g_box
            
#             rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
#                                      linewidth=1, edgecolor=color,
#                                      facecolor='none', linestyle=styles['FN'])
#             ax.add_patch(rect)
#             ax.text(xmin, ymin - 3, f'Missed: {class_name}',
#                     color='white',
#                     bbox=dict(facecolor=color, alpha=0.8, pad=1, edgecolor='none'),
#                     fontsize=8, weight='bold')
                    
#     # --- 4. 添加图例 ---
#     # a. 类别图例
#     class_legend_elements = [patches.Patch(facecolor=class_colors[cid], label=cname)
#                              for cid, cname in class_names.items()]
#     leg1 = ax.legend(handles=class_legend_elements, loc='lower right',
#                      title='(Colors)', fontsize=9, title_fontsize=10,
#                      facecolor='lightgray', framealpha=0.7)

#     # b. 状态图例
#     status_legend_elements = [
#         Line2D([0], [0], color='black', lw=2, linestyle=styles['TP'], label='(TP)'),
#         Line2D([0], [0], color='black', lw=2, linestyle=styles['FP'], label='(FP)'),
#         Line2D([0], [0], color='black', lw=2, linestyle=styles['FN'], label='(FN)')
#     ]
#     ax.legend(handles=status_legend_elements, loc='lower left',
#               title='(Line Styles)', fontsize=9, title_fontsize=10,
#               facecolor='lightgray', framealpha=0.7)

#     # 必须重新添加第一个图例，因为第二个会覆盖它
#     ax.add_artist(leg1)

#     save_path = os.path.join(save_dir, f"{image_name}_analyzed.png")
#     plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=200)
#     plt.close(fig)


# def get_class_colors(class_names: dict) -> dict:
#     """为每个类别动态生成一个颜色。"""
#     # 使用 matplotlib 的 'tab10' colormap，它有10种清晰可辨的颜色
#     # 如果类别多于10个，颜色会循环使用
#     colors = plt.get_cmap('tab10')(np.linspace(0, 1, 10))
#     class_colors = {}
#     sorted_class_ids = sorted(class_names.keys())
#     for i, class_id in enumerate(sorted_class_ids):
#         class_colors[class_id] = colors[i % len(colors)]
#     return class_colors


def calculate_iou(box1, box2):
    """计算两个边界框的IoU。box格式为 (xmin, ymin, xmax, ymax)。"""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area