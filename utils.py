import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, KLDivLoss
from segmentation_models_pytorch.losses import DiceLoss
from segmentation_models_pytorch.metrics import get_stats, accuracy, iou_score, f1_score
import segmentation_models_pytorch as smp

def build_student_model(encoder='timm-efficientnet-b0', decoder='unet', weight=None, dropout=0.0):
    if decoder == 'unet':
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=weight if weight else None,
            in_channels=3,
            classes=19,
        )
    elif decoder == 'fpn':
        model = smp.FPN(
            encoder_name=encoder,
            encoder_weights=weight if weight else None,
            in_channels=3,
            classes=19,
        )
    elif decoder == 'deeplabv3':
        model = smp.DeepLabV3(
            encoder_name=encoder,
            encoder_weights=weight if weight else None,
            in_channels=3,
            classes=19,
        )
    elif decoder == 'segformer':
        model = smp.Segformer(
            encoder_name=encoder,
            encoder_weights=weight if weight else None,
            in_channels=3,
            classes=19,
        )
    else:
        raise ValueError(f"Unsupported decoder: {decoder}. Supported decoders are 'unet', 'fpn', 'deeplabv3', and 'segformer'.")
    
    if dropout > 0:
        model.decoder.dropout = nn.Dropout(dropout)
    
    return model

def compute_loss(student_logits, hard_labels, teacher_logits=None, use_kd=True, kd_weight=0.5, T=2.0):
    ce = CrossEntropyLoss(ignore_index=255)
    dice = DiceLoss(mode='multiclass', ignore_index=255)
    kl = KLDivLoss(reduction='batchmean')
    
    ce_loss = ce(student_logits, hard_labels)
    dice_loss = dice(student_logits, hard_labels)
    seg_loss = ce_loss + dice_loss
    
    if use_kd:
        log_probs_student = torch.log_softmax(student_logits / T, dim=1)
        probs_teacher = torch.softmax(teacher_logits / T, dim=1)
        soft_loss = kl(log_probs_student, probs_teacher) * (T ** 2)
        loss = kd_weight * soft_loss + seg_loss
    else:
        loss = seg_loss
        
    return {
        'total': loss,
        'seg': seg_loss,
        'kd': soft_loss if use_kd else None
    }
    
def negative_sampling_loss(student_logits, teacher_probs, confidence_threshold=0.5, topk=3):
    max_conf, _ = torch.max(teacher_probs, dim=1)
    ambigous_mask = max_conf <= confidence_threshold
    
    topk_vals, topk_indices = teacher_probs.topk(topk, dim=1)
    
    loss = 0.0
    count = 0
    for k in range(topk):
        neg_class = topk_indices[:, k]
        masked_indices = ambigous_mask.nonzero(as_tuple=False)
        
        if masked_indices.numel() == 0:
            continue
        
        batch_idx, h_idx, w_idx = masked_indices[:, 0], masked_indices[:, 1], masked_indices[:, 2]
        student_logits_flat = student_logits[batch_idx, :, h_idx, w_idx]
        neg_class_flat = neg_class[batch_idx, h_idx, w_idx]
        
        neg_loss = F.cross_entropy(student_logits_flat, neg_class_flat, reduction='mean')
        loss += neg_loss
        count += 1
    
    if count > 0:
        loss /= count
    else:
        loss = torch.tensor(0.0, device=student_logits.device)
    
    return loss

def compute_metrics(preds, labels, num_classes=19, ignore_index=255):
    tp, fp, tn, fn = get_stats(preds, labels, mode='multiclass', num_classes=num_classes, ignore_index=ignore_index)
        
    # Micro: Compute metric across all pixels and classes at once.
    acc_micro = accuracy(tp, fp, tn, fn, reduction='micro')
    iou_micro = iou_score(tp, fp, tn, fn, reduction='micro')
    f1_micro = f1_score(tp, fp, tn, fn, reduction='micro')
    
    # Micro-imagewise: Compute metric for each image, then average over all images.
    acc_micro_imagewise = accuracy(tp, fp, tn, fn, reduction='micro-imagewise')
    iou_micro_imagewise = iou_score(tp, fp, tn, fn, reduction='micro-imagewise')
    f1_micro_imagewise = f1_score(tp, fp, tn, fn, reduction='micro-imagewise')
    
    # Macro: Compute metric per class, then average over all classes.
    acc_macro = accuracy(tp, fp, tn, fn, reduction='macro')
    iou_macro = iou_score(tp, fp, tn, fn, reduction='micro')
    f1_macro = f1_score(tp, fp, tn, fn, reduction='macro')
    
    # Macro-imagewise: Compute metric per class per image, then average over all images and classes.
    acc_macro_imagewise = accuracy(tp, fp, tn, fn, reduction='macro-imagewise')
    iou_macro_imagewise = iou_score(tp, fp, tn, fn, reduction='macro-imagewise')
    f1_macro_imagewise = f1_score(tp, fp, tn, fn, reduction='macro-imagewise')
    
    return {
        'acc': {
            'micro': acc_micro,
            'micro_imagewise': acc_micro_imagewise,
            'macro': acc_macro,
            'macro_imagewise': acc_macro_imagewise
        },
        'iou': {
            'micro': iou_micro,
            'micro_imagewise': iou_micro_imagewise,
            'macro': iou_macro,
            'macro_imagewise': iou_macro_imagewise
        },
        'f1': {
            'micro': f1_micro,
            'micro_imagewise': f1_micro_imagewise,
            'macro': f1_macro,
            'macro_imagewise': f1_macro_imagewise
        }
    }
    
    
def colorize_segmentation(image, num_classes=19):
    color_map = torch.tensor([
        [128, 64,128], [244, 35,232], [ 70, 70, 70], [102,102,156],
        [190,153,153], [153,153,153], [250,170, 30], [220,220,  0],
        [107,142, 35], [152,251,152], [ 70,130,180], [220, 20, 60],
        [255,  0,  0], [  0,  0,142], [  0,  0, 70], [  0, 60,100],
        [  0, 80,100], [  0,  0,230], [119, 11, 32]  # 19 classes
    ], dtype=torch.uint8)
    
    h, w = image.shape
    color_image = torch.zeros((h, w, 3), dtype=torch.uint8, device=image.device)
    for c in range(num_classes):
        mask = image == c
        color_image[mask] = color_map[c]
    return color_image.permute(2, 0, 1)