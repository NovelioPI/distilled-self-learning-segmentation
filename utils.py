import torch
from torch.nn import CrossEntropyLoss, KLDivLoss
from segmentation_models_pytorch.losses import DiceLoss
from segmentation_models_pytorch.metrics import get_stats, accuracy, iou_score

_ce = CrossEntropyLoss(ignore_index=255)
_kl = KLDivLoss(reduction='batchmean')
_dice = DiceLoss(mode='multiclass', ignore_index=255)

def compute_loss(student_output, hard_labels, soft_labels, use_kd=True, alpha=0.5):
    ce_loss = _ce(student_output, hard_labels)
    dice_loss = _dice(student_output, hard_labels)
    hard_loss = ce_loss + dice_loss
    
    if use_kd:
        log_probs_student = torch.log_softmax(student_output, dim=1)
        soft_loss = _kl(log_probs_student, soft_labels)
        loss = alpha * soft_loss + (1 - alpha) * hard_loss
    else:
        loss = hard_loss
        
    return loss

def compute_metrics(preds, labels, num_classes=19, ignore_index=255):
    tp, fp, tn, fn = get_stats(preds, labels, mode='multiclass', num_classes=num_classes, ignore_index=ignore_index)
        
    # Micro: Compute metric across all pixels and classes at once.
    acc_micro = accuracy(tp, fp, tn, fn, reduction='micro')
    iou_micro = iou_score(tp, fp, tn, fn, reduction='micro')
    
    # Micro-imagewise: Compute metric for each image, then average over all images.
    acc_micro_imagewise = accuracy(tp, fp, tn, fn, reduction='micro-imagewise')
    iou_micro_imagewise = iou_score(tp, fp, tn, fn, reduction='micro-imagewise')
    
    # Macro: Compute metric per class, then average over all classes.
    acc_macro = accuracy(tp, fp, tn, fn, reduction='macro')
    iou_macro = iou_score(tp, fp, tn, fn, reduction='micro')
    
    # Macro-imagewise: Compute metric per class per image, then average over all images and classes.
    acc_macro_imagewise = accuracy(tp, fp, tn, fn, reduction='macro-imagewise')
    iou_macro_imagewise = iou_score(tp, fp, tn, fn, reduction='macro-imagewise')
    
    return {
        'acc_micro': acc_micro,
        'iou_micro': iou_micro,
        'acc_micro_imagewise': acc_micro_imagewise,
        'iou_micro_imagewise': iou_micro_imagewise,
        'acc_macro': acc_macro,
        'iou_macro': iou_macro,
        'acc_macro_imagewise': acc_macro_imagewise,
        'iou_macro_imagewise': iou_macro_imagewise,
    }