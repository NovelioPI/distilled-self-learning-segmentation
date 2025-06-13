import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, KLDivLoss
from segmentation_models_pytorch.losses import DiceLoss
from segmentation_models_pytorch.metrics import get_stats, accuracy, iou_score, f1_score
import segmentation_models_pytorch as smp
import itertools
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import numpy as np



def build_hyperparameters(params: dict):
    all_combinations = list(itertools.product(*params.values()))
    keys = params.keys()
    hyperparams_list = [dict(zip(keys, values)) for values in all_combinations]
    return hyperparams_list

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

def compute_loss(student_logits, hard_labels, teacher_logits=None, use_kd=False, kd_weight=0.5, T=2.0):
    ce = CrossEntropyLoss(ignore_index=255)
    dice = DiceLoss(mode='multiclass', ignore_index=255)
    kl = KLDivLoss(reduction='batchmean')
    
    ce_loss = ce(student_logits, hard_labels)
    dice_loss = dice(student_logits, hard_labels)
    seg_loss = ce_loss + dice_loss
    
    if use_kd and teacher_logits is not None:
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
    
def negative_sampling_loss(student_logits, teacher_probs, weight, threshold=0.5, topk=2):
    if weight == 0.0:
        return student_logits.new_zeros(())

    max_conf, _ = teacher_probs.max(dim=1)            # [B,H,W]
    ambiguous = max_conf < threshold                  # bool mask
    if not ambiguous.any():
        return student_logits.new_zeros(())

    # gather top‑k indices per pixel
    topk_vals, topk_inds = teacher_probs.topk(topk, dim=1)  # [B,k,H,W]
    # convert student logits for fancy indexing
    student_logit_perm = student_logits.permute(0, 2, 3, 1)  # [B,H,W,C]

    total_loss, count = 0.0, 0
    for k in range(topk):
        neg_cls = topk_inds[:, k, :, :]                    # [B,H,W]
        mask_k = ambiguous & (neg_cls != student_logits.argmax(1))
        if mask_k.any():
            # gather logits for negative classes only where mask_k is true
            selected_logits = student_logit_perm[mask_k, neg_cls[mask_k]]
            neg_labels = torch.zeros_like(selected_logits, dtype=torch.long)
            loss_k = F.binary_cross_entropy_with_logits(selected_logits, neg_labels.float())
            total_loss += loss_k
            count += 1
    if count == 0:
        return student_logits.new_zeros(())
    return weight * total_loss / count

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

def entropy(probs):
    """
    Compute the entropy of the predicted probabilities.
    
    Args:
        probs (torch.Tensor): Predicted probabilities of shape (C, H, W) or (B, C, H, W).
        
    Returns:
        torch.Tensor: Entropy of shape (H, W) or (B, H, W).
    """
    if probs.dim() == 3:  # (C, H, W)
        dim = 0
    elif probs.dim() == 4:  # (B, C, H, W)
        dim = 1
    return -torch.sum(probs * torch.log(probs + 1e-10), dim=dim)  # Avoid log(0) with small epsilon

def normalize_entropy(probs):
    """
    Normalize the entropy values to the range [0, 1].
    
    Args:
        probs (torch.Tensor): Predicted probabilities of shape (C, H, W) or (B, C, H, W).
        
    Returns:
        torch.Tensor: Normalized entropy of shape (H, W) or (B, H, W).
    """
    if probs.dim() == 3:  # (C, H, W)
        dim = 0
    elif probs.dim() == 4:  # (B, C, H, W)
        dim = 1
    ent = entropy(probs)
    max_ent = torch.log(torch.tensor(probs.shape[dim], dtype=probs.dtype, device=probs.device))
    return ent / max_ent

def refine_pseudo_labels(image, prob, iters=10, sxy=3, srgb=20, compat=3):
    """
    Refine pseudo labels using DenseCRF.
    Args:
        image (torch.Tensor): Input image of shape (C, H, W).
        prob (torch.Tensor): Predicted probabilities of shape (C, H, W).
        iters (int): Number of CRF iterations.
        sxy (int): Spatial kernel size.
        srgb (int): RGB kernel size.
        compat (int): Compatibility factor for pairwise potentials.
    Returns:
        torch.Tensor: Refined pseudo labels of shape (H, W).
    """
    # Ensure input is on CPU and in the correct format
    image = image.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
    image = (image * 255).astype(np.uint8).copy(order='C')
    
    # Apply CRF
    c, h, w = prob.shape
    d = dcrf.DenseCRF2D(h, w, c)
    unary = unary_from_softmax(prob.numpy()) 
    d.setUnaryEnergy(unary)
    
    # pair-wise Gaussian (smooth)
    d.addPairwiseGaussian(sxy=sxy, compat=compat, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    
    # pair-wise bilateral (edge-aware)
    d.addPairwiseBilateral(sxy=sxy, srgb=srgb, rgbim=image, compat=compat, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    
    Q = d.inference(iters)  # Run inference
    refined = np.array(Q).reshape((c, h, w)).argmax(axis=0)
    return refined.astype(np.uint8)
