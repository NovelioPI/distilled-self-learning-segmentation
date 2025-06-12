from dataset.ugm import UGMDataset, get_val_transform
from pathlib import Path
from tqdm import tqdm
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import torch
from utils import colorize_segmentation
import numpy as np
from PIL import Image

ENTROPY_THRESHOLD = [1.0, 0.05, 0.15, 0.25, 0.35, 0.45]
UGM_ROOT = Path("/media/esr/ssd0/dataset/2025-01-10")
OUT_UGM_DIR = UGM_ROOT / "camera/seg/pseudo_labels_segformer_b5"
OUT_UGM_DIR.mkdir(parents=True, exist_ok=True)
OUT_PSEUDO_LABELS = OUT_UGM_DIR / "pseudo_labels"
OUT_PSEUDO_LABELS.mkdir(parents=True, exist_ok=True)

ugm_dataset = UGMDataset(UGM_ROOT, return_teacher_logits=True, size=(256, 256), transform=get_val_transform(size=(256, 256)))

def refine_pseudo_labels(image, prob, entropy_threshold=0.75, num_classes=19):
    # Compute entropy
    entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=0)  # [H, W]
    max_entropy = torch.log(torch.tensor(num_classes, dtype=prob.dtype, device=prob.device))
    normalized_entropy = entropy / max_entropy
    mask = normalized_entropy < entropy_threshold  # [H, W]
    
    h, w = image.shape[1], image.shape[2]
    image = image.permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
    image = (image * 255).astype(np.uint8).copy(order='C')  # Convert to uint8
    
    # Apply CRF
    unary = unary_from_softmax(prob.numpy()) 
    d = dcrf.DenseCRF2D(h, w, num_classes)
    d.setUnaryEnergy(unary)
    
    # Add pairwise potentials (optional, can be tuned)
    sxy = 3  # Spatial kernel size
    srgb = 20  # Color kernel size
    compat = 3  # Compatibility factor
    d.addPairwiseGaussian(sxy=sxy, compat=compat, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseBilateral(sxy=sxy, srgb=srgb, rgbim=image, compat=compat, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    
    refined_logits = d.inference(5)  # Run inference
    refined_logits = np.array(refined_logits).reshape((num_classes, h, w)) # [C, H, W]
    
    refined_labels = torch.argmax(torch.tensor(refined_logits), dim=0)
    refined_labels[~mask] = 255  # Set uncertain pixels to ignore index (255)
    
    return torch.tensor(refined_labels, dtype=torch.long)

for threshold in ENTROPY_THRESHOLD:
    OUT_REFINED_PSEUDO_LABELS = OUT_UGM_DIR / f"refined-{threshold}_pseudo_labels"
    OUT_REFINED_PSEUDO_LABELS.mkdir(parents=True, exist_ok=True)
    for i, batch in enumerate(tqdm(ugm_dataset, desc="Refining pseudo labels")):
        image, t_logits = batch
        file_name = ugm_dataset.get_image_path(i).stem
        prob = t_logits.softmax(dim=0)  # [C, H, W]
        
        # Save original pseudo labels
        pl_path = OUT_PSEUDO_LABELS / f"{file_name}.png"
        pl_col_path = OUT_PSEUDO_LABELS / f"{file_name}_colorized.png"
        if not pl_path.exists() or not pl_col_path.exists():
            pseudo_label = torch.argmax(prob, dim=0)  # [H, W]
            Image.fromarray(pseudo_label.cpu().numpy().astype(np.uint8)).save(pl_path)
            colorized_pseudo_label = colorize_segmentation(pseudo_label, num_classes=ugm_dataset.num_classes)
            Image.fromarray(colorized_pseudo_label.permute(1, 2, 0).cpu().numpy()).save(pl_col_path)
        
        rpl_path = OUT_REFINED_PSEUDO_LABELS / f"{file_name}.png"
        rpl_col_path = OUT_REFINED_PSEUDO_LABELS / f"{file_name}_colorized.png"
        if rpl_path.exists() and rpl_col_path.exists():
            continue
        
        refined_pseudo_labels = refine_pseudo_labels(
            image,
            prob,
            entropy_threshold=threshold,
            num_classes=ugm_dataset.num_classes
        )
        refined_colorized = colorize_segmentation(refined_pseudo_labels, num_classes=ugm_dataset.num_classes)
        
        # Save refined pseudo labels
        Image.fromarray(refined_pseudo_labels.cpu().numpy().astype(np.uint8)).save(rpl_path)
        Image.fromarray(refined_colorized.permute(1, 2, 0).cpu().numpy()).save(rpl_col_path)
        