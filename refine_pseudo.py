from dataset.ugm import UGMDataset
from pathlib import Path
from tqdm import tqdm
from utils import refine_pseudo_labels, colorize_segmentation
from PIL import Image
import pytorch_lightning as pl

pl.seed_everything(42, workers=True)


# ──────────────── SETTINGS ────────────────────────────────────────────────────
ENTROPY_THRESHOLD = [1.0, 0.05, 0.15, 0.25, 0.35, 0.45]

UGM_ROOT = Path("/media/esr/ssd0/dataset/2025-01-10")

OUT_UGM_DIR = UGM_ROOT / "camera/seg/pseudo_labels_segformer_b5"
OUT_UGM_DIR.mkdir(parents=True, exist_ok=True)

OUT_REFINED_PSEUDO_LABELS = OUT_UGM_DIR / "refined_pseudo_labels"
OUT_REFINED_PSEUDO_LABELS.mkdir(parents=True, exist_ok=True)
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────── DATASET ────────────────────────────────────────────────────
ugm_dataset = UGMDataset(UGM_ROOT, return_teacher_logits=True, size=(256, 256))
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────── REFINING PSEUDO LABELS ──────────────────────────────────────
for i, batch in enumerate(tqdm(ugm_dataset, desc="Refining pseudo labels")):
    file_name = ugm_dataset.get_image_path(i).stem
    rpl_path = OUT_REFINED_PSEUDO_LABELS / f"{file_name}.png"
    rpl_col_path = OUT_REFINED_PSEUDO_LABELS / f"{file_name}_colorized.png"
    if rpl_path.exists() and rpl_col_path.exists():
        continue
    
    image, t_logits = batch
    prob = t_logits.softmax(dim=0)  # [C, H, W]
    
    # Refine pseudo labels using DenseCRF
    refined_pseudo_labels = refine_pseudo_labels(
        image,
        prob,
    )
    
    # Colorize the refined pseudo labels
    refined_colorized = colorize_segmentation(refined_pseudo_labels, num_classes=ugm_dataset.num_classes)

    # Save refined pseudo labels
    Image.fromarray(refined_pseudo_labels).save(rpl_path)
    Image.fromarray(refined_colorized.permute(1, 2, 0).cpu().numpy()).save(rpl_col_path)
        
# ──────────────────────────────────────────────────────────────────────────────