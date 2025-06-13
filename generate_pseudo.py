# generate_pseudo_labels.py
import os
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from dataset.ugm import UGMDataset
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
from models.teacher_model import build_teacher_model, inference
from utils import colorize_segmentation
from PIL import Image

torch.hub.set_dir("/media/esr/ssd0/cache")
pl.seed_everything(42, workers=True)


# ──────────────── SETTINGS ────────────────────────────────────────────────────
UGM_ROOT = Path("/media/esr/ssd0/dataset/2025-01-10")
OUT_UGM_DIR = UGM_ROOT / "camera/seg/pseudo_labels_segformer_b5/"
OUT_UGM_DIR.mkdir(parents=True, exist_ok=True)

OUT_LOGITS = OUT_UGM_DIR / "logits"
OUT_LOGITS.mkdir(parents=True, exist_ok=True)

OUT_PSEUDO_LABELS = OUT_UGM_DIR / "pseudo_labels"
OUT_PSEUDO_LABELS.mkdir(parents=True, exist_ok=True)

MC_DROPOUT_TIMES = 5
NUM_CLASSES = 19
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────── SETUP DATASET & MODEL ───────────────────────────────────────
ugm_dataset = UGMDataset(UGM_ROOT)
teacher_model, processor = build_teacher_model()
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────── GENERATE PSEUDO-LABELS ──────────────────────────────────────
print("Generating pseudo-labels for unlabeled datasets (UGM)...")
for i, dataset in enumerate(tqdm(ugm_dataset, desc="Generating pseudo-labels")):
    image = dataset.to(DEVICE)
    pil_image = [transforms.ToPILImage()(image).convert("RGB")]

    # MC-Dropout prediction
    mean_logits = inference(
        pil_image, teacher_model, processor, 
        mc_times=MC_DROPOUT_TIMES, device=DEVICE
    )
    
    # Generate pseudo-labels
    probs = mean_logits.softmax(dim=0)  # [C, H, W]
    pseudo_labels = torch.argmax(probs, dim=0)  # [H, W]
    colorized_pseudo_labels = colorize_segmentation(pseudo_labels, num_classes=NUM_CLASSES)
    
    # Save
    file_name = ugm_dataset.get_image_path(i).stem
    logits_path = OUT_LOGITS / f"{file_name}.npy"
    pl_path = OUT_PSEUDO_LABELS / f"{file_name}.png"
    pl_color_path = OUT_PSEUDO_LABELS / f"{file_name}_colorized.png"
    
    if not logits_path.exists() and not pl_path.exists() and not pl_color_path.exists():
        np.save(OUT_LOGITS / f"{file_name}.npy", mean_logits.cpu().numpy())
        Image.fromarray(pseudo_labels.cpu().numpy().astype(np.uint8)).save(pl_path)
        Image.fromarray(colorized_pseudo_labels.permute(1, 2, 0).cpu().numpy()).save(pl_color_path)
# ──────────────────────────────────────────────────────────────────────────────
