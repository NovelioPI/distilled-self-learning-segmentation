from collections import deque
import numpy as np
import torch
from dataset.ugm import UGMDataset
from pathlib import Path
from tqdm import tqdm
from utils import colorize_segmentation
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl

pl.seed_everything(42, workers=True)


# ──────────────── SETTINGS ────────────────────────────────────────────────────
UGM_ROOT = Path("/media/esr/ssd0/dataset/2025-01-10")
OUT_UGM_DIR = UGM_ROOT / "camera/seg/pseudo_labels_segformer_b5"

OUT_ACT = OUT_UGM_DIR / "refined-act_pseudo_labels"
OUT_ACT.mkdir(parents=True, exist_ok=True)
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────── DATASET ─────────────────────────────────────────────────────
ugm_dataset = UGMDataset(
    UGM_ROOT,
    size=(256, 256),
    return_teacher_logits=True,
    use_refinement=True,
    entropy_threshold=1.0 
)
# ──────────────────────────────────────────────────────────────────────────────


# ─────────────── Adaptive Confidence Thresholding (ACT) ───────────────────────
class ACT:
    def __init__(self, num_classes, history=2048, fallback=0.25):
        self.buf = [deque(maxlen=history) for _ in range(num_classes)]
        self.fb  = fallback

    @torch.no_grad()
    def update(self, probs, hard):
        conf = probs.max(0).values   # [B,H,W]
        for c, dq in enumerate(self.buf):
            mask = (hard == c)
            if mask.any():
                dq.extend(conf[mask].flatten().tolist())

    def sigma(self, c):
        # current threshold σ_c
        return np.median(self.buf[c]) if self.buf[c] else self.fb
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────── REFINING PSEUDO LABELS ──────────────────────────────────────
act = ACT(num_classes=ugm_dataset.num_classes, history=4096) # More history = more stable thresholds

writer = SummaryWriter("logs/act")
log_every = 500 # Images
    
for i, batch in enumerate(tqdm(ugm_dataset, desc="Calculating ACT thresholds")):
    file_name = ugm_dataset.get_image_path(i).stem
    act_rpl_path = OUT_ACT / f"{file_name}.png"
    act_rpl_col_path = OUT_ACT / f"{file_name}_colorized.png"
    
    image, t_logits, refined_label = batch
    prob = t_logits.softmax(dim=0)  # [C, H, W]
    hard_labels = t_logits.argmax(dim=0)
    
    # Update ACT with current probabilities and hard labels
    act.update(prob, hard_labels)
    if i % log_every == 0:
        for c in range(ugm_dataset.num_classes):
            writer.add_scalar(f"act/threshold_{c}", act.sigma(c), i)

    # Apply ACT thresholding
    conf = prob.max(dim=0).values  # [H, W]
    keep  = torch.zeros_like(hard_labels.squeeze(0), dtype=torch.bool)
    for c in range(ugm_dataset.num_classes):
        keep |= (hard_labels == c) & (conf >= act.sigma(c))
    refined_label[~keep] = 255

    # Colorize the refined pseudo labels
    refined_colorized = colorize_segmentation(refined_label, num_classes=ugm_dataset.num_classes)

    # Save refined pseudo labels
    Image.fromarray(refined_label.cpu().numpy().astype(np.uint8)).save(act_rpl_path)
    Image.fromarray(refined_colorized.permute(1, 2, 0).cpu().numpy()).save(act_rpl_col_path)

writer.close()
# ──────────────────────────────────────────────────────────────────────────────

