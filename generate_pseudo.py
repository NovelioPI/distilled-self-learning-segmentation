# generate_pseudo_labels.py
import os
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from dataset.ugm import UGMDataset
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from tqdm import tqdm
from pathlib import Path

torch.hub.set_dir("/media/esr/ssd0/cache")
pl.seed_everything(42, workers=True)

# ----------- SETTINGS -----------
UGM_ROOT = "/media/esr/ssd0/dataset/2025-01-10"
OUT_UGM_DIR = "/media/esr/ssd0/dataset/2025-01-10/camera/seg/pseudo_labels_segformer_b5/logits"
MC_DROPOUT_TIMES = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 19
# --------------------------------

# Ensure output directory exists
OUT_UGM_DIR = Path(OUT_UGM_DIR)
OUT_UGM_DIR.mkdir(parents=True, exist_ok=True)

# Load UGM dataset
ugm_dataset = UGMDataset(UGM_ROOT)

processor = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
)
teacher_model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
    num_labels=NUM_CLASSES
).to(DEVICE)
teacher_model.eval()


def enable_mc_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_dropout_predict(pil_images, model, processor, mc_times=4, percentile=0.75, device="cuda"):
    enable_mc_dropout(model)
    
    logits_mc = []
    for _ in range(mc_times):
        inputs = processor(images=pil_images, return_tensors="pt").to(device)
        logits = model(**inputs).logits  # [B, C, H, W]
        logits_mc.append(logits)
        
    logits_mc = torch.stack(logits_mc, dim=0)  # [mc_times, B, C, H, W]
    mean_logits = logits_mc.mean(0) # [B, C, H, W]
    
    return mean_logits.squeeze(0)  # [C, H, W]
    

if __name__ == "__main__":
    print("Generating pseudo-labels for unlabeled datasets (UGM)...")
    for i, dataset in enumerate(tqdm(ugm_dataset, desc="Generating pseudo-labels")):
        image = dataset.to(DEVICE)
        pil_image = [transforms.ToPILImage()(image).convert("RGB")]

        # MC-Dropout prediction
        mean_logits = mc_dropout_predict(
            pil_image, teacher_model, processor, mc_times=MC_DROPOUT_TIMES, device=DEVICE
        )
        
        file_name = ugm_dataset.get_image_path(i).stem
        np.save(OUT_UGM_DIR / f"{file_name}.npy", mean_logits.cpu().numpy())
        
