import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# Function to build the teacher model and processor
def build_teacher_model():
    processor = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    )
    teacher_model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
    )
    return teacher_model, processor

# Funvtion to perform inference with MC Dropout
@torch.no_grad()
def inference(pil_images, model, processor, use_mc_dropout=False, mc_times=1, device="cuda"):
    model = model.to(device)
    
    # Enable MC Dropout in the model
    if use_mc_dropout:
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
    else:
        model.eval()
    
    logits_list = []
    for _ in range(mc_times):
        inputs = processor(images=pil_images, return_tensors="pt").to(device)
        logits = model(**inputs).logits  # [B, C, H, W]
        logits_list.append(logits)
        
    logits_cat = torch.stack(logits_list, dim=0)  # [mc_times, B, C, H, W]
    mean_logits = logits_cat.mean(0) # [B, C, H, W]
    
    return mean_logits.squeeze(0)  # [C, H, W]
