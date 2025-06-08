import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


class Segformer(nn.Module):
    def __init__(self):
        super(Segformer, self).__init__()
        model_name = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name, 
            output_hidden_states=True, 
            output_attentions=True
        )
        self.model.eval()

    @torch.no_grad()
    def forward(self, x):
        inputs = self.processor(images=x, return_tensors="pt").to(x.device)
        outputs = self.model(**inputs)
        return outputs.logits