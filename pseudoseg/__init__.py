import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from utils import (
    build_student_model,
    compute_loss,
    compute_metrics,
)
import numpy as np

class PseudoSeg(pl.LightningModule):
    def __init__(
        self,
        encoder="efficientnet-b0",
        decoder="unet",
        weight=None,
        lr=1e-3,
        weight_decay=1e-4,
        decoder_dropout=0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_classes = 19
        self.ignore_index = 255

        # Models
        self.student = build_student_model(
            encoder=encoder, decoder=decoder, weight=weight, dropout=decoder_dropout
        )
            
    def forward(self, x):
        return self.student(x)
    
    def loss_with_conf(self, logits, labels, conf, weight=1.0):
        loss = compute_loss(logits, labels)
        mask = conf > 0.7
        denom = mask.sum().clamp(min=1.0)
        return weight * (loss['total'] * mask).sum() / denom
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=2, verbose=True
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "monitor": "loss/total/val"}]
    
    def step(self, batch, stage):
        images, pseudo_logits = batch
        pseudo_probs = F.softmax(pseudo_logits, dim=1)
        confs, labels = pseudo_probs.max(dim=1)

        logits = self(images)
        pred = torch.argmax(logits, dim=1)
        if stage == 'train':
            loss = self.loss_with_conf(logits, labels, confs)
        else:
            loss = compute_loss(logits, labels)['total']
        metrics = compute_metrics(pred, labels)

        self.log(f"loss/total/{stage}", loss.item(), prog_bar=True)
        self.log(f"loss/student/{stage}", loss.item(), prog_bar=False)
        self.log(f"iou/student/{stage}", metrics['iou']['micro_imagewise'].item(), prog_bar=True)
        self.log(f"f1/student/{stage}", metrics['f1']['micro_imagewise'].item(), prog_bar=True)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, 'val')
    
    def test_step(self, batch, batch_idx):
        return self.step(batch, 'test')