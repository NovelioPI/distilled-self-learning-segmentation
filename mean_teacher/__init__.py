import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from utils import (
    build_student_model,
    compute_loss,
    compute_metrics,
)
import numpy as np

class MeanTeacher(pl.LightningModule):
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
        self.teacher = build_student_model(
            encoder=encoder, decoder=decoder, weight=weight, dropout=decoder_dropout
        )
        
        for param in self.teacher.parameters():
            param.detach_()
            
    def forward(self, x):
        return self.student(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=2, verbose=True
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "monitor": "loss/total/val"}]
    
    def training_step(self, batch, batch_idx):
        images, pseudo_logits = batch
        pseudo_probs = F.softmax(pseudo_logits, dim=1)
        _, pseudo_labels = pseudo_probs.max(dim=1)

        s_logits = self(images)
        t_logits = self.teacher(images).requires_grad_(False)
        
        s_pred = torch.argmax(s_logits, dim=1)
        t_pred = torch.argmax(t_logits, dim=1)
        
        s_loss = compute_loss(s_logits, pseudo_labels)
        t_loss = compute_loss(t_logits, pseudo_labels)
        consistency_weight = self._get_consistency_weight()
        consistency_loss = consistency_weight * torch.mean(torch.sum(F.softmax(s_logits, dim=1) * F.mse_loss(s_logits, t_logits, reduction='none'), dim=1))
        loss = s_loss['total'] + consistency_loss
        
        s_metrics = compute_metrics(s_pred, pseudo_labels)
        t_metrics = compute_metrics(t_pred, pseudo_labels)
        
        self.log("loss/total/train", loss.item(), prog_bar=True)
        self.log("loss/student/train", s_loss['total'].item(), prog_bar=False)
        self.log("loss/teacher/train", t_loss['total'].item(), prog_bar=False)
        self.log("loss/consistency/train", consistency_loss.item(), prog_bar=False)
        self.log("iou/student/train", s_metrics['iou']['micro_imagewise'].item(), prog_bar=True)
        self.log("f1/student/train", s_metrics['f1']['micro_imagewise'].item(), prog_bar=True)
        self.log("iou/teacher/train", t_metrics['iou']['micro_imagewise'].item(), prog_bar=False)
        self.log("f1/teacher/train", t_metrics['f1']['micro_imagewise'].item(), prog_bar=False)

        self._update_ema(0.99)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, pseudo_logits = batch
        pseudo_probs = F.softmax(pseudo_logits, dim=1)
        _, pseudo_labels = pseudo_probs.max(dim=1)

        s_logits = self(images)
        t_logits = self.teacher(images)
        
        s_pred = torch.argmax(s_logits, dim=1)
        t_pred = torch.argmax(t_logits, dim=1)
        
        s_loss = compute_loss(s_logits, pseudo_labels)
        t_loss = compute_loss(t_logits, pseudo_labels)
        
        s_metrics = compute_metrics(s_pred, pseudo_labels)
        t_metrics = compute_metrics(t_pred, pseudo_labels)
        
        self.log("loss/total/val", s_loss['total'].item(), prog_bar=True)
        self.log("loss/student/val", s_loss['total'].item(), prog_bar=False)
        self.log("loss/teacher/val", t_loss['total'].item(), prog_bar=False)
        self.log("iou/student/val", s_metrics['iou']['micro_imagewise'].item(), prog_bar=True)
        self.log("f1/student/val", s_metrics['f1']['micro_imagewise'].item(), prog_bar=True)
        self.log("iou/teacher/val", t_metrics['iou']['micro_imagewise'].item(), prog_bar=False)
        self.log("f1/teacher/val", t_metrics['f1']['micro_imagewise'].item(), prog_bar=False)

        return s_loss
    
    def test_step(self, batch, batch_idx):
        images, pseudo_logits = batch
        pseudo_probs = F.softmax(pseudo_logits, dim=1)
        _, pseudo_labels = pseudo_probs.max(dim=1)

        s_logits = self(images)
        t_logits = self.teacher(images)
        
        s_pred = torch.argmax(s_logits, dim=1)
        t_pred = torch.argmax(t_logits, dim=1)
        
        s_loss = compute_loss(s_logits, pseudo_labels)
        t_loss = compute_loss(t_logits, pseudo_labels)
        
        s_metrics = compute_metrics(s_pred, pseudo_labels)
        t_metrics = compute_metrics(t_pred, pseudo_labels)
        
        self.log("loss/total/test", s_loss['total'].item(), prog_bar=True)
        self.log("loss/student/test", s_loss['total'].item(), prog_bar=False)
        self.log("loss/teacher/test", t_loss['total'].item(), prog_bar=False)
        self.log("iou/student/test", s_metrics['iou']['micro_imagewise'].item(), prog_bar=True)
        self.log("f1/student/test", s_metrics['f1']['micro_imagewise'].item(), prog_bar=True)
        self.log("iou/teacher/test", t_metrics['iou']['micro_imagewise'].item(), prog_bar=False)
        self.log("f1/teacher/test", t_metrics['f1']['micro_imagewise'].item(), prog_bar=False)

        return s_loss

    def _get_consistency_weight(self):
        epoch = self.current_epoch
        rampup_length = 5.0
        if rampup_length == 0:
            return 1.0
        else:
            epoch = max(0.0, min(epoch, rampup_length))
            phase = 1.0 - epoch / rampup_length
            return float(np.exp(-5.0 * phase * phase)) * 10.0
        
    def _update_ema(self, alpha):
        alpha = min(1 - 1 / (self.global_step + 1), alpha)
        for ema_param, param in zip(self.teacher.parameters(), self.student.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1-alpha)
        
