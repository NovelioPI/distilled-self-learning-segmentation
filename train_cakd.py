import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from models.efficientnet import EfficientNet
from models.segformer import Segformer
from utils import compute_loss, compute_metrics
from dataset.ugm import UGMDataModule
from transformers import SegformerForSemanticSegmentation
import os
import itertools
import segmentation_models_pytorch as smp


class CAKDModel(pl.LightningModule):
    def __init__(self, encoder='efficientnet-b0', decoder='unet', weight=None, use_kd=True, alpha=0.5, lr=1e-3, weight_decay=1e-4, weight_rkd=0.1, weight_cad=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.use_kd = use_kd
        self.alpha = alpha
        self.lr = lr
        self.weight_decay = weight_decay
        self.weight_rkd = weight_rkd
        self.weight_cad = weight_cad
        self.num_classes = 19
        self.ignore_index = 255
        
        # Models
        self.student = self.model = smp.Unet(
                encoder_name=f'timm-{encoder}',
                encoder_weights=weight if weight else None,
                in_channels=3,
                classes=19,
            )
        self.teacher = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        )
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
            
        # Feature hooks
        self.teacher_feats = {}
        self.student_feats = {}
        self.teacher.segformer.encoder.register_forward_hook(self._get_hook(self.teacher_feats, 'feat'))
        self.student.encoder.register_forward_hook(self._get_hook(self.student_feats, 'feat'))
        
    def _get_hook(self, store_dict, key):
        def hook(module, input, output):
            store_dict[key] = output
        return hook
    
    @staticmethod
    def pairwise_cosine_similarity(x):
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1).permute(0, 2, 1)  # [B, H*W, C]
        sim = torch.bmm(x_flat, x_flat.permute(0, 2, 1))
        return sim
    
    def relational_distillation_loss(self, student_feats, teacher_feats):
        teacher_sim = self.pairwise_cosine_similarity(teacher_feats)
        student_sim = self.pairwise_cosine_similarity(student_feats)
        return F.mse_loss(student_sim, teacher_sim)
    
    @staticmethod
    def channel_attention(x):
        attn = x.mean(dim=[2, 3]) 
        attn = F.softmax(attn, dim=1)
        return attn
    
    def attention_distillation_loss(self, student_feats, teacher_feats):
        attn_teacher = self.channel_attention(teacher_feats)
        attn_student = self.channel_attention(student_feats)
        return F.mse_loss(attn_student, attn_teacher)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.student.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=self.trainer.max_epochs, power=0.9)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
    
    def step(self, batch, stage='train'):
        images = batch
        input_for_student = F.interpolate(images, size=(images.shape[2] // 4, images.shape[3] // 4), mode='bilinear', align_corners=False)
        student_logits = self.student(input_for_student)
        student_feats = self.student_feats['feat'][-1]
        
        with torch.no_grad():
            teacher_logits = self.teacher(pixel_values=images).logits
            teacher_feats = self.teacher_feats['feat']['last_hidden_state']
            
        print(f"Student feats shape: {student_feats.shape}, Teacher feats shape: {teacher_feats.shape}")
            
        # Spatial alignment if needed
        if student_feats.shape[-2:] != teacher_feats.shape[-2:]:
            student_feats = F.interpolate(student_feats, size=teacher_feats.shape[-2:], mode='bilinear', align_corners=False)
        
        # Pseudo-labels from teacher
        soft_labels = torch.softmax(teacher_logits, dim=1)
        hard_labels = torch.argmax(soft_labels, dim=1)
        
        # Compute losses
        loss_seg = compute_loss(student_logits, hard_labels, soft_labels, use_kd=self.use_kd, alpha=self.alpha)
        loss_rkd = self.relational_distillation_loss(student_feats, teacher_feats)
        loss_cad = self.attention_distillation_loss(student_feats, teacher_feats)
        loss = loss_seg + self.weight_rkd * loss_rkd + self.weight_cad * loss_cad
        
        # Compute metrics
        preds = torch.argmax(student_logits, dim=1)
        metrics = compute_metrics(preds, hard_labels, num_classes=self.num_classes, ignore_index=self.ignore_index)
        
        return {
            'loss': {
                'total': loss,
                'seg': loss_seg,
                'rkd': loss_rkd,
                'cad': loss_cad,
            },
            'metrics': metrics,
        }
    
    def training_step(self, batch, _):
        output = self.step(batch, stage='train')
        self.log('train_loss/total', output['loss']['total'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_loss/seg', output['loss']['seg'], on_step=True, on_epoch=True)
        self.log('train_loss/rkd', output['loss']['rkd'], on_step=True, on_epoch=True)
        self.log('train_loss/cad', output['loss']['cad'], on_step=True, on_epoch=True)
        self.log_dict({f'train_{k}': v for k, v in output['metrics'].items()}, on_step=True, on_epoch=True)
        return output['loss']
    
    def validation_step(self, batch, _):
        output = self.step(batch, stage='val')
        self.log('val_loss/total', output['loss']['total'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_loss/seg', output['loss']['seg'], on_step=True, on_epoch=True)
        self.log('val_loss/rkd', output['loss']['rkd'], on_step=True, on_epoch=True)
        self.log('val_loss/cad', output['loss']['cad'], on_step=True, on_epoch=True)
        self.log_dict({f'val_{k}': v for k, v in output['metrics'].items()}, on_step=True, on_epoch=True)
        return output['loss']
    
    def test_step(self, batch, _):
        output = self.step(batch, stage='test')
        self.log('test_loss/total', output['loss']['total'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('test_loss/seg', output['loss']['seg'], on_step=True, on_epoch=True)
        self.log('test_loss/rkd', output['loss']['rkd'], on_step=True, on_epoch=True)
        self.log('test_loss/cad', output['loss']['cad'], on_step=True, on_epoch=True)
        self.log_dict({f'test_{k}': v for k, v in output['metrics'].items()}, on_step=True, on_epoch=True)
        return output['loss']

if __name__ == "__main__":
    torch.hub.set_dir("/media/esr/ssd0/cache")
    pl.seed_everything(42, workers=True)
    
    # Parameters
    USE_KD = [False]
    ALPHAS = [0.3]
    LRS = [1e-3]
    BATCH_SIZE = [12]  # Batch sizes
    SIZE = (256, 256)
    ENCODERS = ['efficientnet-b0']
    DECODERS = ['unet']
    WEIGHTS = ['imagenet']
    WEIGHT_RKD = [0.1]
    WEIGHT_CAD = [0.1]
    
    all_combinations = list(itertools.product(
        USE_KD, ALPHAS, LRS, BATCH_SIZE, ENCODERS, DECODERS, WEIGHTS, WEIGHT_RKD, WEIGHT_CAD
    ))
    
    keys = ["use_kd", "alpha", "lr", "batch_size", "encoder", "decoder", "weight", "weight_rkd", "weight_cad"]
    combinations_dict = [
        dict(zip(keys, values)) for values in all_combinations
    ]

    print(f"Total combinations: {len(combinations_dict)}")
    
    for i, params in enumerate(combinations_dict):
        model = CAKDModel(
            encoder=params['encoder'],
            decoder=params['decoder'],
            weight=params['weight'],
            use_kd=params['use_kd'], 
            alpha=params['alpha'],
            lr=params['lr'],
            weight_rkd=params['weight_rkd'],
            weight_cad=params['weight_cad'],
        )
        
        dm = UGMDataModule(
            root='/media/esr/ssd0/dataset/2025-01-10/',
            batch_size=params['batch_size'], size=SIZE
        )
        dm.setup()
        
        kd_str = f"_kd_alpha_{params['alpha']}" if params['use_kd'] else ""
        version = (
            f"cakd_{params['weight']}_lr-{params['lr']}_batch-{params['batch_size']}_rkd-{params['weight_rkd']}_cad-{params['weight_cad']}"
            f"{kd_str}_{SIZE[0]}x{SIZE[1]}"
        )
        name = f"{params['encoder']}_{params['decoder']}"
        logger = pl.loggers.TensorBoardLogger('logs/', name=name, version=version)
        trainer = pl.Trainer(
            max_epochs=100,
            accelerator='gpu',
            devices=1,
            logger=logger,
            callbacks=[
                pl.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=True),
            ]
        )
        trainer.fit(model, dm)
        trainer.test(model, dm)
        
        # Save the model
        save_dir = f"saved_models/{name}/{version}/last.pth"
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        torch.save(model.state_dict(), save_dir)