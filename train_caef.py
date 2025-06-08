import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from models.efficientnet import EfficientNet
from utils import compute_loss, compute_metrics
from dataset.ugm import UGMDataModule
from transformers import SegformerForSemanticSegmentation
import os
import itertools


class CAEFModel(pl.LightningModule):
    def __init__(self, encoder='efficientnet-b0', decoder='unet', percentile=0.75, weight=None, use_kd=True, alpha=0.5, lr=1e-3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.use_kd = use_kd
        self.alpha = alpha
        self.lr = lr
        self.percentile = percentile
        self.weight_decay = weight_decay
        self.num_classes = 19
        self.ignore_index = 255
        self.warmup_epochs = 5
        
        # Models
        self.student = EfficientNet(encoder=encoder, decoder=decoder, weight=weight)
        self.teacher = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b5-finetuned-cityscapes-1024-1024", 
            output_hidden_states=True, 
            output_attentions=True
        )
        self.teacher.eval()
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.student.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=self.trainer.max_epochs, power=0.9)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
    
    def step(self, batch, stage='train'):
        images = batch
        student_logits = self.student(images)
        student_logits = F.interpolate(student_logits, size=(images.shape[2] // 4, images.shape[3] // 4), mode='bilinear', align_corners=False)
        with torch.no_grad():
            teacher_logits = self.teacher(pixel_values=images).logits
        
        # Pseudo-labels from teacher
        soft_labels = torch.softmax(teacher_logits, dim=1)
        hard_labels = torch.argmax(soft_labels, dim=1)
        
        # Apply percentile-based filtering to hard labels
        if stage == 'train' and self.trainer.current_epoch >= self.warmup_epochs:
            entropy = -torch.sum(soft_labels * torch.log(soft_labels + 1e-10), dim=1) # [B, H, W]
            trusted_mask = torch.zeros_like(hard_labels, dtype=torch.bool)
            for c in range(self.num_classes):
                class_mask = (hard_labels == c)
                if class_mask.sum() == 0:
                    continue
                class_entropy = entropy[class_mask]
                threshold = torch.quantile(class_entropy, self.percentile)
                keep_mask = class_entropy <= threshold
                trusted_mask[class_mask] = keep_mask
            hard_labels[~trusted_mask] = self.ignore_index
        
        # Compute losses
        loss = compute_loss(student_logits, hard_labels, soft_labels, use_kd=self.use_kd, alpha=self.alpha)
        
        # Compute metrics
        preds = torch.argmax(student_logits, dim=1)
        metrics = compute_metrics(preds, hard_labels, num_classes=self.num_classes, ignore_index=self.ignore_index)
        
        return {'loss': loss, 'metrics': metrics,}
    
    def training_step(self, batch, _):
        output = self.step(batch, stage='train')
        self.log('train_loss', output['loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict({f'train_{k}': v for k, v in output['metrics'].items()}, on_step=True, on_epoch=True)
        return output['loss']
    
    def validation_step(self, batch, _):
        output = self.step(batch, stage='val')
        self.log('val_loss', output['loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict({f'val_{k}': v for k, v in output['metrics'].items()}, on_step=True, on_epoch=True)
        return output['loss']
    
    def test_step(self, batch, _):
        output = self.step(batch, stage='test')
        self.log('test_loss', output['loss'], on_step=True, on_epoch=True, prog_bar=True)
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
    SIZE = (1024, 1024)
    ENCODERS = ['efficientnet-b0']
    DECODERS = ['unet']
    WEIGHTS = ['imagenet']
    PERCENTILE = [0.95]
    
    all_combinations = list(itertools.product(
        USE_KD, ALPHAS, LRS, BATCH_SIZE, ENCODERS, DECODERS, WEIGHTS, PERCENTILE
    ))
    
    keys = ["use_kd", "alpha", "lr", "batch_size", "encoder", "decoder", "weight", "percentile"]
    combinations_dict = [
        dict(zip(keys, values)) for values in all_combinations
    ]

    print(f"Total combinations: {len(combinations_dict)}")
    
    for i, params in enumerate(combinations_dict):
        model = CAEFModel(
            encoder=params['encoder'],
            decoder=params['decoder'],
            weight=params['weight'],
            percentile=params['percentile'],
            use_kd=params['use_kd'], 
            alpha=params['alpha'],
            lr=params['lr'],
        )
        
        dm = UGMDataModule(
            root='/media/esr/ssd0/dataset/2025-01-10/', 
            batch_size=params['batch_size'], size=SIZE
        )
        dm.setup()
        
        kd_str = f"_kd_alpha_{params['alpha']}" if params['use_kd'] else ""
        version = (
            f"caef_{params['weight']}_pctl-{params['percentile']}_lr-{params['lr']}_batch-{params['batch_size']}"
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