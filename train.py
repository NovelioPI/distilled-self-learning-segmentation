import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from utils import build_student_model, compute_loss, compute_metrics, colorize_segmentation
from dataset.ugm import UGMDataModule
from transformers import get_cosine_schedule_with_warmup
import os
import itertools


class BaseModel(pl.LightningModule):
    def __init__(self, 
                 encoder='efficientnet-b0', 
                 decoder='unet', 
                 weight=None, 
                 use_kd=True, 
                 kd_weight=0.5, 
                 lr=1e-3, 
                 weight_decay=1e-4, 
                 decoder_dropout=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.use_kd = use_kd
        self.kd_weight = kd_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_classes = 19
        self.ignore_index = 255
        
        # Models
        self.student = build_student_model(
            encoder=encoder,
            decoder=decoder,
            weight=weight,
            dropout=decoder_dropout
        )
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.student.parameters(), weight_decay=self.weight_decay)

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.01 * total_steps)  # 1% warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'step',   # call every training step
            'frequency': 1
        }
        
        return [optimizer], [scheduler_config]
    
    def step(self, batch):
        images, t_logits = batch
        s_logits = self.student(images)
        
        t_probs = F.softmax(t_logits, dim=1)
        pseudo_labels = torch.argmax(t_probs, dim=1)
        preds = torch.argmax(s_logits, dim=1)
        
        loss = compute_loss(s_logits, pseudo_labels, t_probs, use_kd=self.use_kd, kd_weight=self.kd_weight)
        metrics = compute_metrics(preds, pseudo_labels, num_classes=self.num_classes, ignore_index=self.ignore_index)
        
        return {
            'loss': loss,
            'metrics': metrics,
            'image': images[0],
            'pseudo_label': pseudo_labels[0],
            'pred': preds[0],
        }
    
    def logging(self, step_output, stage='train'):
        # Log losses
        self.log(f'loss/total/{stage}', step_output['loss']['total'], on_step=True, on_epoch=True, prog_bar=True)
        if self.use_kd:
            self.log(f'loss/seg/{stage}', step_output['loss']['categorical'], on_step=True, on_epoch=True)
            self.log(f'loss/kd/{stage}', step_output['loss']['kd'], on_step=True, on_epoch=True)
        
        # Log metrics
        for metric, values in step_output['metrics'].items():
            for sub_metric, value in values.items():
                self.log(f'{metric}/{sub_metric}/{stage}', value, on_step=True, on_epoch=True)

        # Log images every 10 steps for non-training stages
        if stage != 'train':
            self.logger.experiment.add_image(
                f'{stage}_samples/image', step_output['image'], self.global_step, dataformats='CHW'
            )
            self.logger.experiment.add_image(
                f'{stage}_samples/target', colorize_segmentation(step_output['pseudo_label'].cpu()), self.global_step, dataformats='CHW'
            )
            self.logger.experiment.add_image(
                f'{stage}_samples/prediction', colorize_segmentation(step_output['pred'].cpu()), self.global_step, dataformats='CHW'
            )     
    
    def on_train_epoch_start(self):
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'])
    
    def training_step(self, batch, _):
        output = self.step(batch)
        self.logging(output, stage='train')
        return output['loss']['total']
    
    def validation_step(self, batch, _):
        output = self.step(batch)
        self.logging(output, stage='val')
        return output['loss']['total']
    
    def test_step(self, batch, _):
        output = self.step(batch)
        self.logging(output, stage='test')
        return output['loss']['total']
    
    
if __name__ == "__main__":
    torch.hub.set_dir("/media/esr/ssd0/cache")
    pl.seed_everything(42, workers=True)
    
    # Parameters
    USE_KD = [False]
    KD_WEIGHT = [0.3]
    LRS = [1e-3]
    BATCH_SIZE = [12]  # Batch sizes
    SIZE = (256, 256)
    ENCODERS = ['timm-efficientnet-b0']
    DECODERS = ['unet']
    WEIGHTS = ['imagenet']
    DECODER_DROPOUT = [0.5]
    
    all_combinations = list(itertools.product(
        USE_KD, KD_WEIGHT, LRS, BATCH_SIZE, ENCODERS, DECODERS, WEIGHTS, DECODER_DROPOUT
    ))
    
    keys = ["use_kd", "kd_weight", "lr", "batch_size", "encoder", "decoder", "weight", "decoder_dropout"]
    combinations_dict = [
        dict(zip(keys, values)) for values in all_combinations
    ]

    print(f"Total combinations: {len(combinations_dict)}")
    
    for i, params in enumerate(combinations_dict):
        model = BaseModel(
            encoder=params['encoder'],
            decoder=params['decoder'],
            weight=params['weight'],
            use_kd=params['use_kd'], 
            kd_weight=params['kd_weight'],
            lr=params['lr'],
            decoder_dropout=params['decoder_dropout']
        )
        
        dm = UGMDataModule(
            root='/media/esr/ssd0/dataset/2025-01-10/', 
            return_teacher_logits=True, 
            batch_size=params['batch_size'], size=SIZE
        )
        dm.setup()
        
        kd_str = f"_kd_alpha_{params['alpha']}" if params['use_kd'] else ""
        version = (
            f"base_{params['weight']}_dropout-{params['decoder_dropout']}_lr-{params['lr']}_batch-{params['batch_size']}"
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
                pl.callbacks.EarlyStopping(monitor='loss/total/val', patience=10, mode='min', verbose=True),
            ]
        )
        trainer.fit(model, dm)
        trainer.test(model, dm)
        
        # Save the model
        save_dir = f"saved_models/{name}/{version}/last.pth"
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        torch.save(model.state_dict(), save_dir)