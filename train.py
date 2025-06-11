import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from utils import (
    build_student_model,
    compute_loss,
    negative_sampling_loss,
    compute_metrics,
    colorize_segmentation,
)
from dataset.ugm import UGMDataModule
from transformers import get_cosine_schedule_with_warmup
import os
import itertools
from models.refinement import EncoderPRN
from math import cos, pi


class BaseModel(pl.LightningModule):
    def __init__(
        self,
        encoder="efficientnet-b0",
        decoder="unet",
        weight=None,
        use_kd=True,
        kd_weight=0.5,
        lr=1e-3,
        weight_decay=1e-4,
        decoder_dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.use_kd = use_kd
        self.kd_weight = kd_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_classes = 19
        self.ignore_index = 255
        self.confidence_threshold = kwargs.get("confidence_threshold", 0)
        self.use_refinement = kwargs.get("use_refinement", False)
        self.use_curriculum_thr = kwargs.get("use_curriculum_thr", False)
        self.use_neg_loss = kwargs.get("use_neg_loss", False)

        # Models
        self.student = build_student_model(
            encoder=encoder, decoder=decoder, weight=weight, dropout=decoder_dropout
        )
        
        self.prn = EncoderPRN(
            in_channels=self.num_classes+3,
            num_classes=self.num_classes,
            mid_channels=32,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.student.parameters()) + list(self.prn.parameters()), lr=self.lr, weight_decay=self.weight_decay
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.01 * total_steps)  # 1% warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",  # call every training step
            "frequency": 1,
        }

        return [optimizer], [scheduler_config]

    def step(self, batch, stage="train"):
        images, t_logits = batch
        s_logits = self.student(images)
        preds = torch.argmax(s_logits, dim=1)

        t_probs = F.softmax(t_logits, dim=1)
        conf, pseudo_labels = torch.max(t_probs, dim=1)

        # Apply confidence threshold to pseudo labels
        if ((self.confidence_threshold > 0) or self.use_curriculum_thr) and stage == "train":
            if self.use_curriculum_thr:
                self.confidence_threshold = self.curriculum_threshold(self.current_epoch, self.trainer.max_epochs)
            conf_mask = conf > self.confidence_threshold
            pseudo_labels[~conf_mask] = self.ignore_index
          
        loss = compute_loss(
            s_logits,
            pseudo_labels,
            t_logits if self.use_kd else None,
            use_kd=self.use_kd,
            kd_weight=self.kd_weight,
        )
        
        # Apply negative sampling loss if enabled
        if self.use_neg_loss and stage == "train":
            neg_loss = negative_sampling_loss(s_logits, t_probs, confidence_threshold=self.confidence_threshold)
            loss["total"] += neg_loss
            loss["neg"] = neg_loss  
        
        # Apply Refinement to teacher logits if enabled
        if self.use_refinement and stage == "train":
            refined_logits = self.prn(images, t_logits)
            loss_prn = F.cross_entropy(refined_logits, pseudo_labels, ignore_index=self.ignore_index)
            pseudo_labels = refined_logits.argmax(dim=1)

            loss["total"] += loss_prn
            loss["prn"] = loss_prn
        
            
        metrics = compute_metrics(
            preds,
            pseudo_labels,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )

        return {
            "loss": loss,
            "metrics": metrics,
            "image": images[0],
            "pseudo_label": pseudo_labels[0],
            "pred": preds[0],
        }

    def logging(self, step_output, stage="train"):
        # Log losses
        self.log(
            f"loss/total/{stage}",
            step_output["loss"]["total"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        if self.use_kd:
            self.log(
                f"loss/seg/{stage}",
                step_output["loss"]["seg"],
                on_step=True,
                on_epoch=True,
            )
            self.log(
                f"loss/kd/{stage}",
                step_output["loss"]["kd"],
                on_step=True,
                on_epoch=True,
            )

        # Log metrics
        for metric, values in step_output["metrics"].items():
            for sub_metric, value in values.items():
                self.log(
                    f"{metric}/{sub_metric}/{stage}", value, on_step=True, on_epoch=True
                )

        # Log images every 10 steps for non-training stages
        if stage != "train":
            self.logger.experiment.add_image(
                f"{stage}_samples/image",
                step_output["image"],
                self.global_step,
                dataformats="CHW",
            )
            self.logger.experiment.add_image(
                f"{stage}_samples/target",
                colorize_segmentation(step_output["pseudo_label"].cpu()),
                self.global_step,
                dataformats="CHW",
            )
            self.logger.experiment.add_image(
                f"{stage}_samples/prediction",
                colorize_segmentation(step_output["pred"].cpu()),
                self.global_step,
                dataformats="CHW",
            )

    def on_train_epoch_start(self):
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])

    def training_step(self, batch, _):
        output = self.step(batch, stage="train")
        self.logging(output, stage="train")
        return output["loss"]["total"]

    def validation_step(self, batch, _):
        output = self.step(batch, stage="val")
        self.logging(output, stage="val")
        return output["loss"]["total"]

    def test_step(self, batch, _):
        output = self.step(batch, stage="test")
        self.logging(output, stage="test")
        return output["loss"]["total"]

    def curriculum_threshold(self, epoch, total_epochs, min_thr=0.75, max_thr=0.95):
        threshold = min_thr + 0.5 * (max_thr - min_thr) * (1 + cos(pi * epoch / total_epochs))
        self.log("curriculum_threshold", threshold, on_step=True, on_epoch=True)
        return threshold

if __name__ == "__main__":
    torch.hub.set_dir("/media/esr/ssd0/cache")
    pl.seed_everything(42, workers=True)

    # Parameters
    USE_KD = [True]
    KD_WEIGHT = [0.001]
    LR = [1e-3]
    BATCH_SIZE = [12]  # Batch sizes
    ENCODER = ["timm-efficientnet-b0"]
    DECODER = ["unet"]
    WEIGHT = ["imagenet"]
    DECODER_DROPOUT = [0.5]
    CONFIDENCE_THRESHOLD = [0.95]
    USE_REFINEMENT = [True]
    USE_CURRICULUM_THR = [True]
    USE_NEG_LOSS = [True]
    
    SIZE = (256, 256)

    all_combinations = list(
        itertools.product(
            USE_KD,
            KD_WEIGHT,
            LR,
            BATCH_SIZE,
            ENCODER,
            DECODER,
            WEIGHT,
            DECODER_DROPOUT,
            CONFIDENCE_THRESHOLD,
            USE_REFINEMENT,
            USE_CURRICULUM_THR,
            USE_NEG_LOSS,
        )
    )

    keys = [
        "use_kd",
        "kd_weight",
        "lr",
        "batch_size",
        "encoder",
        "decoder",
        "weight",
        "decoder_dropout",
        "confidence_threshold",
        "use_refinement",
        "use_curriculum_thr",
        "use_neg_loss",
    ]
    combinations_dict = [dict(zip(keys, values)) for values in all_combinations]

    print(f"Total combinations: {len(combinations_dict)}")

    for i, params in enumerate(combinations_dict):
        model = BaseModel(
            encoder=params["encoder"],
            decoder=params["decoder"],
            weight=params["weight"],
            use_kd=params["use_kd"],
            kd_weight=params["kd_weight"],
            lr=params["lr"],
            decoder_dropout=params["decoder_dropout"],
            confidence_threshold=params["confidence_threshold"],
            use_refinement=params["use_refinement"],
            use_curriculum_thr=params["use_curriculum_thr"],
            use_neg_loss=params["use_neg_loss"],
        )

        dm = UGMDataModule(
            root="/media/esr/ssd0/dataset/2025-01-10/",
            return_teacher_logits=True,
            batch_size=params["batch_size"],
            size=SIZE,
        )
        dm.setup()

        kd_str = f"_kd_alpha_{params['kd_weight']}" if params["use_kd"] else ""
        if params['confidence_threshold'] > 0 or params['use_refinement'] or params['use_curriculum_thr']:
            version = (
                f"test_conf-{params['confidence_threshold']}_{params['weight']}_dropout-{params['decoder_dropout']}_lr-{params['lr']}_batch-{params['batch_size']}"
                f"{kd_str}_{SIZE[0]}x{SIZE[1]}"
            )
            if params['use_refinement']:
                version += "_refinement"
            if params['use_curriculum_thr']:
                version += "_curriculum_thr"
            if params['use_neg_loss']:
                version += "_neg_loss"
        else:
            version = (
                f"base_{params['weight']}_dropout-{params['decoder_dropout']}_lr-{params['lr']}_batch-{params['batch_size']}"
                f"{kd_str}_{SIZE[0]}x{SIZE[1]}"
            )
            
            
        name = f"{params['encoder']}_{params['decoder']}"
        logger = pl.loggers.TensorBoardLogger("logs/", name=name, version=version)
        trainer = pl.Trainer(
            max_epochs=100,
            accelerator="gpu",
            devices=1,
            logger=logger,
            callbacks=[
                pl.callbacks.EarlyStopping(
                    monitor="loss/total/val", patience=10, mode="min", verbose=True
                ),
            ],
        )
        trainer.fit(model, dm)
        trainer.test(model, dm)

        # Save the model
        save_dir = f"saved_models/{name}/{version}/last.pth"
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        torch.save(model.state_dict(), save_dir)
