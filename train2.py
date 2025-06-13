import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from utils import (
    build_hyperparameters,
    build_student_model,
    compute_loss,
    compute_metrics,
    colorize_segmentation
)
from dataset.ugm import UGMDataModule
import os


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
        self.T = kwargs.get("T", 1.0)
        self.entropy_threshold = kwargs.get("entropy_threshold", 1.0)
        self.use_refinement = kwargs.get("use_refinement", False)
        self.use_curriculum_thr = kwargs.get("use_curriculum_thr", False)
        self.use_neg_loss = kwargs.get("use_neg_loss", False)

        # Models
        self.student = build_student_model(
            encoder=encoder, decoder=decoder, weight=weight, dropout=decoder_dropout
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=2, verbose=True)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "monitor": "loss/total/val"}]

    def step(self, batch, stage="train"):
        if self.use_refinement:
            images, t_logits, pseudo_labels = batch
        else:
            images, t_logits = batch
            t_probs = F.softmax(t_logits, dim=1)
            _, pseudo_labels = t_probs.max(dim=1)
            
        s_logits = self.student(images)
        preds = torch.argmax(s_logits, dim=1)
        
        loss = compute_loss(
            s_logits,
            pseudo_labels,
            t_logits if self.use_kd else None,
            use_kd=self.use_kd,
            kd_weight=self.kd_weight,
            T=self.T,
        )
            
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
        self.student.train()
        output = self.step(batch, stage="train")
        self.logging(output, stage="train")
        return output["loss"]["total"]

    def validation_step(self, batch, _):
        self.student.eval()
        with torch.no_grad():
            output = self.step(batch, stage="val")
        self.logging(output, stage="val")
        return output["loss"]["total"]

    def test_step(self, batch, _):
        self.student.eval()
        with torch.no_grad():
            output = self.step(batch, stage="test")
        self.logging(output, stage="test")
        return output["loss"]["total"]

if __name__ == "__main__":
    torch.hub.set_dir("/media/esr/ssd0/cache")
    pl.seed_everything(42, workers=True)

    # Parameters
    USE_KD = [True]
    T = [2.0]
    KD_WEIGHT = [0.001]
    LR = [1e-3]
    ENCODER = ["timm-efficientnet-b0"]
    DECODER = ["unet"]
    WEIGHT = ["imagenet"]
    DECODER_DROPOUT = [0.5]
    ENTROPY_THRESHOLD = [1.0, 0.05, 0.15, 0.35, 0.45]
    USE_REFINEMENT = [True]
    USE_CURRICULUM_THR = [False]
    USE_NEG_LOSS = [False]
    
    SIZE = (256, 256)
    BATCH_SIZE = 12  # Batch sizes

    hyperparameter_list = {
        "use_kd": USE_KD,
        "T": T,
        "kd_weight": KD_WEIGHT,
        "lr": LR,
        "encoder": ENCODER,
        "decoder": DECODER,
        "weight": WEIGHT,
        "decoder_dropout": DECODER_DROPOUT,
        "entropy_threshold": ENTROPY_THRESHOLD,
        "use_refinement": USE_REFINEMENT,
        "use_curriculum_thr": USE_CURRICULUM_THR,
        "use_neg_loss": USE_NEG_LOSS,
    }
    combinations_dict = build_hyperparameters(hyperparameter_list)

    print(f"Total combinations: {len(combinations_dict)}")

    for i, params in enumerate(combinations_dict):
        model = BaseModel(
            encoder=params["encoder"],
            decoder=params["decoder"],
            weight=params["weight"],
            use_kd=params["use_kd"],
            T=params["T"],
            kd_weight=params["kd_weight"],
            lr=params["lr"],
            decoder_dropout=params["decoder_dropout"],
            entropy_threshold=params["entropy_threshold"],
            use_refinement=params["use_refinement"],
            use_curriculum_thr=params["use_curriculum_thr"],
            use_neg_loss=params["use_neg_loss"],
        )

        dm = UGMDataModule(
            root="/media/esr/ssd0/dataset/2025-01-10/",
            return_teacher_logits=True,
            use_refinement=params["use_refinement"],
            entropy_threshold=params["entropy_threshold"],
            batch_size=BATCH_SIZE,
            size=SIZE,
        )
        dm.setup()

        kd_str = f"_kd_T{params['T']}_w-{params['kd_weight']}" if params["use_kd"] else ""
        version = (
            f"{params['weight']}_dropout-{params['decoder_dropout']}_lr-{params['lr']}"
            f"{kd_str}_{SIZE[0]}x{SIZE[1]}"
        )
        if params['use_refinement']:
            if params['entropy_threshold'] < 1.0:
                prefix = f"refinement_thr-{params['entropy_threshold']}"
            else:
                prefix = "refinement"
            version = f"{prefix}_{version}"
        if params['use_curriculum_thr']:
            prefix = "curriculum_thr"
            version = f"{prefix}_{version}"
        if params['use_neg_loss']:
            prefix = "neg_loss"
            version = f"{prefix}_{version}"
            
        name = f"{params['encoder']}_{params['decoder']}"
        logger = pl.loggers.TensorBoardLogger("logs/", name=name, version=version)
        trainer = pl.Trainer(
            max_epochs=100,
            accelerator="gpu",
            devices=1,
            logger=logger,
            callbacks=[
                pl.callbacks.EarlyStopping(
                    monitor="loss/total/val", patience=7, mode="min", verbose=True
                ),
            ],
        )
        trainer.fit(model, dm)
        trainer.test(model, dm)

        # Save the model
        save_dir = f"saved_models/{name}/{version}/last.pth"
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        torch.save(model.state_dict(), save_dir)
