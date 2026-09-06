import torch
import pytorch_lightning as pl
from dataset.ugm import UGMDataModule
from mean_teacher import MeanTeacher

if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    
    model = MeanTeacher(lr=1e-3)
    
    dm = UGMDataModule(
        root="/media/esr/ssd0/dataset/2025-01-10/",
        return_teacher_logits=True,
        batch_size=12,
        size=(256, 256),
    )
    dm.setup()

    logger = pl.loggers.TensorBoardLogger("logs/", name="MeanTeacher", version="v1")
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="loss/total/val", patience=7, mode="min", verbose=True
            ),
            pl.callbacks.ModelCheckpoint(
                monitor="loss/total/val",
                dirpath="saved_models/MeanTeacher/v1",
                filename="best",
                save_top_k=1,
                mode="min",
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
        ]
    )
    trainer.fit(model, dm)
    trainer.test(model, dm)