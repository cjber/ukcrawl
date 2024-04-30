import os
from pathlib import Path

import comet_ml  # noqa: F401
import lightning as L
from lightning import Callback, seed_everything
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    Timer,
)
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.tuner.tuning import Tuner

from model.pl_data.datamodule import DataModule
from model.pl_data.polars_dataset import PolarsDataset
from model.pl_module.class_model import WebsiteClassificationModel


def build_callbacks() -> list[Callback]:
    return [
        LearningRateMonitor(
            logging_interval="step",
            log_momentum=False,
        ),
        EarlyStopping(
            monitor="val_f1",
            mode="max",
            verbose=True,
            min_delta=0.0,
            patience=3,
        ),
        ModelCheckpoint(
            monitor="val_f1",
            mode="max",
            verbose=True,
            filename="{epoch:02d}-{val_loss:.2f}",
            save_top_k=1,
        ),
        Timer(verbose=False),
    ]


def run() -> None:
    seed_everything(42, workers=True)

    datamodule: L.LightningDataModule = DataModule(
        dataset=PolarsDataset(
            path=Path("./data/classification/classification.parquet")
        ),
        num_workers=int(os.cpu_count() or 1),
        seed=42,
        batch_size=1,
    )
    callbacks: list[Callback] = build_callbacks()

    model = WebsiteClassificationModel()
    comet_logger = CometLogger(
        api_key=os.environ.get("COMET_API_KEY"),
        project_name="website-classification",
    )
    trainer: L.Trainer = L.Trainer(
        devices="auto",
        accelerator="auto",
        deterministic=True,
        default_root_dir="ckpts",
        callbacks=callbacks,
        max_epochs=25,
        logger=comet_logger,
    )
    tuner = Tuner(trainer)
    tuner.scale_batch_size(model, datamodule=datamodule, mode="power")

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    run()
