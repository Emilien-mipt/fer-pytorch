import argparse
import os

import pandas as pd
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader

from fer_pytorch.augmentations import get_transforms
from fer_pytorch.config import CFG
from fer_pytorch.train import FERPLModel
from fer_pytorch.train_test_dataset import FERDataset
from fer_pytorch.utils.utils import init_logger, save_batch

CLASS_NAMES = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Define whether to save train batch figs or find optimal learning rate"
    )
    parser.add_argument(
        "--save_batch_fig",
        action="store_true",
        help="Whether to save a sample of a batch or not",
    )

    args = parser.parse_args()
    save_single_batch = args.save_batch_fig

    # Path to log
    logger_path = os.path.join(CFG.LOG_DIR, CFG.OUTPUT_DIR)

    os.makedirs(os.path.join(logger_path), exist_ok=True)

    # Define logger to save train logs
    logger = init_logger(os.path.join(logger_path, "train.log"))

    # Set seed
    seed_everything(42, workers=True)

    logger.info("Reading data...")
    train_fold = pd.read_csv(CFG.TRAIN_CSV)

    logger.info("train shape: ")
    logger.info(train_fold.shape)

    valid_fold = pd.read_csv(CFG.VAL_CSV)
    logger.info("valid shape: ")
    logger.info(valid_fold.shape)

    logger.info("train fold: ")
    logger.info(train_fold.groupby([CFG.target_col]).size())
    logger.info("validation fold: ")
    logger.info(valid_fold.groupby([CFG.target_col]).size())

    train_dataset = FERDataset(train_fold, mode="train", transform=get_transforms(data="train"))
    valid_dataset = FERDataset(valid_fold, mode="valid", transform=get_transforms(data="valid"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Save batch with images after applying transforms to see the effect of augmentations
    if save_single_batch:
        logger.info("Creating dir to save samples of a batch...")
        path_to_figs = os.path.join(logger_path, "batch_figs")
        os.makedirs(path_to_figs)
        logger.info("Saving figures of a single batch...")
        save_batch(train_loader, CLASS_NAMES, path_to_figs, CFG.MEAN, CFG.STD)
        logger.info("Figures have been saved!")

    logger.info(f"Batch size {CFG.batch_size}")
    logger.info(f"Input size {CFG.size}")
    logger.info("Select CrossEntropyLoss criterion")

    precision = 32

    if CFG.MIXED_PREC is True:
        logger.info("Enable half precision")
        precision = 16

    fer_model = FERPLModel()

    checkpoint_callback = ModelCheckpoint(
        filename="fer_model-{epoch:02d}-{val_loss:.3f}-{val_acc:.3f}-{val_f1:.3f}",
        monitor="val_f1",
        mode="max",
        save_weights_only=True,
    )
    early_stopping_callback = EarlyStopping(monitor="val_loss", strict=True, mode="min", patience=CFG.early_stopping)
    lr_monitor_checkpoint = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor_checkpoint],
        max_epochs=CFG.epochs,
        strategy="ddp",
        accelerator="gpu",
        devices=[0, 1],
        precision=precision,
        default_root_dir=logger_path,
        resume_from_checkpoint=CFG.chk,
    )

    trainer.fit(model=fer_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


if __name__ == "__main__":
    main()
