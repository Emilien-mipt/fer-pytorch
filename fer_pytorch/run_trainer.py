import os

import hydra
import pandas as pd
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader

from fer_pytorch.augmentations import get_transforms
from fer_pytorch.train import FERPLModel
from fer_pytorch.train_test_dataset import FERDataset
from fer_pytorch.utils.utils import save_batch

CLASS_NAMES = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear"]


def run_trainer(cfg: DictConfig) -> None:
    """
    Run pytorch-lightning model.
    Args:
        cfg: hydra config
    """

    path_to_root = hydra.utils.get_original_cwd()
    path_to_dataset = os.path.join(path_to_root, cfg.dataset.path_to_dataset)

    seed_everything(cfg.general.seed, workers=True)

    train_fold = pd.read_csv(os.path.join(path_to_dataset, cfg.dataset.train_csv))
    print(f"train shape: {train_fold.shape}")
    print(train_fold.groupby([cfg.dataset.target_col]).size())

    valid_fold = pd.read_csv(os.path.join(path_to_dataset, cfg.dataset.val_csv))
    print(f"valid shape: {valid_fold.shape}")
    print(valid_fold.groupby([cfg.dataset.target_col]).size())

    train_dataset = FERDataset(
        train_fold, path_to_dataset=path_to_dataset, mode="train", transform=get_transforms(data="train", cfg=cfg)
    )
    valid_dataset = FERDataset(
        valid_fold, path_to_dataset=path_to_dataset, mode="valid", transform=get_transforms(data="valid", cfg=cfg)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.general.batch_size,
        shuffle=True,
        num_workers=cfg.general.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.general.batch_size,
        shuffle=False,
        num_workers=cfg.general.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Save batch with images after applying transforms to see the effect of augmentations
    if cfg.general.save_single_batch:
        print("Creating dir to save samples of a batch...")
        path_to_figs = "batch_imgs/"
        os.makedirs(path_to_figs, exist_ok=True)
        print("Saving figures of a single batch...")
        save_batch(train_loader, CLASS_NAMES, path_to_figs, cfg.dataset.mean, cfg.dataset.std)
        print("Figures have been saved!")

    fer_model = FERPLModel(cfg)

    filename = "".join([cfg.model.model_name, "-{epoch:02d}-{val_loss:.3f}-{val_acc:.3f}-{val_f1:.3f}"])

    checkpoint_callback = ModelCheckpoint(filename=filename, **cfg.callbacks.model_checkpoint.params)
    early_stopping_callback = EarlyStopping(**cfg.callbacks.early_stopping.params)
    lr_monitor_checkpoint = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor_checkpoint], **cfg.trainer.trainer_params
    )

    trainer.fit(model=fer_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    run_trainer(cfg)


if __name__ == "__main__":
    run()
