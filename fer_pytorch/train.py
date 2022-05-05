from typing import Any, List, Tuple, Union

import pytorch_lightning as pl
import torch
import torchmetrics
from omegaconf import DictConfig
from torch.nn import functional as F

from fer_pytorch.model import FERModel
from fer_pytorch.utils.utils import load_obj


class FERPLModel(pl.LightningModule):
    """
    The FER Pytorch Lightning class.

    Implemented for training and validation of the Facial Emotion Recognition model on FER+ dataset.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.accuracy = torchmetrics.Accuracy()
        self.f1_score = torchmetrics.F1Score(num_classes=self.cfg.dataset.target_size, average="weighted")

        self.model = FERModel(
            model_arch=self.cfg.model.model_name,
            pretrained=self.cfg.model.pretrained,
            num_classes=cfg.dataset.target_size,
        )

    def forward(self, x: torch.Tensor):  # type: ignore
        return self.model(x)

    def configure_optimizers(self) -> Union[Tuple[List[Any], List[Any]], List[Any]]:
        optimizer = load_obj(self.cfg.optimizer.class_name)(self.model.parameters(), **self.cfg.optimizer.params)
        if self.cfg.scheduler.class_name is not None:
            lr_scheduler = load_obj(self.cfg.scheduler.class_name)(optimizer, **self.cfg.scheduler.params)
            scheduler = {"scheduler": lr_scheduler, "interval": "step"}
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):  # type: ignore
        images, labels = batch
        predictions = self.model(images)
        predicted_classes = predictions.argmax(dim=1)
        loss = F.cross_entropy(predictions, labels)
        acc = self.accuracy(predicted_classes, labels)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):  # type: ignore
        images, labels = batch
        predictions = self.model(images)
        predicted_classes = predictions.argmax(dim=1)
        loss = F.cross_entropy(predictions, labels)
        acc = self.accuracy(predicted_classes, labels)
        f1 = self.f1_score(predicted_classes, labels)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log("val_f1", f1, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        return {"Accuracy": acc, "F1_score": f1}

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):  # type: ignore
        return self.validation_step(batch, batch_idx)
