from typing import Any, Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch.nn import functional as F
from torch.optim import Optimizer

from fer_pytorch.config import CFG
from fer_pytorch.model import FERModel


class FERPLModel(pl.LightningModule):
    """
    The FER Pytorch Lightning class.

    Implemented for training and validation of the Facial Emotion Recognition model on FER+ dataset.
    """

    def __init__(self) -> None:
        super().__init__()
        self.accuracy = torchmetrics.Accuracy()
        self.f1_score = torchmetrics.F1Score(num_classes=CFG.target_size, average="weighted")
        self.model = FERModel(model_arch=CFG.model_name, pretrained=CFG.pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self) -> Tuple[Union[Optimizer, List[Optimizer], List[LightningOptimizer]], List[Any]]:
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=CFG.lr, momentum=CFG.momentum, weight_decay=CFG.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=CFG.min_lr, max_lr=CFG.lr, mode="triangular2", step_size_up=1319
        )
        scheduler = {"scheduler": lr_scheduler, "interval": "step"}
        return [optimizer], [scheduler]

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, labels = batch
        predictions = self.model(images)
        predicted_classes = predictions.argmax(dim=1)
        loss = F.cross_entropy(predictions, labels)
        acc = self.accuracy(predicted_classes, labels)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
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

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        return self.validation_step(batch, batch_idx)
