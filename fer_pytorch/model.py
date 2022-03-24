import os

import timm
import torch
import torch.nn as nn

from fer_pytorch.config import CFG


class FERModel(nn.Module):
    def __init__(self, model_arch: str = CFG.model_name, pretrained: bool = CFG.pretrained):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained, num_classes=CFG.target_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

    def save(self, epoch: int, trainloss: float, valloss: float, metric: float, name: str) -> None:

        torch.save(
            {
                "model": self.model.state_dict(),
                "epoch": epoch,
                "train_loss": trainloss,
                "val_loss": valloss,
                "metric_loss": metric,
            },
            os.path.join(os.path.join(CFG.LOG_DIR, CFG.OUTPUT_DIR, "weights"), name),
        )

    def load_weights(self, path_to_weights: str) -> None:
        cp = torch.load(path_to_weights)
        epoch, train_loss, val_loss, metric_loss = None, None, None, None
        if "model" in cp:
            self.model.load_state_dict(cp["model"])
        else:
            self.model.load_state_dict(cp)
        if "epoch" in cp:
            epoch = int(cp["epoch"])
        if "train_loss" in cp:
            train_loss = cp["train_loss"]
        if "val_loss" in cp:
            val_loss = cp["val_loss"]
        if "metric_loss" in cp:
            metric_loss = cp["metric_loss"]
        print(
            "Uploading model from the checkpoint...",
            "\nEpoch:",
            epoch,
            "\nTrain Loss:",
            train_loss,
            "\nVal Loss:",
            val_loss,
            "\nMetrics:",
            metric_loss,
        )
