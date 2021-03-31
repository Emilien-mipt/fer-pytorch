import os

import timm
import torch
import torch.nn as nn

from config import CFG


def save_model(model, epoch, trainloss, valloss, metric, optimizer, name):
    """Saves PyTorch model."""
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch,
            "train_loss": trainloss,
            "val_loss": valloss,
            "metric_loss": metric,
        },
        os.path.join(os.path.join(CFG.LOG_DIR, CFG.OUTPUT_DIR, "weights"), name),
    )


def get_model(cfg):
    """Get PyTorch model from timm library."""
    if cfg.chk:  # Loading model from the checkpoint
        print("Model:", cfg.model_name)
        model = timm.create_model(cfg.model_name, pretrained=False)
        # Changing the last layer according the number of classes
        lastlayer = list(model._modules)[-1]
        try:
            setattr(
                model,
                lastlayer,
                nn.Linear(in_features=getattr(model, lastlayer).in_features, out_features=cfg.target_size, bias=True),
            )
        except AttributeError:
            setattr(
                model,
                lastlayer,
                nn.Linear(
                    in_features=getattr(model, lastlayer)[1].in_features, out_features=cfg.target_size, bias=True
                ),
            )
        cp = torch.load(cfg.chk)
        epoch, train_loss, val_loss, metric_loss = None, None, None, None
        if "model" in cp:
            model.load_state_dict(cp["model"])
        else:
            model.load_state_dict(cp)
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
    else:  # Creating a new model
        print("Model:", cfg.model_name)
        model = timm.create_model(cfg.model_name, pretrained=cfg.pretrained)
        # Changing the last layer according the number of classes
        lastlayer = list(model._modules)[-1]
        try:
            setattr(
                model,
                lastlayer,
                nn.Linear(in_features=getattr(model, lastlayer).in_features, out_features=cfg.target_size, bias=True),
            )
        except AttributeError:
            setattr(
                model,
                lastlayer,
                nn.Linear(
                    in_features=getattr(model, lastlayer)[1].in_features, out_features=cfg.target_size, bias=True
                ),
            )
    return model
