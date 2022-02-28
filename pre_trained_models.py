from collections import namedtuple
from torch.utils import model_zoo
from model import FERModel

from config import CFG

model = namedtuple("model", ["url", "model"])

models = {
    "resnet34_best": model(
        url="https://github.com/Emilien-mipt/FERplus-Pytorch/releases/download/0.0.1/resnet34_best.pt",
        model=FERModel,
    )
}


def get_pretrained_model(model_name: str):
    model = models[model_name].model(model_arch=CFG.model_name, pretrained=False)
    weights = model_zoo.load_url(models[model_name].url, progress=True, map_location="cpu")
    if "model" in weights:
        state_dict = weights["model"]
    else:
        state_dict = weights
    # Loading stated dict
    state_dict = {".".join(["model", k]): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    if "epoch" in weights:
        epoch = int(weights["epoch"])
    if "train_loss" in weights:
        train_loss = weights["train_loss"]
    if "val_loss" in weights:
        val_loss = weights["val_loss"]
    if "metric_loss" in weights:
        metric_loss = weights["metric_loss"]
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
    return model
