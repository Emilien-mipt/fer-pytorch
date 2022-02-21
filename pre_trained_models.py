from collections import namedtuple
from torch.utils import model_zoo
from model import FERModel

import torch

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
    state_dict = weights["model"]
    state_dict = {'.'.join(['model', k]): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model

fer_model = get_pretrained_model('resnet34_best')
print(fer_model)