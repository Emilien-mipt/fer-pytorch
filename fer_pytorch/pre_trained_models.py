from collections import namedtuple

from torch.utils import model_zoo

from fer_pytorch.model import FERModel

model = namedtuple("model", ["url", "model"])

models = {
    "resnet34": model(
        url="https://github.com/Emilien-mipt/fer-pytorch/releases/download/0.0.1/"
        "resnet34-epoch.12-val_loss.0.494-val_acc.0.846-val_f1.0.843.ckpt",
        model=FERModel,
    ),
    "mobilenetv2_140": model(
        url="https://github.com/Emilien-mipt/fer-pytorch/releases/download/1.0.1/"
        "mobilenetv2_140-epoch.12-val_loss.0.629-val_acc.0.827-val_f1.0.825.ckpt",
        model=FERModel,
    ),
}


def get_pretrained_model(model_name: str) -> FERModel:
    fer_model = models[model_name].model(model_arch=model_name, pretrained=False)
    cp = model_zoo.load_url(models[model_name].url, progress=True, map_location="cpu")
    state_dict = cp["state_dict"]
    state_dict = {k.replace("model.model.", "model."): v for k, v in state_dict.items()}
    fer_model.load_state_dict(state_dict)
    return fer_model
