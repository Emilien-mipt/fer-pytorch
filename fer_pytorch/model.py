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

    def load_weights(self, path_to_weights: str) -> None:
        cp = torch.load(path_to_weights)
        state_dict = cp["state_dict"]
        state_dict = {k.replace("model.model.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
