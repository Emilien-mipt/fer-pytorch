import timm
import torch.nn as nn

from config import CFG


# ====================================================
# MODEL
# ====================================================
class CustomModel(nn.Module):
    def __init__(self, model_arch=CFG.model_name, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, CFG.target_size)

    def forward(self, x):
        x = self.model(x)
        return x
