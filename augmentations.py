from albumentations import (
    CenterCrop,
    CoarseDropout,
    Compose,
    Cutout,
    HorizontalFlip,
    Normalize,
    RandomBrightnessContrast,
    RandomResizedCrop,
    RandomRotate90,
    Resize,
    ShiftScaleRotate,
    Transpose,
    VerticalFlip,
)
from albumentations.pytorch import ToTensorV2

from config import CFG


# ====================================================
# Transforms
# ====================================================
def get_transforms(*, data):

    if data == "train":
        return Compose(
            [
                Resize(CFG.size, CFG.size, p=1.0),
                Normalize(
                    mean=CFG.MEAN,
                    std=CFG.STD,
                ),
                ToTensorV2(),
            ]
        )

    elif data == "valid":
        return Compose(
            [
                Resize(CFG.size, CFG.size),
                Normalize(
                    mean=CFG.MEAN,
                    std=CFG.STD,
                ),
                ToTensorV2(),
            ]
        )
