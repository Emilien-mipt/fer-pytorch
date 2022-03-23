import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import CFG


def get_transforms(*, data: str) -> A.Compose:
    transforms = None
    if data == "train":
        transforms = A.Compose(
            [
                A.Resize(CFG.size, CFG.size),
                A.HorizontalFlip(p=0.5),
                A.Rotate(p=0.5, limit=(-30, 30)),
                A.RandomBrightnessContrast(
                    p=0.5, brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), brightness_by_max=False
                ),
                A.Normalize(
                    mean=CFG.MEAN,
                    std=CFG.STD,
                ),
                ToTensorV2(),
            ]
        )

    elif data == "valid":
        transforms = A.Compose(
            [
                A.Resize(CFG.size, CFG.size),
                A.Normalize(
                    mean=CFG.MEAN,
                    std=CFG.STD,
                ),
                ToTensorV2(),
            ]
        )
    return transforms
