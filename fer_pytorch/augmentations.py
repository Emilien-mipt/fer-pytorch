import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig


def get_transforms(*, data: str, cfg: DictConfig) -> A.Compose:
    transforms = None
    if data == "train":
        transforms = A.Compose(
            [
                A.Resize(cfg.dataset.size, cfg.dataset.size),
                A.HorizontalFlip(p=0.5),
                A.Rotate(p=0.5, limit=(-30, 30)),
                A.RandomBrightnessContrast(
                    p=0.5, brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), brightness_by_max=False
                ),
                A.Normalize(
                    mean=cfg.dataset.mean,
                    std=cfg.dataset.std,
                ),
                ToTensorV2(),
            ]
        )

    elif data == "valid":
        transforms = A.Compose(
            [
                A.Resize(cfg.dataset.size, cfg.dataset.size),
                A.Normalize(
                    mean=cfg.dataset.mean,
                    std=cfg.dataset.std,
                ),
                ToTensorV2(),
            ]
        )
    return transforms
