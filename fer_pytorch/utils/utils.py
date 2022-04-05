import os
from logging import INFO, FileHandler, Formatter, Logger, StreamHandler, getLogger
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


def init_logger(log_file_name: str) -> Logger:
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file_name)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def save_input(
    input_tensor: torch.Tensor, title: str, fig_path: str, index: int, mean: List[float], std: List[float]
) -> None:
    """Show a single image."""
    mean = np.array(mean)
    std = np.array(std)
    image = input_tensor.permute(1, 2, 0).numpy()
    image = std * image + mean
    plt.imshow(image.clip(0, 1))
    plt.title(title)
    fig_name = f"{index}.png"
    plt.savefig(os.path.join(fig_path, fig_name))


def save_batch(
    dataloader: DataLoader, class_names: List[str], fig_path: str, mean: List[float], std: List[float]
) -> None:
    """Show images for a batch."""
    x_batch, y_batch = next(iter(dataloader))
    for index, (x_item, y_item) in enumerate(zip(x_batch, y_batch)):
        save_input(x_item, class_names[y_item], fig_path, index, mean, std)
