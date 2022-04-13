import importlib
import os
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """
    Extract an object from a given path.
    https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from `{obj_path}`.")
    return getattr(module_obj, obj_name)


def save_input(input_tensor: torch.Tensor, title: str, fig_path: str, index: int, mean: Any, std: Any) -> None:
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
