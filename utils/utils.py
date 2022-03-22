import math
import os
import random
import time
from logging import INFO, FileHandler, Formatter, Logger, StreamHandler, getLogger
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader


def get_score(y_true: np.array, y_pred: np.array, metric: str) -> Any:
    score = None
    if metric == "accuracy":
        score = accuracy_score(y_true, y_pred)
    if metric == "f1_score":
        score = f1_score(y_true, y_pred, average="weighted")
    return score


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


def seed_torch(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_printoptions(precision=10)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s: float) -> str:
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since: float, percent: float) -> str:
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return f"{asMinutes(s)} (remain {asMinutes(rs)})"


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
    fig_path = fig_path
    plt.savefig(os.path.join(fig_path, fig_name))


def save_batch(
    dataloader: DataLoader, class_names: List[str], fig_path: str, mean: List[float], std: List[float]
) -> None:
    """Show images for a batch."""
    x_batch, y_batch = next(iter(dataloader))
    for index, (x_item, y_item) in enumerate(zip(x_batch, y_batch)):
        save_input(x_item, class_names[y_item], fig_path, index, mean, std)
