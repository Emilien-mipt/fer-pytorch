import math
import os
import random
import time
from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def get_score(y_true, y_pred, metric):
    if metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    if metric == "f1_score":
        return f1_score(y_true, y_pred, average="weighted")


def init_logger(log_file_name):
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file_name)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def seed_torch(seed=42):
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

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "{} (remain {})".format(asMinutes(s), asMinutes(rs))


def save_input(input_tensor, title, fig_path, index, config):
    """Show a single image."""
    mean = np.array(config.MEAN)
    std = np.array(config.STD)
    image = input_tensor.permute(1, 2, 0).numpy()
    image = std * image + mean
    plt.imshow(image.clip(0, 1))
    plt.title(title)
    fig_name = f"{index}.png"
    fig_path = fig_path
    plt.savefig(os.path.join(fig_path, fig_name))


def save_batch(dataloader, class_names, fig_path, config):
    """Show images for a batch."""
    X_batch, y_batch = next(iter(dataloader))
    for index, (x_item, y_item) in enumerate(zip(X_batch, y_batch)):
        save_input(x_item, class_names[y_item], fig_path, index, config)


def weight_class(data_df):
    class_labels = [i for i in range(5)]
    count_classes = []
    for class_n in class_labels:
        count_class = data_df[data_df["label"] == class_n].shape[0]
        print(f"Total train images for class {class_n}: {count_class}")
        count_classes.append(count_class)
    min_class = min(count_classes)
    weight_tensor = min_class / np.array(count_classes)
    return weight_tensor
