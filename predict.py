import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from augmentations import get_transforms
from config import CFG
from model import get_model
from train_test_dataset import FERDataset


def _inference(model, test_loader, device):
    tk0 = enumerate(tqdm(test_loader))
    pred_probs = []
    for i, (images, _) in tk0:
        images = images.to(device)
        with torch.no_grad():
            y_preds = model(images)
        pred_probs.append(y_preds.softmax(1).to("cpu").numpy())
    probs = np.concatenate(pred_probs)
    return probs


def predict(test_fold, model, device):
    test_dataset = FERDataset(test_fold, mode="test", transform=get_transforms(data="valid"))
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    predictions = _inference(model, test_loader, device)
    test_fold["predictions"] = predictions.argmax(1)
    return {
        "accuracy": accuracy_score(test_fold["predictions"], test_fold["label"]),
        "f1": f1_score(test_fold["predictions"], test_fold["label"], average="weighted"),
    }


def main():
    model = get_model(CFG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    model.eval()
    test_fold = pd.read_csv(CFG.TEST_CSV)
    result_dict = predict(test_fold, model, device)
    print("Test accuracy = {:.4f}".format(result_dict["accuracy"]))
    print("Test f1 score = {:.4f}".format(result_dict["f1"]))


if __name__ == "__main__":
    main()
