import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from augmentations import get_transforms
from config import CFG
from model import CustomModel
from train_test_dataset import FERDataset


def _inference(model, model_state, test_loader, device):
    model.to(device)
    model.load_state_dict(torch.load(model_state)["model"])
    model.eval()
    tk0 = enumerate(tqdm(test_loader))
    pred_probs = []
    for i, (images, _) in tk0:
        images = images.to(device)
        with torch.no_grad():
            y_preds = model(images)
        pred_probs.append(y_preds.softmax(1).to("cpu").numpy())
    probs = np.concatenate(pred_probs)
    return probs

def predict(test_fold, state, device):
    model = CustomModel(CFG.model_name, pretrained=False)
    test_dataset = FERDataset(test_fold, mode="test", transform=get_transforms(data="valid"))
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    predictions = _inference(model, state, test_loader, device)
    test_fold["predictions"] = predictions.argmax(1)
    return {'accuracy': accuracy_score(test_fold["predictions"], test_fold["label"]),
                'f1': f1_score(test_fold["predictions"], test_fold["label"], average="weighted")
            }
    

def main():
    parser = argparse.ArgumentParser(description="Parse the arguments to define the dict state for the model")
    parser.add_argument(
        "--state",
        type=str,
        help="Model state, which will be used to load the model weights",
    )
    args = parser.parse_args()
    state = args.state
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_fold = pd.read_csv(CFG.TEST_CSV)
    result_dict = predict(test_fold, state, device)
    print("Test accuracy = {}".format(result_dict['accuracy']))
    print("Test f1 score = {}".format(result_dict['f1']))

if __name__ == "__main__":
    main()