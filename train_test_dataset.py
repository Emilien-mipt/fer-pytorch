import cv2
import torch
from torch.utils.data import Dataset

from config import CFG


# ====================================================
# Dataset creation
# ====================================================
class FERDataset(Dataset):
    def __init__(self, df, mode=None, transform=None):
        self.df = df
        self.file_names = df["image_id"].values
        self.labels = df["label"].values
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        # Select mode
        if self.mode == "train":
            file_path = f"{CFG.TRAIN_PATH}/{file_name}"
        elif self.mode == "valid":
            file_path = f"{CFG.VAL_PATH}/{file_name}"
        else:
            file_path = f"{CFG.TEST_PATH}/{file_name}"
        # Read images
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        label = torch.tensor(self.labels[idx]).long()
        return image, label
