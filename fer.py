import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN
from PIL import Image

from config import CFG
from model import FERModel
from pre_trained_models import get_pretrained_model

emotion_dict = {0: "neutral", 1: "happiness", 2: "surprise", 3: "sadness", 4: "anger", 5: "disgust", 6: "fear"}

fer_transforms = transforms.Compose(
    [
        transforms.Resize(CFG.size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(
            mean=CFG.MEAN,
            std=CFG.STD,
        ),
    ]
)


class FER:
    def __init__(self):
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.mtcnn = MTCNN(select_largest=True, device=self.device)

    def get_pretrained_model(self, model_name):
        self.model = get_pretrained_model(model_name)
        self.model.to(self.device)
        self.model.eval()

    def load_weights(self, path_to_weigths):
        fer_model = FERModel(model_arch=CFG.model_name, pretrained=False)
        self.model = fer_model.load_weights(path_to_weigths)
        self.model.to(self.device)
        self.model.eval()

    def predict_image(self, path_to_image, show_top=False, save_result=True):
        pass

    def predict_list_images(self, path_to_folder):
        pass

    def analyze_video(self, path_to_video):
        pass

    def run_webcam(self, camera_id):
        pass
