import json
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from augmentations import get_transforms
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
        self.mtcnn = MTCNN(keep_all=True, select_largest=True, device=self.device)

    def get_pretrained_model(self, model_name):
        self.model = get_pretrained_model(model_name)
        self.model.to(self.device)
        self.model.eval()

    def load_weights(self, path_to_weigths):
        fer_model = FERModel(model_arch=CFG.model_name, pretrained=False)
        self.model = fer_model.load_weights(path_to_weigths)
        self.model.to(self.device)
        self.model.eval()

    def predict_image(self, path_to_image, show_top=False, save_image=True, path_to_output="."):
        if not os.path.isfile(path_to_image):
            FileNotFoundError("File not found!")
        result_list = []
        frame = cv2.imread(path_to_image)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        boxes, _ = self.mtcnn.detect(frame, landmarks=False)
        if boxes is not None:
            for (x, y, w, h) in boxes:
                image = gray[int(y) : int(h), int(x) : int(w)]
                # Convert to PIL image
                image = Image.fromarray(image)
                # Apply tranformations
                image = fer_transforms(image).float()
                image_tensor = image.unsqueeze_(0)
                # Apply model to the transformed image
                input = image_tensor.to(self.device)
                output = self.model(input)
                # Predict probabilities
                probs = torch.nn.functional.softmax(output, dim=1).data.cpu().numpy()
                # Return results
                if show_top:
                    result_list.append(
                        {
                            "box": [x, y, w, h],
                            "top_emotion": {emotion_dict[probs[0].argmax()]: np.amax(probs[0])},
                        }
                    )
                else:
                    result_list.append(
                        {
                            "box": [x, y, w, h],
                            "emotions": {
                                emotion_dict[0]: probs[0, 0],
                                emotion_dict[1]: probs[0, 1],
                                emotion_dict[2]: probs[0, 2],
                                emotion_dict[3]: probs[0, 3],
                                emotion_dict[4]: probs[0, 4],
                                emotion_dict[5]: probs[0, 5],
                                emotion_dict[6]: probs[0, 6],
                            },
                        }
                    )
                if save_image:
                    cv2.rectangle(frame, (x, y), (w, h), (255, 0, 0), 2)
                    cv2.putText(
                        frame,
                        f"{emotion_dict[probs[0].argmax()]}: {np.amax(probs[0]):.2f}",
                        (x, int(y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255.0),
                        2,
                    )
                    image_name = os.path.basename(path_to_image)
                    save_path = os.path.join(path_to_output, image_name)
                    cv2.imwrite(save_path, frame)
        else:
            print("No faces detected!")
        return result_list

    def predict_list_images(self, path_to_folder, save_images=False):
        path_to_output = "./output_images"
        if save_images:
            os.makedirs(path_to_output, exist_ok=True)

        result_list = []
        list_files = os.listdir(path_to_folder)
        for file_name in tqdm(list_files):
            print(file_name)
            result_dict = {}
            file_path = os.path.join(path_to_folder, file_name)
            print(path_to_output)
            output_list = self.predict_image(
                file_path, show_top=True, save_image=save_images, path_to_output=path_to_output
            )
            result_dict["image_name"] = file_name
            if output_list:
                output_dict = output_list[0]
                result_dict["box"] = [round(float(n), 2) for n in output_dict["box"]]
                result_dict["emotion"] = [k for k in output_dict["top_emotion"]][0]
                result_dict["probabilty"] = round(
                    float([output_dict["top_emotion"][k] for k in output_dict["top_emotion"]][0]), 2
                )
            else:
                result_dict["box"] = []
                result_dict["emotion"] = ""
                result_dict["probabilty"] = np.nan
            result_list.append(result_dict)

        result_json = json.dumps(result_list, allow_nan=True, indent=4)
        with open("result.json", "w") as f:
            f.write(result_json)
        return result_json

    def analyze_video(self, path_to_video):
        pass

    def run_webcam(self, camera_id):
        pass
