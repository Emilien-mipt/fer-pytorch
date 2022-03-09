import os

import json
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


class FERInferenceDataset(Dataset):
    def __init__(self, dict_with_boxes, transform=None):
        self.df = pd.DataFrame.from_dict(dict_with_boxes)
        self.df_without_nans = self.df[~self.df.isnull().any(axis=1)]
        self.image_paths = self.df_without_nans["image_paths"].values
        self.x = self.df_without_nans["x"].values
        self.y = self.df_without_nans["y"].values
        self.w = self.df_without_nans["w"].values
        self.h = self.df_without_nans["h"].values
        self.transform = transform

    def __len__(self):
        return self.df_without_nans.shape[0]

    def __getitem__(self, idx):
        file_path = self.image_paths[idx]
        file_name = os.path.basename(file_path)
        x = self.x[idx]
        y = self.y[idx]
        w = self.w[idx]
        h = self.h[idx]
        # Read image
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[int(y) : int(h), int(x) : int(w)]
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return {"face_coordinates": np.array([x, y, w, h]), "image_name": file_name, "image": image}


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

    def predict_image(self, path_to_image, show_top=False, save_result=True):
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
                if save_result:
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
                    cv2.imwrite("./result.png", frame)
        else:
            print("No faces detected!")
        return result_list

    def detect_faces(self, path_to_folder):
        image_list = os.listdir(path_to_folder)
        result_dict = {}
        image_paths = []
        x_list = []
        y_list = []
        w_list = []
        h_list = []
        for image_name in tqdm(image_list):
            # Load image
            image_path = os.path.join(path_to_folder, image_name)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # When batch is full, detect faces and reset batch list
            boxes, _ = self.mtcnn.detect(image, landmarks=False)
            image_paths.append(image_path)
            if boxes is not None:
                box = boxes[0]
                x_list.append(box[0])
                y_list.append(box[1])
                w_list.append(box[2])
                h_list.append(box[3])
            else:
                x_list.append(np.nan)
                y_list.append(np.nan)
                w_list.append(np.nan)
                h_list.append(np.nan)
        result_dict["image_paths"] = image_paths
        result_dict["x"] = x_list
        result_dict["y"] = y_list
        result_dict["w"] = w_list
        result_dict["h"] = h_list
        return result_dict

    def predict_list_images(self, path_to_folder, batch_size):
        result_dict = {}
        dict_with_boxes = self.detect_faces(path_to_folder)
        print(dict_with_boxes)
        inference_dataset = FERInferenceDataset(
            dict_with_boxes=dict_with_boxes, transform=get_transforms(data="valid")
        )
        inference_loader = DataLoader(
            inference_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
        image_names = []
        box_coordinates = []
        pred_probs = []
        # Inference loop
        for inference_batch in tqdm(inference_loader):
            # Append image names
            names = inference_batch["image_name"]
            image_names.append(names)
            print(image_names)
            # Append box coordinates
            face_coordinates = inference_batch["face_coordinates"]
            box_coordinates.append(face_coordinates)
            print(box_coordinates)
            # Inference on the batch
            images = inference_batch["image"]
            images = images.to(self.device)
            with torch.no_grad():
                y_preds = self.model(images)
            pred_probs.append(y_preds.softmax(1).to("cpu").numpy())
        # Concatenate output arrays
        image_names = np.concatenate(image_names)
        box_coordinates = np.concatenate(box_coordinates)
        pred_probs = np.concatenate(pred_probs)
        # Get prediction classes
        emotion_index_array = pred_probs.argmax(1)
        emotion_class_array = np.array([emotion_dict[idx] for idx in emotion_index_array])
        # Get max probabilty values
        max_prob_array = pred_probs.max(axis=1)
        # Fill the result dictionary
        result_dict["image_name"] = image_names.tolist()
        result_dict["box_coordinate"] = box_coordinates.tolist()
        result_dict["emotion"] = emotion_class_array.tolist()
        result_dict["probability"] = max_prob_array.tolist()
        # Convert dictionary of lists to the list of dictionaries to make it convenient in json format
        result_list = [dict(zip(result_dict, t)) for t in zip(*result_dict.values())]
        # Write json with results
        result_json = json.dumps(result_list, indent=4)
        with open('result.json', 'w') as f:
            f.write(result_json)
        return result_dict

    def analyze_video(self, path_to_video):
        pass

    def run_webcam(self, camera_id):
        pass
