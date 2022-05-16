import json
import os
import warnings
from typing import Any, Dict, List, Optional, Union

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2
from facenet_pytorch import MTCNN
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from fer_pytorch.model import FERModel
from fer_pytorch.pre_trained_models import get_pretrained_model
from fer_pytorch.train_test_dataset import FERDataset

warnings.simplefilter(action="always")

EMOTION_DICT = {0: "neutral", 1: "happiness", 2: "surprise", 3: "sadness", 4: "anger", 5: "disgust", 6: "fear"}


class FER:
    """
    The FER inference class.

    Implemented for inference of the Facial Emotion Recognition model on different types of data
    (image, list of images, video files and e.t.c.)
    """

    def __init__(self, size: int = 224, device: int = 0) -> None:
        self.device_id = device
        self.device = torch.device(f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu")
        self.size = size
        self.model: Optional[FERModel] = None
        self.mtcnn = MTCNN(keep_all=True, select_largest=True, device=self.device)

    def get_pretrained_model(self, model_name: str) -> None:
        """The method initializes the FER model and uploads the pretrained weights from the github page of the project.

        Args:
            model_name (str): The name that stands for the weights to be downloaded from the internet. The name
            coincides with the name of the model for convenience.
        """

        self.model = get_pretrained_model(model_name=model_name)
        self.model.to(self.device)
        self.model.eval()

    def load_user_weights(self, model_arch: str, path_to_weights: str) -> None:
        """The method initializes the FER model and uploads the user weights that are stored locally.

        Args:
            model_arch (str): Model architecture (timm.list_models() returns a complete list of available models in
                timm).
            path_to_weights (str): Path to the user weights to be loaded by the model.
        """

        self.model = FERModel(model_arch=model_arch, pretrained=False)
        self.model.load_weights(path_to_weights)
        self.model.to(self.device)
        self.model.eval()

    def predict_image(
        self, frame: Optional[np.ndarray], show_top: bool = False, path_to_output: Optional[str] = None
    ) -> List[dict]:
        """The method makes the prediction of the FER model on a single image.

        Args:
            frame (np.array): Input image in np.array format after it is read by OpenCV.
            show_top (bool): Whether to output only one emotion with maximum probability or all emotions with
            corresponding probabilities.
            path_to_output (str, optional): If the output path is given, the image with bounding box and top emotion
            with corresponding probability is saved.

        Returns:
            The list of dictionaries with bounding box coordinates and recognized emotion probabilities for all the
            people detected on the image.
        """

        fer_transforms = transforms.Compose(
            [
                transforms.Resize(self.size),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t.repeat(3, 1, 1)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet values
            ]
        )

        if frame is None:
            raise TypeError("A frame is None! Please, check the path to the image, when you read it with OpenCV.")

        result_list = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        boxes, _ = self.mtcnn.detect(frame, landmarks=False)

        if boxes is not None:
            for (x, y, w, h) in boxes:
                if not all(coordinate >= 0 for coordinate in (x, y, w, h)):
                    warnings.warn("Invalid face crop!")
                    continue

                image = gray[int(y) : int(h), int(x) : int(w)]
                image = Image.fromarray(image)
                image = fer_transforms(image).float()
                image_tensor = image.unsqueeze_(0)
                input = image_tensor.to(self.device)

                if self.model is not None:
                    with torch.no_grad():
                        output = self.model(input)
                else:
                    raise TypeError("Nonetype is not callable! Please, initialize the model and upload the weights.")
                probs = torch.nn.functional.softmax(output, dim=1).data.cpu().numpy()

                if show_top:
                    result_list.append(
                        {
                            "box": [x, y, w, h],
                            "top_emotion": {EMOTION_DICT[probs[0].argmax()]: np.amax(probs[0])},
                        }
                    )
                else:
                    result_list.append(
                        {
                            "box": [x, y, w, h],
                            "emotions": {
                                EMOTION_DICT[0]: probs[0, 0],
                                EMOTION_DICT[1]: probs[0, 1],
                                EMOTION_DICT[2]: probs[0, 2],
                                EMOTION_DICT[3]: probs[0, 3],
                                EMOTION_DICT[4]: probs[0, 4],
                                EMOTION_DICT[5]: probs[0, 5],
                                EMOTION_DICT[6]: probs[0, 6],
                            },
                        }
                    )
                self.visualize(frame, [x, y, w, h], EMOTION_DICT[probs[0].argmax()], np.amax(probs[0]))
        else:
            warnings.warn("No faces detected!")
        if path_to_output is not None:
            cv2.imwrite(path_to_output, frame)
        return result_list

    def predict_list_images(
        self, path_to_input: str, path_to_output: str, save_images: bool = False
    ) -> List[Dict[str, Union[str, List[float], float]]]:
        """The method makes the prediction of the FER model on a list of images.

        Args:
            path_to_input (str): Path to the folder with images.
            path_to_output (str): Path to the output folder, where the json with recognition results and optionally
            the output images are saved.
            save_images (bool): Whether to save output images or not.

        Returns:
            The list of dictionaries with bounding box coordinates, recognized top emotions and corresponding
            probabilities for each image in the folder.
        """

        if not os.path.exists(path_to_input):
            raise FileNotFoundError("Please, check the path to the input directory.")

        os.makedirs(path_to_output, exist_ok=True)

        result_list = []
        path_to_output_file = None

        list_files = os.listdir(path_to_input)

        if len(list_files) == 0:
            warnings.warn(f"The input folder {path_to_input} is empty!")

        for file_name in tqdm(list_files):
            print(file_name)
            result_dict = {"image_name": file_name}

            if save_images:
                path_to_output_file = os.path.join(path_to_output, file_name)

            file_path = os.path.join(path_to_input, file_name)
            frame = cv2.imread(file_path)

            output_list = self.predict_image(frame, show_top=True, path_to_output=path_to_output_file)

            result_dict = self.preprocess_output_list(output_list, result_dict)
            result_list.append(result_dict)

        result_json = json.dumps(result_list, allow_nan=True, indent=4)

        path_to_json = os.path.join(path_to_output, "result.json")

        with open(path_to_json, "w") as f:
            f.write(result_json)

        return result_list

    def analyze_video(self, path_to_video: str, path_to_output: str, save_video: bool = False, fps: int = 25) -> None:
        """The method makes the prediction of the FER model on a video file.

        The method saves the output json file with emotion recognition results for each frame of the input video. The
        json can be read further by Pandas and analyzed if it is needed.

        Args:
            path_to_video (str): Path to the input video file.
            path_to_output (str): Path to the output folder, where the json with recognition results and optionally
            the output video is saved.
            save_video (bool): Whether to save output video or not.
            fps (int): Number of fps for output video.
        """

        if not os.path.exists(path_to_video):
            raise FileNotFoundError("Please, check the path to the input video file.")

        result_list = []
        frame_array = []
        size = None

        filename = os.path.basename(path_to_video)

        print(f"Processing videofile {filename}...")

        os.makedirs(path_to_output, exist_ok=True)

        v_cap = cv2.VideoCapture(path_to_video)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in tqdm(range(v_len)):
            success, frame = v_cap.read()
            if not success:
                warnings.warn(f"The {i}-th frame could not be loaded. Continue processing...")
                continue

            height, width, layers = frame.shape
            size = (width, height)

            output_list = self.predict_image(frame, show_top=True)
            frame_array.append(frame)

            result_dict = {"frame_id": f"{i}"}
            result_dict = self.preprocess_output_list(output_list, result_dict)
            result_list.append(result_dict)

        result_json = json.dumps(result_list, allow_nan=True, indent=4)
        path_to_json = os.path.join(path_to_output, "result.json")

        with open(path_to_json, "w") as f:
            f.write(result_json)

        if save_video:
            path_to_video = os.path.join(path_to_output, filename)
            out = cv2.VideoWriter(path_to_video, cv2.VideoWriter_fourcc(*"DIVX"), fps, size)
            print("Writing the output videofile...")
            for i in tqdm(range(len(frame_array))):
                out.write(frame_array[i])
            out.release()

    def run_webcam(self) -> None:
        """The method makes the prediction of the FER model for the stream from the web camera and shows the results in
        real-time."""

        cap = cv2.VideoCapture(0)

        while True:
            success, frame = cap.read()

            if not success:
                warnings.warn("The frame could not be loaded. Continue processing...")
                continue

            output_list = self.predict_image(frame, show_top=True)
            print(output_list)

            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def test_fer(
        self,
        path_to_dataset: str = "fer_pytorch/dataset",
        path_to_csv: str = "fer_pytorch/dataset/new_test.csv",
        batch_size: int = 32,
        num_workers: int = 8,
    ) -> Dict[str, Any]:
        """The method is intended for convenient calculation of metrics (accuracy and f1 score) on the test part of the
        FER dataset.

        Args:
            path_to_dataset (str): Path to the folder with FER+ dataset.
            path_to_csv (str): Path to the csv file with labels for the test part.
            batch_size (int): Batch size for inference on the test part.
            num_workers (int): Number of workers to use while inference.

        Returns:
            The dictionary with accuracy and f1 score values.
        """

        test_fold = pd.read_csv(path_to_csv)

        test_transforms = A.Compose(
            [
                A.Resize(self.size, self.size),
                A.Normalize(),
                ToTensorV2(),
            ]
        )

        test_dataset = FERDataset(test_fold, path_to_dataset=path_to_dataset, mode="test", transform=test_transforms)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        pred_probs = []

        for _, (images, _) in enumerate(tqdm(test_loader)):
            images = images.to(self.device)
            if self.model is not None:
                with torch.no_grad():
                    y_preds = self.model(images)
            else:
                raise TypeError("Nonetype is not callable!")
            pred_probs.append(y_preds.softmax(1).to("cpu").numpy())
        predictions = np.concatenate(pred_probs)

        test_fold["max_prob"] = predictions.max(axis=1)
        test_fold["predictions"] = predictions.argmax(1)

        accuracy = accuracy_score(test_fold["predictions"], test_fold["label"])
        f1 = f1_score(test_fold["predictions"], test_fold["label"], average="weighted")

        return {
            "accuracy": np.round(accuracy, 2),
            "f1": np.round(f1, 2),
        }

    @staticmethod
    def json_to_pandas(json_file: str) -> pd.DataFrame:
        """The helper method to transform output json file to Pandas dataframe in convenient way.

        Args:
            json_file (str): Path to json file.

        Returns:
            The Pandas dataframe.
        """
        return pd.read_json(json_file, orient="records")

    @staticmethod
    def preprocess_output_list(output_list: list, result_dict: dict) -> dict:
        """The method is intended to process output list with recognition results to make it more convenient to save
        them in json format.

        Args:
            output_list (list): Output list with result from the prediction on a single image.
            result_dict (dict): The dictionary that is modified as a result of this method and contains all the needed
            information about FER on a single image.

        Returns:
            The dictionary with FER results for a single image.
        """
        if output_list:
            output_dict = output_list[0]
            result_dict["box"] = [round(float(n), 2) for n in output_dict["box"]]
            result_dict["emotion"] = next(iter(output_dict["top_emotion"]))
            result_dict["probability"] = round(float(next(iter(output_dict["top_emotion"].values()))), 2)
        else:
            result_dict["box"] = []
            result_dict["emotion"] = ""
            result_dict["probability"] = np.nan
        return result_dict

    @staticmethod
    def visualize(frame: Optional[np.ndarray], box_coordinates: List[float], emotion: str, prob: float) -> None:
        """The function for easy visualization.

        Args:
            frame (Optional[np.ndarray]): Input frame.
            box_coordinates (list): The list with face box coordinates.
            emotion (str): Emotion output class from the fer model.
            prob (float): Emotion output probability from the fer model.
        """
        x, y, w, h = (
            box_coordinates[0],
            box_coordinates[1],
            box_coordinates[2],
            box_coordinates[3],
        )
        cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), (255, 0, 0), 2)
        cv2.putText(
            frame,
            f"{emotion}: {prob:.2f}",
            (int(x), int(y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255.0),
            2,
        )
