import json
import os
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from fer_pytorch.augmentations import get_transforms
from fer_pytorch.config import CFG
from fer_pytorch.model import FERModel
from fer_pytorch.pre_trained_models import get_pretrained_model
from fer_pytorch.train_test_dataset import FERDataset

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
    """
    The FER inference class.

    Implemented for inference of the Facial Emotion Recognition model on different types of data
    (image, list of images, video files and e.t.c.)
    """

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[FERModel] = None
        self.mtcnn = MTCNN(keep_all=True, select_largest=True, device=self.device)

    def get_pretrained_model(self, model_name: str) -> None:
        """The method initializes the FER model and uploads the pretrained weights from the internet.

        Args:
            model_name (str): The name that stands for the weights to be uploaded.
        """
        self.model = get_pretrained_model(model_name)
        self.model.to(self.device)
        self.model.eval()

    def load_user_weights(self, path_to_weigths: str) -> None:
        """The method initializes the FER model and uploads the user weights that are stored locally.

        Args:
            path_to_weigths (str): Path to the user weights to be uploaded.
        """
        self.model = FERModel(model_arch=CFG.model_name, pretrained=False)
        self.model.load_weights(path_to_weigths)
        self.model.to(self.device)
        self.model.eval()

    def predict_image(
        self, frame: np.array, show_top: bool = False, path_to_output: Optional[str] = None
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

        result_list = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        boxes, _ = self.mtcnn.detect(frame, landmarks=False)

        if boxes is not None:
            for (x, y, w, h) in boxes:
                image = gray[int(y) : int(h), int(x) : int(w)]
                image = Image.fromarray(image)
                image = fer_transforms(image).float()
                image_tensor = image.unsqueeze_(0)
                input = image_tensor.to(self.device)
                if self.model is not None:
                    output = self.model(input)
                else:
                    raise TypeError("Nonetype is not callable!")
                probs = torch.nn.functional.softmax(output, dim=1).data.cpu().numpy()

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

                if path_to_output is not None:
                    cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), (255, 0, 0), 2)
                    cv2.putText(
                        frame,
                        f"{emotion_dict[probs[0].argmax()]}: {np.amax(probs[0]):.2f}",
                        (int(x), int(y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255.0),
                        2,
                    )
                    cv2.imwrite(path_to_output, frame)
        else:
            print("No faces detected!")
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

        os.makedirs(path_to_output, exist_ok=True)

        result_list = []
        path_to_output_file = None

        list_files = os.listdir(path_to_input)

        for file_name in tqdm(list_files):
            print(file_name)
            result_dict = {"image_name": file_name}

            if save_images:
                path_to_output_file = os.path.join(path_to_output, file_name)

            file_path = os.path.join(path_to_input, file_name)
            frame = cv2.imread(file_path)

            output_list = self.predict_image(frame, show_top=True, path_to_output=path_to_output_file)

            result_dict = self._preprocess_output_list(output_list, result_dict)
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
                print(f"The {i}-th frame could not be loaded. Continue processing...")
                continue

            height, width, layers = frame.shape
            size = (width, height)

            output_list = self.predict_image(frame, show_top=True)

            result_dict = {"frame_id": f"{i}"}
            result_dict = self._preprocess_output_list(output_list, result_dict)
            result_list.append(result_dict)

            if output_list:
                x, y, w, h = result_dict["box"][0], result_dict["box"][1], result_dict["box"][2], result_dict["box"][3]
                cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    f"{result_dict['emotion']}: {result_dict['probability']}",
                    (int(x), int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255.0),
                    2,
                )

            frame_array.append(frame)

        result_json = json.dumps(result_list, allow_nan=True, indent=4)
        path_to_json = os.path.join(path_to_output, "result.json")

        with open(path_to_json, "w") as f:
            f.write(result_json)

        if save_video is not None:
            path_to_video = os.path.join(path_to_output, filename)
            out = cv2.VideoWriter(path_to_video, cv2.VideoWriter_fourcc(*"DIVX"), fps, size)
            print("Writing videofile...")
            for i in tqdm(range(len(frame_array))):
                out.write(frame_array[i])
            out.release()

    def run_webcam(self) -> None:
        """The method makes the prediction of the FER model for the stream from the web camera and shows the results in
        real-time."""

        cap = cv2.VideoCapture(0)

        while True:
            result_dict: dict = {}

            success, frame = cap.read()

            if not success:
                print("The frame could not be loaded. Continue processing...")
                continue

            output_list = self.predict_image(frame, show_top=True)

            result_dict = self._preprocess_output_list(output_list, result_dict)

            if output_list:
                x, y, w, h = result_dict["box"][0], result_dict["box"][1], result_dict["box"][2], result_dict["box"][3]
                cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    f"{result_dict['emotion']}: {result_dict['probability']}",
                    (int(x), int(y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255.0),
                    2,
                )

            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

    def test_fer(self) -> Dict[str, Any]:
        """The method is intended for convenient calculation of metrics (accuracy and f1 score) on the test part of the
        FER dataset.

        Returns:
            The dictionary with accuracy and f1 score values.
        """

        test_fold = pd.read_csv(CFG.TEST_CSV)

        test_dataset = FERDataset(test_fold, mode="test", transform=get_transforms(data="valid"))
        test_loader = DataLoader(
            test_dataset,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
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
    def _preprocess_output_list(output_list: list, result_dict: dict) -> dict:
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
