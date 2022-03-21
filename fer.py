import json
import os
from typing import Any, Collection, Dict, List, Union

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

from augmentations import get_transforms
from config import CFG
from model import FERModel
from pre_trained_models import get_pretrained_model
from train_test_dataset import FERDataset

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

    def get_pretrained_model(self, model_name: str) -> None:
        self.model = get_pretrained_model(model_name)
        self.model.to(self.device)
        self.model.eval()

    def load_user_weights(self, path_to_weigths: str) -> None:
        self.model = FERModel(model_arch=CFG.model_name, pretrained=False)
        self.model.load_weights(path_to_weigths)
        self.model.to(self.device)
        self.model.eval()

    def predict_image(
        self, frame: np.array, show_top: bool = False, path_to_output: str = None
    ) -> List[Dict[str, dict]]:
        result_list = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        boxes, _ = self.mtcnn.detect(frame, landmarks=False)

        if boxes is not None:
            for (x, y, w, h) in boxes:
                image = gray[int(y): int(h), int(x): int(w)]
                image = Image.fromarray(image)
                image = fer_transforms(image).float()
                image_tensor = image.unsqueeze_(0)
                input = image_tensor.to(self.device)
                output = self.model(input)
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

    def analyze_video(self, path_to_video: str, video_name: str = None, fps: int = 25) -> None:
        result_list = []
        frame_array = []
        size = None

        pathname, extension = os.path.splitext(path_to_video)
        filename = pathname.split("/")[-1]

        print(f"Processing videofile {filename}...")

        path_to_output_dir = f"./{filename}"
        os.makedirs(path_to_output_dir, exist_ok=True)

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
        path_to_json = os.path.join(path_to_output_dir, "result.json")

        with open(path_to_json, "w") as f:
            f.write(result_json)

        if video_name is not None:
            path_to_video = os.path.join(path_to_output_dir, video_name)
            out = cv2.VideoWriter(path_to_video, cv2.VideoWriter_fourcc(*"DIVX"), fps, size)
            print("Writing videofile...")
            for i in tqdm(range(len(frame_array))):
                out.write(frame_array[i])
            out.release()

    def run_webcam(self) -> None:
        cap = cv2.VideoCapture(0)

        while True:
            result_dict: dict = {}
            success, frame = cap.read()

            if not success:
                print(f"The frame could not be loaded. Continue processing...")
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

        for i, (images, _) in enumerate(tqdm(test_loader)):
            images = images.to(self.device)
            with torch.no_grad():
                y_preds = self.model(images)
            pred_probs.append(y_preds.softmax(1).to("cpu").numpy())
        predictions = np.concatenate(pred_probs)

        test_fold["max_prob"] = predictions.max(axis=1)
        test_fold["predictions"] = predictions.argmax(1)

        accuracy = accuracy_score(test_fold["predictions"], test_fold["label"])
        f1 = f1_score(test_fold["predictions"], test_fold["label"], average="weighted")

        return {
            "accuracy": accuracy,
            "f1": f1,
        }

    @staticmethod
    def json_to_pandas(json_file: str) -> pd.DataFrame:
        return pd.read_json(json_file, orient="records")

    @staticmethod
    def _preprocess_output_list(output_list: list, input_dict: dict) -> dict:
        if output_list:
            output_dict = output_list[0]
            input_dict["box"] = [round(float(n), 2) for n in output_dict["box"]]
            input_dict["emotion"] = [k for k in output_dict["top_emotion"]][0]
            input_dict["probability"] = round(
                float([output_dict["top_emotion"][k] for k in output_dict["top_emotion"]][0]), 2
            )
        else:
            input_dict["box"] = []
            input_dict["emotion"] = ""
            input_dict["probability"] = np.nan
        return input_dict
