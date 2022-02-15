import os

import cv2
import torch
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN
from PIL import Image

from config import CFG
from model import get_model

path = "../FERPlus/FER_Andrey/All"

device = torch.device(f"cuda:{CFG.GPU_ID}" if torch.cuda.is_available() else "cpu")

emotion_dict = {0: "neutral", 1: "happiness", 2: "surprise", 3: "sadness", 4: "anger", 5: "disgust", 6: "fear"}

test_transforms = transforms.Compose(
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


def write_to_file(image, emotion):
    with open("my_predict.csv", "a") as file:
        file.write(image)
        file.write(",")
        file.write(emotion)
        file.write("\n")


def get_probs(image, test_transforms, model, device):
    image = Image.fromarray(image)
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = image_tensor.to(device)
    output = model(input)
    probs = torch.nn.functional.softmax(output, dim=1).data.cpu().numpy()
    return probs[0]


def FER_image_list():
    # Load models
    mtcnn = MTCNN(select_largest=True, device=device)
    model_classfier = get_model(CFG)
    model_classfier.to(device)
    model_classfier.eval()

    # Get the list of files from the path
    image_list = os.listdir(path)
    print(image_list)

    for image in image_list:
        print(image)
        path_to_image = os.path.join(path, image)
        frame = cv2.imread(path_to_image)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        boxes, _ = mtcnn.detect(frame, landmarks=False)
        print(boxes)
        if boxes is None:
            print(f"No faces detected on image {image}!")
            write_to_file(image, "None")
        try:
            for i, (x, y, w, h) in enumerate(boxes):
                print(i)
                cv2.rectangle(frame, (x, y), (w, h), (255, 0, 0), 2)
                gray_image = gray[int(y) : int(h), int(x) : int(w)]
                # Predict probabilities
                probs = get_probs(gray_image, test_transforms, model_classfier, device)
                # Make predictions based on probabilties
                pred = emotion_dict[probs.argmax()]
                print(pred)
                write_to_file(image, pred)
        except Exception:
            print("Caught exception!")


if __name__ == "__main__":
    FER_image_list()
