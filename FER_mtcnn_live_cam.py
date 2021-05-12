import cv2
import torch
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN
from PIL import Image

from config import CFG
from model import get_model

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


def get_probs(image, test_transforms, model, device):
    image = Image.fromarray(image)
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    print(image_tensor.shape)
    input = image_tensor.to(device)
    output = model(input)
    probs = torch.nn.functional.softmax(output, dim=1).data.cpu().numpy()
    return probs[0]


def FER_live_cam():
    # Load models
    mtcnn = MTCNN(select_largest=True, device=device)
    model_classfier = get_model(CFG)
    model_classfier.to(device)
    model_classfier.eval()

    # Capture the stream from camera
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        boxes, _ = mtcnn.detect(frame, landmarks=False)
        try:
            for (x, y, w, h) in boxes:
                cv2.rectangle(frame, (x, y), (w, h), (255, 0, 0), 2)
                image = gray[int(y) : int(h), int(x) : int(w)]
                # Predict probabilities
                probs = get_probs(image, test_transforms, model_classfier, device)
                print(probs)
                # Make predictions based on probabilties
                pred = emotion_dict[probs.argmax()]
                cv2.putText(frame, pred, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        except Exception:
            pass

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    FER_live_cam()
