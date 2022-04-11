import cv2

from fer_pytorch.fer import FER
from fer_pytorch.inference_config import CFG

cfg = CFG()

cfg.device_id = 1

fer = FER(cfg)
print(fer.device)

fer.get_pretrained_model(model_name="resnet34")

frame = cv2.imread("tests/test_images/happy.jpg")
result = fer.predict_image(frame)
print(result)

result = fer.predict_image(frame, show_top=True)
print(result)

result_json_list = fer.predict_list_images(
    path_to_input="./tests/test_images", path_to_output="./tests/output_images", save_images=True
)
df = FER.json_to_pandas("./tests/output_images/result.json")
print(df.head())

fer.analyze_video(
    path_to_video="./tests/test_videos/test_video.mp4", path_to_output="./tests/test_video", save_video=True
)
df = FER.json_to_pandas("./tests/test_video/result.json")
print(df.head())

result_dict = fer.test_fer()
print(result_dict)
