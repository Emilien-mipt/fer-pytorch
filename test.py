import cv2

from fer import FER

fer = FER()
fer.get_pretrained_model("resnet34_best")
frame = cv2.imread("/home/emin/Desktop/anger.jpg")
result = fer.predict_image(frame, show_top=True, path_to_output="./test.png")
print(result)

result_json_list = fer.predict_list_images(path_to_folder='./test_images/', save_images=False)
print(result_json_list)

result_json = fer.analyze_video(path_to_video="./test_video.mp4", video_name="output_video.mp4")

df = fer.json_to_pandas("./result.json")
print(df.head())