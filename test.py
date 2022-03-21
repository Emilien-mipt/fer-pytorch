import cv2

from fer import FER

fer = FER()
fer.get_pretrained_model("resnet34_best")

frame = cv2.imread("/home/emin/Desktop/surprize.jpg")
result = fer.predict_image(frame)
print(result)

result = fer.predict_image(frame, show_top=True)
print(result)

frame = cv2.imread("./test_images/no_face.png")
result = fer.predict_image(frame)
print(result)

frame = cv2.imread("/home/emin/Desktop/Dollarphotoclub_71896210.jpg")
result = fer.predict_image(frame)
print(result)
result = fer.predict_image(frame, show_top=True)
print(result)

result_json_list = fer.predict_list_images(
    path_to_input="./test_images/", path_to_output="./output_images", save_images=True
)
print(result_json_list)

fer.analyze_video(path_to_video="./test_video.mp4", video_name="output_video.mp4")

df = FER.json_to_pandas("./test_video/result.json")
print(df.head())

fer.analyze_video(path_to_video="./test_video_2.mp4", video_name="output_video.mp4")

result_dict = fer.test_fer()
print(result_dict)
