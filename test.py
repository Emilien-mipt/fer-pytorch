import cv2

from fer import FER

fer = FER()
fer.get_pretrained_model("resnet34_best")
result_dict = fer.test_fer()
print(result_dict)
