import cv2

from fer import FER

fer = FER()
fer.get_pretrained_model("resnet34_best")
fer.run_webcam()
