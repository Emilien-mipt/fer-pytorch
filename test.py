from fer import FER

fer = FER()
fer.get_pretrained_model("resnet34_best")
result = fer.predict_image("/home/emin/Desktop/udacha.jpg", show_top=True, save_result=True)
print(result)
