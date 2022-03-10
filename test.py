from fer import FER

fer = FER()
fer.get_pretrained_model("resnet34_best")
result = fer.predict_image("/home/emin/Desktop/anger.jpg", show_top=True, save_image=False)
print(result)
result_list = fer.predict_list_images(path_to_folder='./test_images/', save_images=True)
print(result_list)
