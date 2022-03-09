from fer import FER

fer = FER()
fer.get_pretrained_model("resnet34_best")
result = fer.predict_image("/home/emin/Desktop/168.png", show_top=True, save_result=True)
print(result)
result_dict = fer.predict_list_images(path_to_folder='./test_images', batch_size=2)
