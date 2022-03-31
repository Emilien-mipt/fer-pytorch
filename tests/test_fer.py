import os

from fer_pytorch.config import CFG
from fer_pytorch.fer import FER

fer = FER()
fer.get_pretrained_model(model_arch="resnet34", model_name="resnet34_best")


def test_fer():
    path_to_data = os.path.join(CFG.DATASET_PATH, "data")
    if not os.path.isdir(path_to_data) or len(os.listdir(path_to_data)) == 0:
        os.system("sh get_data.sh")

    test_result = fer.test_fer()

    assert isinstance(test_result, dict)
    assert ("accuracy" in test_result.keys()) and ("f1" in test_result.keys())
    assert (test_result["accuracy"] > 0.8) and (test_result["f1"] > 0.8)
