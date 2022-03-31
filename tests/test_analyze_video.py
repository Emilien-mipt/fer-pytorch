import os
import shutil
import urllib.request

import cv2

from fer_pytorch.fer import FER

URL = "https://github.com/Emilien-mipt/FERplus-Pytorch/releases/download/0.0.2/test_video.mp4"
PATH_TO_VIDEO = "tests/test_video.mp4"

fer = FER()
fer.get_pretrained_model(model_arch="resnet34", model_name="resnet34_best")


def test_analyze_video():
    urllib.request.urlretrieve(URL, "tests/test_video.mp4")

    v_cap = cv2.VideoCapture(PATH_TO_VIDEO)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    path_to_output = "tests/testing_video"
    path_to_output.split()
    fer.analyze_video(path_to_video=PATH_TO_VIDEO, path_to_output=path_to_output)

    path_to_json = os.path.join(path_to_output, "result.json")

    result_df = FER.json_to_pandas(path_to_json)
    result_df.set_index("frame_id", inplace=True)

    assert result_df.shape[0] == v_len
    assert result_df.shape[1] == 3

    assert not result_df.isnull().values.any()

    assert (result_df["probability"][:9] > 0.9).all()
    assert (result_df["emotion"][:10] == "happiness").all()

    assert (result_df["probability"][-20:] > 0.9).all()
    assert (result_df["emotion"][-20:] == "anger").all()

    shutil.rmtree(path_to_output)
