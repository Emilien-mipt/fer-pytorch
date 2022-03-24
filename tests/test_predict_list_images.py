import os
import shutil

import numpy as np

from fer_pytorch.fer import FER

PATH_TO_FOLDER = "tests/test_images/"

fer = FER()
fer.get_pretrained_model(model_name="resnet34_best")


def test_predict_list_images():
    output_dir = "tests/test_list_images"
    result_list = fer.predict_list_images(PATH_TO_FOLDER, output_dir)

    assert isinstance(result_list, list)
    assert len(result_list) == len(os.listdir(PATH_TO_FOLDER))
    assert all(isinstance(x, dict) for x in result_list)

    path_to_json = os.path.join(output_dir, "result.json")
    result_df = FER.json_to_pandas(path_to_json)
    result_df.set_index("image_name", inplace=True)

    assert result_df.shape[0] == len(result_list)
    assert result_df.shape[1] == 3

    assert (
        result_df.loc["no_face.jpg", "box"] == []
        and result_df.loc["no_face.jpg", "emotion"] == ""
        and np.isnan(result_df.loc["no_face.jpg", "probability"])
    )

    result_df.dropna(inplace=True)

    assert (result_df["probability"] > 0.8).all()

    shutil.rmtree(output_dir)
