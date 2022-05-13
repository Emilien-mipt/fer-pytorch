import os
import shutil

import numpy as np
import pytest

from fer_pytorch.fer import FER

PATH_TO_FOLDER = "tests/test_images/"


@pytest.mark.parametrize("fer", ["resnet34", "mobilenetv2_140"], indirect=True)
def test_predict_list_images(fer):
    output_dir = "tests/testing_list_images"
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

    assert (result_df["probability"] > 0.75).all()

    shutil.rmtree(output_dir)
