import os

import pytest

os.system("sh get_data.sh")


@pytest.mark.parametrize("fer", ["resnet34", "mobilenetv2_140"], indirect=True)
def test_fer(fer):
    test_result = fer.test_fer()

    assert isinstance(test_result, dict)
    assert ("accuracy" in test_result.keys()) and ("f1" in test_result.keys())
    assert (test_result["accuracy"] > 0.8) and (test_result["f1"] > 0.8)
