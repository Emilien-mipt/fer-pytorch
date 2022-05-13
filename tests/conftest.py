import pytest

from fer_pytorch.fer import FER


@pytest.fixture
def resnet34_fer():
    resnet34_fer = FER()
    resnet34_fer.get_pretrained_model(model_name="resnet34")
    return resnet34_fer


@pytest.fixture
def mobilenet_fer():
    mobilenet_fer = FER()
    mobilenet_fer.get_pretrained_model(model_name="mobilenetv2_140")
    return mobilenet_fer


@pytest.fixture
def fer(request):
    return request.getfixturevalue(request.param)
