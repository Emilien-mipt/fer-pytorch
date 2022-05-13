import pytest

from fer_pytorch.fer import FER


@pytest.fixture
def resnet34():
    resnet34_fer = FER()
    resnet34_fer.get_pretrained_model(model_name="resnet34")
    return resnet34_fer


@pytest.fixture
def mobilenetv2_140():
    mobilenet_fer = FER()
    mobilenet_fer.get_pretrained_model(model_name="mobilenetv2_140")
    return mobilenet_fer


@pytest.fixture
def fer(request):
    return request.getfixturevalue(request.param)
