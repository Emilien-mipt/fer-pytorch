from fer import FER

fer = FER()
fer.get_pretrained_model(model_name="resnet34_best")


def test_fer():
    test_result = fer.test_fer()

    assert isinstance(test_result, dict)
    assert ('accuracy' in test_result.keys()) and ('f1' in test_result.keys())
    assert (test_result['accuracy'] > 0.8) and (test_result['f1'] > 0.8)