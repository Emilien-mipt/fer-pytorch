import cv2
import numpy as np

from fer_pytorch.fer import FER

PATH_HAPPY = "tests/test_images/happy.jpg"
PATH_SURPRIZE = "tests/test_images/surprize.jpg"

PATH_NOFACE = "tests/test_images/no_face.jpg"

fer = FER()
fer.get_pretrained_model(model_name="mobilenetv2_140")


def test_with_face_types():
    input = cv2.imread(PATH_HAPPY)

    # Test full output
    result = fer.predict_image(input)
    assert isinstance(result, list)
    assert len(result) == 1

    result_dict = result[0]
    assert isinstance(result_dict, dict)

    assert ("box" in result_dict.keys()) and ("emotions" in result_dict.keys())
    assert isinstance(result_dict["box"], list)
    assert isinstance(result_dict["emotions"], dict)

    # Test top output
    result_top = fer.predict_image(input, show_top=True)
    assert isinstance(result_top, list)
    assert len(result_top) == 1

    result_dict_top = result_top[0]
    assert isinstance(result_dict_top, dict)

    assert ("box" in result_dict_top.keys()) and ("top_emotion" in result_dict_top.keys())
    assert isinstance(result_dict_top["box"], list)
    assert isinstance(result_dict_top["top_emotion"], dict)


def test_no_face():
    no_face = cv2.imread(PATH_NOFACE)
    result_no_face = fer.predict_image(no_face)

    assert isinstance(result_no_face, list)
    assert len(result_no_face) == 0


def test_happy_values():
    input = cv2.imread(PATH_HAPPY)

    result_dict = fer.predict_image(input)[0]

    np.testing.assert_almost_equal(result_dict["box"], [295.90848, 87.36073, 463.75354, 296.00055], decimal=1)

    emotion_dict = result_dict["emotions"]

    assert (
        ("neutral" in emotion_dict.keys())
        and ("happiness" in emotion_dict.keys())
        and ("surprise" in emotion_dict.keys())
        and ("sadness" in emotion_dict.keys())
        and ("anger" in emotion_dict.keys())
        and ("disgust" in emotion_dict.keys())
        and ("fear" in emotion_dict.keys())
    )
    assert emotion_dict["happiness"] > 0.8

    sum_probs = 0
    for value in emotion_dict.values():
        sum_probs += value

    np.testing.assert_almost_equal(1.0, sum_probs, decimal=3)


def test_surprize():
    input = cv2.imread(PATH_SURPRIZE)

    result_dict = fer.predict_image(input)[0]

    np.testing.assert_almost_equal(result_dict["box"], [260.15295, 213.43015, 472.2445, 509.92935], decimal=1)

    emotion_dict = result_dict["emotions"]

    assert (
        ("neutral" in emotion_dict.keys())
        and ("happiness" in emotion_dict.keys())
        and ("surprise" in emotion_dict.keys())
        and ("sadness" in emotion_dict.keys())
        and ("anger" in emotion_dict.keys())
        and ("disgust" in emotion_dict.keys())
        and ("fear" in emotion_dict.keys())
    )
    assert emotion_dict["surprise"] > 0.8

    sum_probs = 0
    for value in emotion_dict.values():
        sum_probs += value

    np.testing.assert_almost_equal(1.0, sum_probs, decimal=3)
