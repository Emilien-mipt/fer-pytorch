import cv2
import numpy as np
import pytest

PATH_HAPPY = "tests/test_images/happy.jpg"
PATH_SURPRIZE = "tests/test_images/surprize.jpg"
PATH_NOFACE = "tests/test_images/no_face.jpg"
PATH_MULTI = "tests/test_images/multi.jpg"


@pytest.mark.parametrize("fer", ["resnet34", "mobilenetv2_140"], indirect=True)
@pytest.mark.parametrize("show_top", ["True", "False"])
def test_with_face_types(fer, show_top):
    input = cv2.imread(PATH_HAPPY)

    # Test full output
    result = fer.predict_image(input, show_top=show_top)
    assert isinstance(result, list)
    assert len(result) == 1

    result_dict = result[0]
    assert isinstance(result_dict, dict)

    if show_top is False:
        assert ("box" in result_dict.keys()) and ("emotions" in result_dict.keys())
        assert isinstance(result_dict["emotions"], dict)
    else:
        assert ("box" in result_dict.keys()) and ("top_emotion" in result_dict.keys())
        assert isinstance(result_dict["top_emotion"], dict)
    assert isinstance(result_dict["box"], list)


@pytest.mark.parametrize("fer", ["resnet34", "mobilenetv2_140"], indirect=True)
def test_no_face(fer):
    no_face = cv2.imread(PATH_NOFACE)
    result_no_face = fer.predict_image(no_face)

    assert isinstance(result_no_face, list)
    assert len(result_no_face) == 0


@pytest.mark.parametrize("fer", ["resnet34", "mobilenetv2_140"], indirect=True)
def test_happy_values(fer):
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


@pytest.mark.parametrize("fer", ["resnet34", "mobilenetv2_140"], indirect=True)
def test_surprize(fer):
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


@pytest.mark.parametrize("fer", ["resnet34", "mobilenetv2_140"], indirect=True)
def test_multi(fer):
    multi_img = cv2.imread(PATH_MULTI)
    result_multi = fer.predict_image(multi_img, show_top=True)

    assert len(result_multi) == 2
    assert isinstance(result_multi, list)
    assert all(isinstance(result, dict) for result in result_multi)

    neutral_dict = result_multi[0]
    happy_dict = result_multi[1]

    np.testing.assert_almost_equal(neutral_dict["box"], [75.1005, 24.66243, 115.62831, 79.85703], decimal=1)
    np.testing.assert_almost_equal(happy_dict["box"], [165.5209, 48.91686, 203.69823, 97.520325], decimal=1)
    assert next(iter(neutral_dict["top_emotion"].values())) > 0.9
    assert next(iter(happy_dict["top_emotion"].values())) > 0.9
