FER-pytorch
===========

Facial expression recognition package built on Pytorch and [FER+ dataset from Microsoft](https://github.com/microsoft/FERPlus).

[![PyPI version](https://badge.fury.io/py/fer-pytorch.svg)](https://badge.fury.io/py/fer-pytorch)
![CI](https://github.com/albumentations-team/albumentations/workflows/CI/badge.svg)
[![Downloads](https://pepy.tech/badge/fer-pytorch)](https://pepy.tech/project/fer-pytorch)

![dicaprio_result](https://user-images.githubusercontent.com/44554040/163209336-198e3db3-a84f-4d81-9156-19177bba7808.png)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_sTDVvK-673CKyYQP7gsViCkO8eBa9jd?usp=sharing)

# Installation
`pip install fer-pytorch`

# Training
Training is done using the synergy of [Pytorch Lightning](https://www.pytorchlightning.ai/) and
[Hydra](https://hydra.cc/docs/intro/) packages for setting training loops and configs correspondingly.
In order to run training you should clone the repo and

### Install dependencies
```
pip install -r requirements/requirements-dev.txt
```

### Define environmental variables
Just run `export PYTHONPATH="$PWD"` from the root directory and it would be enough
to run the code from the root. You can check the PYTHONPATH with `echo $PYTHONPATH` command.

### Download data
`bash get_data.sh`

This bash command will download the archive with the data and unpack it into the necessary directories in such a way
that everything is ready for training.

### Define config
The config files are located in `fer-pytorch/conf` directory. To see all the parameters
and their default values run

`python fer_pytorch/run_trainer.py -h`

### Run training
Training with default parameter values:

`python fer_pytorch/run_trainer.py`

Thanks to Hydra all the parameters set in config files can be changed
directly from the command line while running the script:

* Example with change of model from resnet34 to resnet18:
  * `python fer_pytorch/run_trainer.py model.model_name="resnet18"`

* Example with change of number of epochs:
  * `python fer_pytorch/run_trainer.py trainer.trainer_params.max_epochs=100`

By default the output is saved to `output/` directory. If you wish to set
the path to output, run

`python fer_pytorch/run_trainer.py hydra.run.dir=path_to_output`


# Inference

### Import inference class
```
import cv2
from fer_pytorch.fer import FER

fer = FER()
```

### Initialize the model
There are 2 options:

1. `fer.get_pretrained_model(model_name)`: download the ready-to-use pretrained on FER+ dataset weights from the github
page of the package and initialize the model automatically. The list of available names are given in
`fer_pytorch/pre_trained_models.py` file as the keys of `models` dictionary. For example,
`fer.get_pretrained_model("resnet34")`.
2. `fer.load_user_weights(model_name, path_to_weights)`:  with this option you can load your own weights
that are stored locally.

### Inference on an image
* Basic prediction
```
img = cv2.imread("tests/test_images/happy.jpg")
result = fer.predict_image(img)
```

Sample output:
```
[{'box': [295.90848, 87.36073, 463.75354, 296.00055],
'emotions': {'neutral': 0.00033704843, 'happiness': 0.98931086, 'surprise': 0.00018355528, 'sadness': 0.0026534477, 'anger': 0.0054451805, 'disgust': 0.0019571118, 'fear': 0.000112833266}}]
```

* Get only top emotion:

`result = fer.predict_image(frame, show_top=True)`

Sample output:

`[{'box': [295.90848, 87.36073, 463.75354, 296.00055], 'top_emotion': {'happiness': 0.98931086}}]`

* In order to save the output image, just set the output path:

`result = fer.predict_image(frame, show_top=True, path_to_output="result.jpg")`

### Inference on a list of images
Inference on a folder with images:
```
result_json_list = fer.predict_list_images(
    path_to_input="tests/test_images",
    path_to_output="tests/output_images",
    save_images=True
)
```

Outputs the json with results and optionally the processed images in a separate directory.

It is also possible to read the result json with pandas leveraging FER class method:

```
result_df = FER.json_to_pandas("tests/output_images/result.json")
print(result_df.head())
```

### Inference on a video file
```
fer.analyze_video(
    path_to_video="tests/test_videos/test_video.mp4",
    path_to_output="tests/test_video",
    save_video=True
)
```

Outputs the json with results and optionally the processed video in a separate directory.

Just like in the previous case it is also possible to read the result json with
pandas for further analysis leveraging FER class method:
```
df = FER.json_to_pandas("tests/test_video/result.json")
print(df.head())
```

### Inference on the test part of the FER dataset
Get the dictionary with accuracy and f1 score for the test part of the FER+ dataset:
```
result_dict = fer.test_fer(
    path_to_dataset = "fer_pytorch/dataset",
    path_to_csv = "fer_pytorch/dataset/new_test.csv",
    batch_size = 32,
    num_workers = 8,
)
print(result_dict)
```

Output:

`{'accuracy': 0.83, 'f1': 0.83}`

### Inference with the web camera
To run the model on the stream from the web camera and show the results in real-time just run

`fer.run_webcam()`

# Web application
In order to demonstrate the fer-pytorch package the web application has been developed
using [Streamlit](https://streamlit.io/) and deployed to [Heroku](heroku.com) via Docker image.

Link to the github page of the app: https://github.com/Emilien-mipt/fer-webapp

Link to the app: https://ferpytorch-webapp.herokuapp.com/
