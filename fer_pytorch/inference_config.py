import os


class CFG:
    # Data path
    DATASET_PATH = "fer_pytorch/dataset"
    TEST_PATH = os.path.join(DATASET_PATH, "data/FER2013Test/")
    TEST_CSV = os.path.join(DATASET_PATH, "new_test.csv")

    # Inference parameters
    device_id = 0
    size = 224
    mean = [0.485, 0.456, 0.406]  # ImageNet values
    std = [0.229, 0.224, 0.225]  # ImageNet values
    batch_size = 32
    num_workers = 2
