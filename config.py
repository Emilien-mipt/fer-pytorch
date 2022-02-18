import os

import pandas as pd


class CFG:
    # Data path
    TRAIN_PATH = "./FERplus_dataset/data/FER2013Train/"
    TRAIN_CSV = "./FERplus_dataset/new_train.csv"
    VAL_PATH = "./FERplus_dataset/data/FER2013Valid/"
    VAL_CSV = "./FERplus_dataset/new_val.csv"
    TEST_PATH = "./FERplus_dataset/data/FER2013Test/"
    TEST_CSV = "./FERplus_dataset/new_test.csv"
    # Logging
    LOG_DIR = "./logs"
    OUTPUT_DIR = "test_new"

    # Model setup
    # chk = os.path.join(LOG_DIR, OUTPUT_DIR, "weights", "best.pt")
    chk = "./logs/test/weights/best.pt"
    # chk = ""
    model_name = "resnet34"
    pretrained = True

    # Main config
    GPU_ID = 0
    seed = 42
    target_size = 7
    target_col = "label"

    # Train configs
    MIXED_PREC = False  # Flag for mixed precision training
    debug = False
    epochs = 50
    early_stopping = 10
    batch_size = 32
    size = 224
    MEAN = [0.485, 0.456, 0.406]  # ImageNet values
    STD = [0.229, 0.224, 0.225]  # ImageNet values
    num_workers = 8
    print_freq = 100

    # Optimizer config
    lr = 2e-2
    momentum = 0.9
    min_lr = 1e-2
    weight_decay = 1e-5
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
