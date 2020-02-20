import torch

# Change These

DATASET_ROOT_DIRECTORY = "/Users/savyasachi/utk/"

BASELINE_CONFIG = {
    'learning_rate': 1e-3,
    'batch_size': 16,
    'num_workers': 4,
    'num_epochs': 100
}

# Need Not Change These

SOURCE_TRAIN_PATH = DATASET_ROOT_DIRECTORY + "source_train.csv"
SOURCE_VAL_PATH = DATASET_ROOT_DIRECTORY + "source_validation.csv"
TARGET_TRAIN_PATH = DATASET_ROOT_DIRECTORY + "target_train.csv"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
