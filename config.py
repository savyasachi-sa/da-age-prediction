import torch

# Change These

DATASET_ROOT_DIRECTORY = "../../da-age-prediction/data/utk/"

BASELINE_CONFIG = {
    'learning_rate': 1e-3,
    'batch_size': 16,
    'num_workers': 16,
    'num_epochs': 100,
    'cdan_hypara': 1.0
}

# Need Not Change These

SOURCE_TRAIN_PATH = DATASET_ROOT_DIRECTORY + "source_train.csv"
SOURCE_VAL_PATH = DATASET_ROOT_DIRECTORY + "source_validation.csv"
TARGET_TRAIN_PATH = DATASET_ROOT_DIRECTORY + "target_train.csv"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

WINDOW_THRESH = 3



REGRESSOR_CONF = {
    'finetune' : True,
    'feature_size': 2048
}

ADV_CONF = {
    'hidden_size' : 256
}
