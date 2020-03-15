import torch

# Experimental Settings
PAIRWISE = False
RANK = False
ADAPTIVE = True
EXPERIMENT_NAME = 'Test'
LOSS = "L1"

ROOT_CONFIG = {
    'learning_rate': 1e-3,
    'batch_size': 16,
    'num_workers': 16,
    'num_epochs': 10,
    'cdan_hypara': 1.0
}

# Model Settings

REGRESSOR_CONF = {
    'fine_tune': True,
    'feature_sizes': [2048, 1024, 256, 128, 1],  # 0th - conv, then fcs, last is '1' by default.
    'adaptive_layers_conf': {
        'n_fc': [1, 2],  # 1 == first fc layer to adapt
        'conv': True  # adapt conv layer (before first fc)
    }
}

ADV_CONF = {
    'hidden_size': 128
}

# Stats Manager Settings

WINDOW_THRESH = 7

# Dataset Settings

ETHNICITIES = {
    "source": 0,
    "target": 1,
}

DATASET_ROOT_DIRECTORY = "/Users/savyasachi/utk/"

SOURCE_TRAIN_PATH = DATASET_ROOT_DIRECTORY + "source_train.csv"
SOURCE_VAL_PATH = DATASET_ROOT_DIRECTORY + "source_validation.csv"
TARGET_TRAIN_PATH = DATASET_ROOT_DIRECTORY + "target_train.csv"

SOURCE_VAL_SPLIT = 80

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
