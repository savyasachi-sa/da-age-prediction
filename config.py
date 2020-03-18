import torch

# ************ Experiment Statitstics Settings ***********
# ******************************************************** # ********************************************************
PAIRWISE = False
RANK = False
ADAPTIVE = True
EXPERIMENT_NAME = 'Test'
LOSS = "L1"

# Model Settings
REGRESSOR_CONF = {
    'fine_tune'           : True,
    'feature_sizes'       : [2048, 1024, 256, 128, 1],  # 0th - conv, then fcs, last is '1' by default.
    'adaptive_layers_conf': {
        'n_fc': [1,2,3],  # 1 == first fc layer to adapt
        'conv': False  # adapt conv layer (before first fc)
    }
}

DATASET_ROOT_DIRECTORY = "./data/utk/"

SOURCE_TRAIN_PATH = DATASET_ROOT_DIRECTORY + "source_train.csv"
SOURCE_VAL_PATH = DATASET_ROOT_DIRECTORY + "source_validation.csv"
TARGET_TRAIN_PATH = DATASET_ROOT_DIRECTORY + "target_train.csv"

SOURCE_VAL_SPLIT = 80

# Stats Manager Settings
WINDOW_THRESH = 7

# Dataset Settings
ETHNICITIES = {
    "source": 0,  # Baseline  or Source-Adaptive
    "target": 1,
}

## Stats Evaluation Settings
STATS_OUTPUT_DIR =  './results/'
STATS_MODEL_NAME = 'baseline_L1'

# ************ Experiment Statitstics Settings End ***********
# ********************************************************# ********************************************************


ROOT_CONFIG = {
    'learning_rate': 1e-3,
    'batch_size'   : 2,
    'num_workers'  : 16,
    'num_epochs'   : 100,
    'cdan_hypara'  : 1.0
}

ADV_CONF = {
    'hidden_size': 128
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
