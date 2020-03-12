import torch

# Change These

DATASET_ROOT_DIRECTORY = "./data/utk/"

BASELINE_CONFIG = {
    'learning_rate': 1e-3,
    'batch_size'   : 16,
    'num_workers'  : 16,
    'num_epochs'   : 1000,
    'cdan_hypara'  : 1.0
}

# Need Not Change These

SOURCE_TRAIN_PATH = DATASET_ROOT_DIRECTORY + "source_train.csv"
SOURCE_VAL_PATH = DATASET_ROOT_DIRECTORY + "source_validation.csv"
TARGET_TRAIN_PATH = DATASET_ROOT_DIRECTORY + "target_train.csv"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

WINDOW_THRESH = 3

REGRESSOR_CONF = {
    'finetune'            : True,
    'feature_sizes'       : [2048, 1024, 256, 128, 1],  # 0th - conv, then fcs, last is '1' by default.
    'adaptive_layers_conf': {
        'n_fc': [1, 2],  # 1 == first fc layer to adapt
        'conv': True  # adapt conv layer (before first fc)
    }
}

ADV_CONF = {
    'hidden_size': 128
}
