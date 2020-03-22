import torch

# ************ Training Parameters Settings ************** #

PAIRWISE = False # Use Pairwise data or not
RANK = False # Ranking Prediction and Loss enabled or not
ADAPTIVE = True # Adaptation is enabled or not
LOSS = "L1" # 'L1' or 'L2
ADVERSARIAL_FLAG = True #ADVERSARIAL ADAPTATION BASED LOSS:
LABEL_NORM = False  # LABEL NORMALIZATION ON DATASET
SIGMA2 = 1e5  # SMOOTHING LOSS FOR REGRESSION DA
MMD_FLAG = True # MMD Flag for MMD Based Loss
SMOOTH_FLAG = True # Smoothing flag for Graph Laplacian Smoothing based Loss
IDENTITY_FLAG = True # IDENTITY CONSTRAINT BASED LOSS

EXPERIMENT_NAME = 'BaselineExperiment'

# ************ Training Experiment Settings ************** #

REGRESSOR_CONF = {
    'fine_tune': True, # Fine Tune Resnet or Not
    'feature_sizes': [2048, 1024, 256, 128, 1],  # 0th - conv, then fcs, last is '1' by default.
    'adaptive_layers_conf': {
        'n_fc': [1],  # 1 == first fc layer to adapt
        'conv': False  # adapt conv layer (before first fc)
    }
}

# Dataset Settings
ETHNICITIES = {
    "source": 0,  # Baseline  or Source-Adaptive
    "target": 1,
}
ROOT_CONFIG = {
    'learning_rate': 1e-3,
    'batch_size': 4,
    'num_workers': 16,
    'num_epochs': 2500,
    'cdan_hypara': 1.0,
    'mmd_hypara': 1.0,
    'smooth_hypara': 1.0,
    'id_hypara': 1.0,
}

ADV_CONF = {
    'hidden_size': 128
}

DATASET_ROOT_DIRECTORY = "./data/utk/"

SOURCE_VAL_SPLIT = 80

SOURCE_TRAIN_PATH = DATASET_ROOT_DIRECTORY + "source_train.csv"
SOURCE_VAL_PATH = DATASET_ROOT_DIRECTORY + "source_validation.csv"
TARGET_TRAIN_PATH = DATASET_ROOT_DIRECTORY + "target_train.csv"

# Stats Manager Settings
WINDOW_THRESH = 7

# Stats Evaluation Settings
STATS_OUTPUT_DIR = './results/'
STATS_MODEL_NAME = 'baseline_L1'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

