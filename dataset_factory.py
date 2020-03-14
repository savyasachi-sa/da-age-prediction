from config import *
from dataset import UTK, PairwiseUTK
from dataset_design import DatasetDesign


def get_datasets():
    # Build Dataset
    _dataset = DatasetDesign(ETHNICITIES, SOURCE_VAL_SPLIT)

    if PAIRWISE:
        train_dataset = PairwiseUTK(SOURCE_TRAIN_PATH, random_flips=True)
        val_dataset = PairwiseUTK(SOURCE_VAL_PATH, random_flips=False)
        target_dataset = PairwiseUTK(TARGET_TRAIN_PATH, random_flips=True)
    else:
        train_dataset = UTK(SOURCE_TRAIN_PATH, random_flips=True)
        val_dataset = UTK(SOURCE_VAL_PATH, random_flips=False)
        target_dataset = UTK(TARGET_TRAIN_PATH, random_flips=True)

    return train_dataset, val_dataset, target_dataset
