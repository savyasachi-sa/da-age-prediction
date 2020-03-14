from baseline_train import train as train_baseline
from adaptive_train import train as train_adaptive
from config import *
from utils import get_experiment_name

if __name__ == "__main__":

    if RANK and not PAIRWISE:
        raise Exception("Invalid Configuration!! Aborting")

    experiment_name = get_experiment_name(EXPERIMENT_NAME, PAIRWISE, RANK, ADAPTIVE, LOSS)

    if ADAPTIVE:
        train_adaptive(experiment_name)
    else:
        train_baseline(experiment_name)