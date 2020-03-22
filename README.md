# da-age-prediction
Domain Adaptation for Age prediction

## Installation

The following Libraries are required to run the dataset - `torch, matplotlib`

We also need to add utk dataset to /da-age-prediction/data/  [Already present in the folder provided]

## Code Structure

All the code is highly modular with self descriptive names. All the configuration is controlled in `config.py`

* `main.py` - Driver Class for training
* `config.py` - Holds all the configuration for training the model.
                PAIRWISE : Set this flag to True to load the images in a pairwise manner and to predict the age difference    between them.
                RANK : Set this flag to True(only in case PAIRWISE == True) to predict both the relative age difference and the order of the identities in the image in terms of age.
* `dataset` - dataset is the Pytorch's dataset class for UTK dataset. 
* `dataset_design` - builds the dataset csvs from the utk dataset based on the training validation splits
* `dataset_factory` - factory to fetch the right dataset based on the configuration
* `final_resnet` - Main Model with various configurations for pairiwse ranking etc. 
* `adversarial_network` - Adversarial Network
* `cdan` - Helper function to evaluate adversarial Loss.
* `mmd` - Helper function to evaluate MMD Loss.
* `mds` - Helper function for MDS evaluation.
* `baseline_train` & `adaptive_train` - Utility classes to trigger baseline and adaptive training respectively
* `adaptive_experiment` - Training code to train the network in adaptive mode.
* `nntools` - Training code to train the network without Adaptation.

## Instructions to Run

## Get The dataset Ready

* The folder contains the data in a zipped format. Please unzip the dataset using command `unzip data.zip`

### Training

* Configure your training parameters in `config.py`. 
* There are flags and configurations for every setting like (Adaptive, Pairwise, Ranking, Smoothning etc.)
* Also set a new Experiment Name in `config.py` if you want to start a new experiment. [Need not do for first run]
* After configuring the setting in config.py - just run the following command - `python main.py` and it will start the training for the set configuration

### Model Evaluation

* In order to evaluate a saved model present in `/models` directory, with name `my_model_name`, Just Run `python experiment_statistics.py my_model_name`.
* For Instance, if you wish to evaluate the model `baseline` - run `python experiment_statistics.py baseline`
* It will generate plots and loss values and store it `./results` directory with the same `model_name`
