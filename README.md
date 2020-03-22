# da-age-prediction
Domain Adaptation for Age prediction

## Installation

The following Libraries are required to run the dataset - `torch, matplotlib`

We also need to add utk dataset to /da-age-prediction/data/  [Already present in the zip provided]

## Code Structure

All the code is highly modular with self descriptive names. All the configuration is controlled in `config.py`

## How to Run

### Training

* Configure your training parameters in `config.py`. 
* There are flags and configurations for every setting like (Adaptive, Pairwise, Ranking, Smoothning etc.)
* Also set a new Experiment Name in `config.py` if you want to start a new experiment. [Need not do for first run]
* After configuring the setting in config.py - just run the following command - `python main.py` and it will start the training for the set configuration

### Model Evaluation

* In order to evaluate a saved model present in `/models` directory, with name `my_model_name`, Just Run `python experiment_statistics.py my_model_name`.
* For Instance, if you wish to evaluate the model `baseline` - run `python experiment_statistics.py baseline`
* It will generate plots and loss values and store it `./results` directory with the same `model_name`