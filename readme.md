# Experiments on uncertainty quantification in neural networks

This repository lists experiments performed during my Master thesis.

## Installation

Install the required libraries:

`pip install -r requirements.txt`

## Usage

### Reproducing the results

To reproduce the results on toy datasets:

`python src/run_experiment.py toy gaussian gaussian_large epistemic bimodal_dataset bimodal`

The created images, tables and saves will be stored in the folder `data_toy/`.

To reproduce the results on OpenML datasets:

`python src/run_experiment.py openml gaussian multimodal`

The created images, tables and saves will be stored in the folder `data_openml/`.
The datasets issued from https://www.openml.org/s/269 will be automatically downloaded.

### Custom experiments

You can run custom experiments using the command:

`python src/run.py {toy,openml} exp_name [options]`

The detailed signature is:

```
usage: run.py [-h] [--small_test] [--debug] [--path_root PATH_ROOT]
              [--units_size UNITS_SIZE] [--nb_hidden NB_HIDDEN]
              [--mixture_size MIXTURE_SIZE] [--datasets_list DATASETS_LIST]
              [--train_size TRAIN_SIZE] [--methods_list METHODS_LIST]
              [--alpha ALPHA] [--dropout_rate DROPOUT_RATE]
              [--ensemble_size ENSEMBLE_SIZE] [--nb_samples NB_SAMPLES]
              [--batch_size BATCH_SIZE] [--nb_repeat NB_REPEAT]
              [--nb_jobs NB_JOBS] [--max_epochs MAX_EPOCHS]
              [--patience PATIENCE]
              {toy,openml} exp_name

Experiments

positional arguments:
  {toy,openml}          Experiment type: "toy" or "openml"
  exp_name              The name of the experiment (used in the file names)

optional arguments:
  -h, --help            show this help message and exit
  --small_test          Run a test with fewer epochs and models
  --debug               Show the debug logging
  --path_root PATH_ROOT
                        The directory in which the data is stored

MLP:
  --units_size UNITS_SIZE
                        Number of units in each layer of the MLP
  --nb_hidden NB_HIDDEN
                        Number of hidden layers in the MLP
  --mixture_size MIXTURE_SIZE
                        Number of components in the mixture distributions

Dataset:
  --datasets_list DATASETS_LIST
                        The datasets to run; by default all the datasets of
                        the experiment
  --train_size TRAIN_SIZE

Methods:
  --methods_list METHODS_LIST
                        The methods to run; by default all the methods of the
                        experiment
  --alpha ALPHA         The desired coverage probability for intervals
  --dropout_rate DROPOUT_RATE
  --ensemble_size ENSEMBLE_SIZE
  --nb_samples NB_SAMPLES
                        The number of samples by methods that output more than
                        1 sample

Training:
  --batch_size BATCH_SIZE
  --nb_repeat NB_REPEAT
                        How many times the experiments have to be repeated
  --nb_jobs NB_JOBS     Number of parallel jobs
  --max_epochs MAX_EPOCHS
                        Max number of epochs until training stops
  --patience PATIENCE   Patience of the early stopping
```