# Experiments on uncertainty quantification in neural networks

This repository lists experiments performed during my Master thesis.

## Installation

Install the necessary libraries:

`pip install -r requirements.txt`

## Toy experiments

To run all the experiments on toy datasets:

`python run_toy.py [--debug] [--single_model=<M>]`

Use `--debug` to see the debug output and `--single_model=<M>` to run only one model with name `M`.
Refer to the file `run_toy.py` for changing the hyperparameters.

The created images, tables and saves will be stored in the folder `data_toy/`.
You can also run the notebook in `notebooks/run_toy.ipynb` for convenience.

## OpenML experiments

To run all the experiments on OpenML datasets:

`python run_openml.py [--debug] [--single_model=<M>]`

Simlarly, use `--debug` to see the debug output and `--single_model=<M>` to run only one model with name `M`.
The datasets issued from https://www.openml.org/s/269 will be automatically downloaded.
The created images, tables and saves will be stored in the folder `data_openml/`.
You can also run the notebook in `notebooks/run_openml.ipynb` for convenience.
