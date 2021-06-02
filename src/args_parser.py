import argparse
from pathlib import Path

def add_mlp_args(parser):
	parser.add_argument('--units_size', type=int, default=100, help='Number of units in each layer of the MLP')
	parser.add_argument('--nb_hidden', type=int, default=3, help='Number of hidden layers in the MLP')
	parser.add_argument('--mixture_size', type=int, default=1, help='Number of components in the mixture distributions')

def add_dataset_args(parser):
	parser.add_argument('--datasets_list', default=None,
			help='The datasets to run; by default all the datasets of the experiment')
	parser.add_argument('--train_size', type=int, default=None)

def add_method_args(parser):
	parser.add_argument('--methods_list', default=None,
			help='The methods to run; by default all the methods of the experiment')
	parser.add_argument('--alpha', type=int, default=0.05, help='The desired coverage probability for intervals')
	parser.add_argument('--dropout_rate', type=int, default=0.1)
	parser.add_argument('--ensemble_size', type=int, default=5)
	parser.add_argument('--nb_samples', type=int, default=200, help='The number of samples by methods that output more than 1 sample')

def add_training_args(parser):
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--nb_repeat', type=int, default=1, help='How many times the experiments have to be repeated')
	parser.add_argument('--nb_jobs', type=int, default=1, help='Number of parallel jobs')
	parser.add_argument('--max_epochs', type=int, default=1000, help='Max number of epochs until training stops')
	parser.add_argument('--patience', type=int, default=10, help='Patience of the early stopping')

def add_general_args(parser):
	parser.add_argument('exp_type', type=str.lower, choices=['toy', 'openml'], help='Experiment type: "toy" or "openml"')
	parser.add_argument('--small_test', action='store_true', help='Run a test with fewer epochs and models')
	parser.add_argument('--debug', action='store_true', help='Show the debug logging')
	parser.add_argument('--path_root', type=Path, default=Path('.'), help='The directory in which the data is stored')

def make_parser(general_args=True):
	parser = argparse.ArgumentParser(description='Experiments')
	if general_args:
		add_general_args(parser)
	add_mlp_args(parser.add_argument_group('MLP'))
	add_dataset_args(parser.add_argument_group('Dataset'))
	add_method_args(parser.add_argument_group('Methods'))
	add_training_args(parser.add_argument_group('Training'))
	return parser