"""
Available experiments:
- gaussian: gaussian predictions
- gaussian_small_mlp: gaussian predictions with less layers and units in the MLP
- multimodal: multimodal predictions
"""

import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import numpy as np
import seaborn as sns
from pathlib import Path
import logging
import pickle

from training import train_models
from models.gaussian_models import get_gaussian_model_builders, gaussian_predictors, interval_predictors
from models.calibration_models import PostHocCalibration
from datasets.datasets_openml import get_datasets as get_openml_datasets, make_dataset, check_dataset
from datasets.stats_storage import StatsStorage
from metrics import (gaussian_metrics_names, gaussian_metrics_values,
	multimodal_gaussian_metrics_names, multimodal_gaussian_metrics_values
)
from utils import filter_dict, reset_seeds
from tables import make_tables_gaussian_dataset
from graphics import CalibrationGraphics


sns.set()
logger = logging.getLogger('thesis')
logging.getLogger('matplotlib').setLevel(logging.INFO)
logging.getLogger('openml').setLevel(logging.WARNING)


def train_models_adapted(datasets, model_name, model_builder, args_mlp, nb_jobs, nb_repeat, **kwargs):
	if model_name == 'Bayes by backprop':
		args_mlp = args_mlp.copy()
		args_mlp['nb_hidden'] = 1
		kwargs = kwargs.copy()
		kwargs['patience'] = 30
	return train_models(datasets, model_name, model_builder, args_mlp, nb_jobs, nb_repeat, **kwargs)


def run_dataset(exp_name, stats, path_saves, path_images, openml_ds, model_builders, nb_repeat, 
		args_mlp, alpha, nb_jobs=1, **kwargs):
	path_saves /= openml_ds.name
	path_saves.mkdir(parents=True, exist_ok=True)
	path_images /= openml_ds.name
	path_images.mkdir(parents=True, exist_ok=True)

	if not check_dataset(openml_ds):
		logger.info('No valid column for this dataset')
		return

	calibration_graphics = CalibrationGraphics(len(filter_dict(model_builders, gaussian_predictors)))

	for model_name, model_builder in model_builders.items():
		logger.info(f'Computing model {model_name}...')
		datasets = [make_dataset(openml_ds) for _ in range(nb_repeat)]
		trained_models = train_models_adapted(datasets, model_name, model_builder, args_mlp, 
				nb_jobs, nb_repeat, **kwargs)
	
		for run, (ds, trained_model) in enumerate(zip(datasets, trained_models)):
			last_epoch, pred_test, pred_val, train_time, prediction_time = trained_model
			# Compute the metrics
			cal_model = PostHocCalibration()
			cal_model.fit(pred_val, ds.valid.y)
			if exp_name == 'multimodal':
				metrics = multimodal_gaussian_metrics_values(pred_test, ds, alpha, cal_model)
			else:
				metrics = gaussian_metrics_values(pred_test, ds, alpha, cal_model)

			# Store the metrics
			stats.add(ds.name, model_name, run, metrics, last_epoch, train_time, prediction_time)
			with open(path_saves / f'{exp_name}_stats.pickle', 'wb') as f: # Resave the whole stats at the end of each dataset
				pickle.dump(stats, f)
			# Plot
			if run == 0:
				if model_name in gaussian_predictors:
					calibration_graphics.plot(pred_test, ds, cal_model, model_name)
	
	calibration_graphics.save(path_images, exp_name)

def get_model_builders(exp_name, small_test, single_model, alpha, dropout_rate, ensemble_size, nb_samples):
	model_builders = get_gaussian_model_builders(alpha=alpha, dropout_rate=dropout_rate,
			ensemble_size=ensemble_size, nb_samples=nb_samples)
	to_pop = ['Gaussian process', 'GPflow', 'SVI'] # Don't use these models
	for mb in to_pop:
		model_builders.pop(mb)
	
	filtered = model_builders.keys()
	if exp_name == 'multimodal':
		filtered = ['MLE', 'MLE ensemble', 'MLE MC dropout', 'Bayes by backprop']
	model_builders = filter_dict(model_builders, filtered)

	if single_model:
		model_builders = filter_dict(model_builders, [single_model])

	if small_test:
		model_builders = filter_dict(model_builders, ['MLE', 'Quantile-based PI', 'HQPI'])
	logger.debug('Models: ' + ', '.join(model_builders.keys()))
	return model_builders

def get_datasets(exp_name):
	datasets = get_openml_datasets()
	datasets = filter_dict(datasets, [
		#'boston', 'Moneyball', 'house_sales', 'us_crime', 'MIP-2016-regression', 'abalone', 'diamonds', 'space_ga'
		'boston', 'us_crime', 'house_sales',
	])
	return datasets

def get_metrics_name(exp_name):
	if exp_name == 'multimodal':
		metrics_name = multimodal_gaussian_metrics_names
	else:
		metrics_name = gaussian_metrics_names
	return metrics_name

exps_list = ['gaussian', 'gaussian_small_mlp', 'multimodal']

def run(exp_name, mixture_size=1, small_test=False, debug=False, single_model=None):
	assert exp_name in exps_list
	if exp_name != 'multimodal':
		assert mixture_size == 1
	use_cpu = True
	if use_cpu:
		tf.config.set_visible_devices([], 'GPU')
	reset_seeds(0)

	logger.setLevel(logging.DEBUG if debug else logging.INFO)

	logger.info('=' * 20)
	logger.info(f'Experiment {exp_name}')
	logger.info('=' * 20)

	root = Path('..')
	root_data = root / 'data_openml'
	path_images = root_data / 'images'
	path_tables = root_data / 'tables'
	path_saves = root_data / 'saves'

	args_mlp = {
		'units_size': 100,
		'nb_hidden': 3,
		'mixture_size': mixture_size,
	}
	if exp_name == 'gaussian_small_mlp':
		args_mlp['units_size'] = 100
		args_mlp['nb_hidden'] = 1

	batch_size = 64
	alpha = 0.05 # The desired coverage probability for the evaluation of intervals
	dropout_rate = 0.1
	ensemble_size = 5
	nb_samples = 200 # The number of samples from methods that output samples
	nb_repeat = 20 # How many times the experiments have to be repeated
	nb_jobs = 1 # Number of parallel jobs
	max_epochs = 400
	patience = 10

	if small_test:
		max_epochs = 5
		nb_repeat = 2

	metrics_name = get_metrics_name(exp_name)
	openml_datasets = get_datasets(exp_name)
	model_builders = get_model_builders(exp_name, small_test, single_model, alpha, dropout_rate, ensemble_size, nb_samples)

	for openml_ds in openml_datasets.values():
		logger.info(f'Evaluating dataset {openml_ds.name}...')
		stats_gaussian = StatsStorage(columns=(
			'Dataset', 'Model', 'Run', *metrics_name, 'Epochs', 'Training time (s)', 'Prediction time (s)'
		))
		run_dataset(exp_name, stats_gaussian, path_saves, path_images, openml_ds, model_builders, nb_repeat=nb_repeat, args_mlp=args_mlp,
				alpha=alpha, nb_jobs=nb_jobs, max_epochs=max_epochs, patience=patience, batch_size=batch_size)
		df = stats_gaussian.build_df()
		make_tables_gaussian_dataset(exp_name, path_tables, df, openml_ds.name, alpha, path_images=path_images)

	return df

def run_all(exps_list, **kwargs):
	for exp_name in exps_list:
		run(exp_name, **kwargs)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='OpenML experiments')
	parser.add_argument('exps_list', nargs='*')
	parser.add_argument('--mixture_size', type=int, default=1)
	parser.add_argument('--small_test', action='store_true')
	parser.add_argument('--debug', action='store_true')
	parser.add_argument('--single_model', nargs='?')
	kwargs = vars(parser.parse_args())
	run_all(**kwargs)