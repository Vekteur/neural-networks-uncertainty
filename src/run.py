"""
Available experiments:
- gaussian: gaussian dataset and gaussian predictions
- gaussian_large: larger dataset
- epistemic: gaussian dataset and gaussian predictions and epistemic uncertainty
- bimodal_dataset: bimodal dataset and gaussian predictions
- bimodal: bimodal dataset and bimodal predictions
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from pathlib import Path
import logging
import pickle

from training import train_models_adapted
from models.gaussian_models import get_gaussian_model_builders, gaussian_predictors, interval_predictors
from models.post_hoc_calibration import PostHocCalibration
from datasets.datasets_toy import get_toy_dataset_builders
from datasets.datasets_openml import get_datasets as get_openml_datasets, make_dataset, check_dataset
from datasets.stats_storage import StatsStorage
from metrics import (gaussian_metrics_names, gaussian_metrics_values,
	multimodal_gaussian_metrics_names, multimodal_gaussian_metrics_values
)
from utils import filter_dict, reset_seeds
from tables import make_tables_gaussian_dataset
from graphics import ToyGraphics, CalibrationGraphics
from args_parser import make_parser

logger = logging.getLogger('thesis')
logging.getLogger('matplotlib').setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)


def plot_toy(toy_graphics, ds, pred_test, alpha, model_name, mixture_size):
	alpha_scatter = 0.2
	if ds.train.size > 1000:
		alpha_scatter = 0.08
	plot_type = 'single_gaussian' if mixture_size == 1 else 'density'
	toy_graphics.plot(ds, pred_test, alpha, model_name, alpha_scatter, plot_type=plot_type)


def run_dataset(exp_type, exp_name, stats, ds_name, ds_builder, path_data, model_builders, nb_repeat, 
		args_mlp, alpha, nb_jobs, train_size=None, **kwargs):
	path_saves = path_data / 'saves' / ds_name
	path_saves.mkdir(parents=True, exist_ok=True)
	path_images = path_data / 'images' / ds_name
	path_images.mkdir(parents=True, exist_ok=True)

	if exp_type == 'openml' and not check_dataset(ds_builder):
		logger.info('No valid column for this dataset')
		return

	if exp_type == 'toy':
		toy_graphics = ToyGraphics(len(model_builders))
	calibration_graphics = CalibrationGraphics(len(filter_dict(model_builders, gaussian_predictors)))
	mixture_size = args_mlp['mixture_size']

	for model_name, model_builder in model_builders.items():
		logger.info(f'Computing model {model_name}...')
		if exp_type == 'toy':
			datasets = [ds_builder(train_size=train_size) for _ in range(nb_repeat)]
		else:
			datasets = [make_dataset(ds_builder) for _ in range(nb_repeat)]
		trained_models = train_models_adapted(datasets, model_name, model_builder, args_mlp,
				nb_jobs, nb_repeat, **kwargs)

		for run, (ds, trained_model) in enumerate(zip(datasets, trained_models)):
			last_epoch, pred_test, pred_val, train_time, prediction_time = trained_model
			# Compute the metrics
			cal_model = PostHocCalibration()
			cal_model.fit(pred_val, ds.valid.y)
			if mixture_size == 1:
				metrics = gaussian_metrics_values(pred_test, ds, alpha, cal_model)
			else:
				metrics = multimodal_gaussian_metrics_values(pred_test, ds, alpha, cal_model)

			# Store the metrics
			stats.add(ds.name, model_name, run, metrics, last_epoch, train_time, prediction_time)
			with open(path_saves / f'{exp_name}_stats.pickle', 'wb') as f: # Resave the whole stats at the end of each dataset
				pickle.dump(stats, f)
			# Plot
			if run == 0:
				if exp_type == 'toy':
					plot_toy(toy_graphics, ds, pred_test, alpha, model_name, mixture_size)
				if model_name in gaussian_predictors:
					calibration_graphics.plot(pred_test, ds, cal_model, model_name)
	
	if exp_type == 'toy':
		toy_graphics.save(path_images, exp_name)
	calibration_graphics.save(path_images, exp_name)


def get_model_builders(methods_list, alpha, dropout_rate, ensemble_size, nb_samples):
	model_builders = get_gaussian_model_builders(alpha=alpha, dropout_rate=dropout_rate,
			ensemble_size=ensemble_size, nb_samples=nb_samples)
	to_pop = ['Gaussian process', 'SVI', 'Quantile-based PI GB'] # Don't use these models
	for mb in to_pop:
		model_builders.pop(mb)

	if methods_list:
		model_builders = filter_dict(model_builders, methods_list)
	logger.debug('Models: ' + ', '.join(model_builders.keys()))
	return model_builders


def get_datasets(exp_type, datasets_list):
	if exp_type == 'toy':
		dataset_builders = get_toy_dataset_builders()
		dataset_builders = filter_dict(dataset_builders, datasets_list)
	else:
		dataset_builders = get_openml_datasets()
		#['boston', 'Moneyball', 'house_sales', 'us_crime', 'MIP-2016-regression', 'abalone', 'diamonds', 'space_ga']
		dataset_builders = filter_dict(dataset_builders, ['boston', 'us_crime', 'house_sales',])
	return dataset_builders


def get_metrics_name(mixture_size):
	return gaussian_metrics_names if mixture_size == 1 else multimodal_gaussian_metrics_names


def run_experiment(exp_type, exp_name, dataset_builders, metrics_name, path_data, alpha, **kwargs):
	for ds_name, ds_builder in dataset_builders.items():
		logger.info(f'Evaluating dataset {ds_name}...')
		stats = StatsStorage(columns=(
			'Dataset', 'Model', 'Run', *metrics_name, 'Epochs', 'Training time (s)', 'Prediction time (s)'
		))
		run_dataset(exp_type, exp_name, stats, ds_name, ds_builder, path_data, alpha=alpha, **kwargs)
		make_tables_gaussian_dataset(exp_name, path_data, stats, ds_name, alpha)


def run(exp_type, exp_name, path_root, small_test, debug, methods_list, datasets_list, units_size, nb_hidden,
		alpha, dropout_rate, ensemble_size, nb_samples, mixture_size, **kwargs):
	use_cpu = True
	if use_cpu:
		tf.config.set_visible_devices([], 'GPU')
	reset_seeds(1)

	logger.setLevel(logging.DEBUG if debug else logging.INFO)

	logger.info('=' * 20)
	logger.info(f'Experiment {exp_name}')
	logger.info('=' * 20)

	data_dir = 'data_toy' if exp_type == 'toy' else 'data_openml'
	path_data = path_root / data_dir

	if methods_list is not None:
		methods_list = methods_list.split(',')
	if datasets_list is not None:
		datasets_list = datasets_list.split(',')

	args_mlp = {
		'units_size': units_size,
		'nb_hidden': nb_hidden,
		'mixture_size': mixture_size
	}

	metrics_name = get_metrics_name(mixture_size)
	dataset_builders = get_datasets(exp_type, datasets_list)
	model_builders = get_model_builders(methods_list, alpha, dropout_rate, ensemble_size, nb_samples)
	assert len(dataset_builders) > 0
	assert len(model_builders) > 0
	run_experiment(exp_type, exp_name, dataset_builders, metrics_name, path_data=path_data,
			model_builders=model_builders, args_mlp=args_mlp, alpha=alpha, **kwargs)


if __name__ == '__main__':
	parser = make_parser()
	parser.add_argument('exp_name', help='The name of the experiment (used in the file names)')
	kwargs = vars(parser.parse_args())
	run(**kwargs)