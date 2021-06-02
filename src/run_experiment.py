import argparse

from args_parser import make_parser, add_general_args
from run import run
from utils import inter


def get_experiment_args(exp_type, exp_name, general_args):
	kwargs = {}

	methods_list = None
	if exp_type == 'openml':
		kwargs['nb_repeat'] = 20
		kwargs['max_epochs'] = 400
	else:
		kwargs['train_size'] = 1000
	
	if exp_name == 'bimodal':
		kwargs['mixture_size'] = 2
	if exp_name == 'multimodal':
		kwargs['mixture_size'] = 5
	if exp_name in ['gaussian_large', 'bimodal_dataset', 'bimodal']:
		kwargs['train_size'] = 5000

	if exp_name == 'bimodal':
		methods_list = ['MLE', 'MLE ensemble', 'MLE MC dropout', 'Bayes by backprop']
		

	kwargs['datasets_list'] = 'heteroscedastic2'
	if exp_name in ['bimodal', 'bimodal_dataset']:
		kwargs['datasets_list'] = 'bimodal'
	elif exp_name in ['epistemic']:
		kwargs['datasets_list'] = 'spaced'
	
	if general_args['small_test']:
		kwargs['nb_repeat'] = 1
		kwargs['max_epochs'] = 5
		filtered = ['MLE', 'MLE MC dropout', 'CRPS', 'Quantile-based PI']
		if methods_list is None:
			methods_list = filtered
		else:
			methods_list = inter(methods_list, filtered)
	
	if methods_list is not None:
		kwargs['methods_list'] = ','.join(methods_list)
	return kwargs

def run_experiment_by_name(exp_type, exp_name, general_args):
	if exp_type == 'toy':
		assert exp_name in ['gaussian', 'gaussian_large', 'epistemic', 'bimodal_dataset', 'bimodal']
	else:
		assert exp_name in ['gaussian', 'multimodal']

	kwargs = get_experiment_args(exp_type, exp_name, general_args)
	parser_args = [
		f'--{key}={value}'
		for key, value in kwargs.items()
	]
	kwargs = vars(make_parser(general_args=False).parse_args(parser_args))
	kwargs = {**kwargs, **general_args}
	run(exp_type, exp_name, **kwargs)

def run_experiments_by_name(exp_type, exps_list, **general_args):
	for exp_name in exps_list:
		run_experiment_by_name(exp_type, exp_name, general_args)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Experiments')
	add_general_args(parser)
	parser.add_argument('exps_list', nargs='+')
	kwargs = vars(parser.parse_args())
	run_experiments_by_name(**kwargs)
