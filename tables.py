from random import gauss
import regex as re
import numpy as np
import pandas as pd
import scipy
from IPython.display import display, Markdown, Latex

from models.gaussian_models import gaussian_predictors, interval_predictors
from utils import inter
from graphics import boxplot_metrics


def to_tabularx(s, width=1.):
	pattern = r'\\begin{tabular}{(.*)}'
	
	def repl_begin_table(match):
		cols = r'p{.24\textwidth}'
		cols += (len(match.group(1)) - 1) * 'X'
		return rf'\begin{{tabularx}}{{{width}\textwidth}}{{{cols}}}'
	
	s = re.sub(pattern, repl_begin_table, s)
	s = re.sub(r'\\end{tabular}', r'\\end{tabularx}', s)
	return s

def add_horizontal_lines(s):
	return s.replace('\\\\\n', '\\\\ \\midrule\n')

def bold_columns(df):
	return df.rename(columns=lambda col: rf'\textbf{{{col}}}')

def make_bold_values(series, indices):
	series[indices] = series[indices].apply(lambda x: rf'\textbf{{{x}}}')

def make_bold_values_full_df(df, alpha):
	df = df.copy()

	def index_seq(x, index=0):
		return np.array([p[index] for p in x])

	def argmin(x):
		if x.size == 0:
			min_value = np.nan
		else: # Throws exception if x is empty
			min_value = np.nanmin(x)
		return (x == min_value).nonzero()[0]
	
	def min_indices(x):
		return argmin(index_seq(x))
	
	def picp_indices(x):
		x = index_seq(x)
		return argmin(np.abs(x - (1 - alpha)))

	for col in df.columns:
		x = df[col]
		if col == 'PICP':
			indices = picp_indices(x)
		else:
			indices = min_indices(x)
		
		df[col] = df[col].apply(format)
		make_bold_values(df[col], indices)
	return df

def format(x):
	mean, sem = x
	if np.isnan(mean):
		return 'NA'
	s = f'{mean:#.3}'
	if sem is not None:
		s += f' $\pm$ {sem:#.2}'
	return s

def agg_run(x):
	mean = np.mean(x)
	sd = None
	if len(x) > 1:
		sd = scipy.stats.sem(x, ddof=1)
	return (mean, sd)

def filter(df, columns=None, labels=None):
	if columns:
		df = df[inter(df.columns, columns)]
	if labels:
		df = df.loc[inter(df.index, labels)]
	return df

def filter_and_bold(df, alpha, columns=None, labels=None):
	df = filter(df, columns=columns, labels=labels)
	return make_bold_values_full_df(df, alpha)

def save_df(name, path_tables, df, width=1.):
	with open(path_tables / f'{name}.tex', 'w') as f:
		df = bold_columns(df.reset_index()).to_latex(index=False, escape=False)
		df_str = add_horizontal_lines(to_tabularx(df, width=width))
		f.write(df_str)
	

def make_tables_gaussian_dataset(exp_name, path_tables, df, ds_name, alpha, path_images=None):
	path_tables /= ds_name
	path_tables.mkdir(parents=True, exist_ok=True)
	display(Markdown(f'### {ds_name}'))
	df = df.reset_index(drop=True)

	# Remove the runs with exceeding NLL
	if 'Log score' in df.columns:
		valid = (df['Log score'] < 100) | ~df['Model'].isin(gaussian_predictors)
	else:
		valid = np.full(len(df), True, dtype=bool)
	df_invalid = df[~valid]
	df = df[valid]

	# Group by model and aggregate
	df = df.groupby('Model', sort=False).agg(agg_run).drop(columns='Run')
	display(df)

	metrics_columns = ['Log score', 'CRPS', 'Interval score', 'MPIW', 'PICP', 'L2 cal.', 'RMSE', 'MAE']
	df_metrics = filter_and_bold(df, alpha, columns=metrics_columns)
	perf_columns = ['Epochs', 'Training time (s)', 'Prediction time (s)']
	df_perf = filter_and_bold(df, alpha, columns=perf_columns)
	
	dist_columns = ['Log score', 'CRPS', 'L2 cal.', 'L2 cal. (post-hoc)']
	df_dist = filter_and_bold(df, alpha, columns=dist_columns, labels=gaussian_predictors)
	interval_columns = ['Interval score', 'MPIW', 'PICP']
	df_interval = filter_and_bold(df, alpha, columns=interval_columns, labels=interval_predictors)

	save_df(f'{exp_name}', path_tables, df, width=1.2)
	save_df(f'{exp_name}_metrics', path_tables, df_metrics, width=1.15)
	save_df(f'{exp_name}_perf', path_tables, df_perf)
	save_df(f'{exp_name}_dist', path_tables, df_dist, width=1.15)
	save_df(f'{exp_name}_interval', path_tables, df_interval)

	if len(df_invalid) > 0:
		display(Markdown(f'### Runs with exceeding log score'))
		display(df_invalid)
		save_df(f'{exp_name}_invalid', path_tables, df_invalid, width=1.2)
