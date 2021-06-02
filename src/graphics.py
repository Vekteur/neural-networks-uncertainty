import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import numpy as np
import math
import itertools
from scipy.stats import norm

from metrics import empirical_frequencies, l2_calibration
from utils import unnormalized_mean_and_std, make_normal_mixture, make_normal_mixture_from_prediction, gaussian_mixture_to_gaussian

sns.set()
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def save_plot(path=None):
	path = path.with_suffix('.png')
	path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(path, dpi=150)
	plt.close()

def plot_or_save(path=None):
	if path:
		save_plot(path)
	else:
		plt.show()

def add_outer_legend(fig, xlabel, ylabel, fontsize=11):
	big_ax = fig.add_subplot(111, frameon=False)
	big_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
	big_ax.grid(False)
	big_ax.set_xlabel(xlabel, fontsize=11)
	big_ax.yaxis.labelpad = 10
	big_ax.set_ylabel(ylabel, fontsize=11)

def add_metrics_legend(axis, metrics):
	blank = mpl.patches.Rectangle((0, 0), 0.5, 0.5, fc='w', fill=False, edgecolor='none', linewidth=0)
	legend = [f'{name}: {float(value):.5}' for name, value in metrics.items()]
	axis.legend([blank] * len(metrics), legend, ncol=2, bbox_to_anchor=(0.5, 1.1), loc='lower center', handletextpad=0., labelspacing=.3)


def plot_predictions_single_gaussian(axis, x_test, pred_means, pred_stds, alpha, coverage_name):
	axis.plot(x_test, pred_means, color='orange', label='Predicted mean')
	left, right = norm(pred_means, pred_stds).interval(1 - alpha)
	axis.fill_between(x_test, left, right, color='orange', alpha=0.2, label=coverage_name)

def plot_predictions_gaussian(axis, x_test, pred_means, pred_stds, pred_mixes, alpha, coverage_name):
	for i in range(pred_mixes.shape[1]):
		s = 10 # Step
		for j in range(0, pred_mixes.shape[0] - 1, s):
			axis.plot(x_test[j:j+s], pred_means[j:j+s, i], color='orange', label='Predicted mean' if i == j == 0 else None)
			left, right = norm(pred_means[j:j+s, i], pred_stds[j:j+s, i]).interval(1 - alpha)
			axis.fill_between(x_test[j:j+s], left, right, color='orange', alpha=0.2 * pred_mixes[j, i], label=coverage_name if i == j == 0 else None)

def plot_predictions_density(axis, x_test, y_train, pred_means, pred_stds, pred_mixes):
	left, right = x_test.min(), x_test.max()
	max_dist = y_train.max() - y_train.min()
	bottom, top = y_train.min() - max_dist / 5, y_train.max() + max_dist / 5
	dist = make_normal_mixture(pred_means, pred_stds, pred_mixes)

	y = np.linspace(bottom, top, 100)
	y = y[:, np.newaxis].repeat(x_test.shape[0], axis=1)
	pdf = dist.prob(y)

	colors_list = [(1, 1, 1, 0), (1, 0.6, 0.15, 1)]
	cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom', colors_list, N=1000)
	return axis.imshow(pdf, interpolation='bilinear', extent=(left, right, bottom, top),
			origin='lower', aspect='auto', cmap=cmap, vmin=0.01, norm=mpl.colors.LogNorm())

def plot_toy_dataset_gaussian(axis, ds, pred, alpha, name=None, scatter_alpha=0.2, plot_type='single_gaussian'):
	assert plot_type in ['single_gaussian', 'gaussian', 'density'], plot_type
	# Predictions
	pred_means, pred_stds, pred_mixes = np.moveaxis(pred, 1, 0)

	if plot_type == 'single_gaussian':
		pred_means, pred_stds = gaussian_mixture_to_gaussian(pred_means, pred_stds, axis=1)

	# Unnormalize
	pred_means, pred_stds = unnormalized_mean_and_std(pred_means, pred_stds, ds.scaler_y)
	x_train = ds.scaler_x.inverse_transform(ds.train.x.reshape(-1))
	y_train = ds.scaler_y.inverse_transform(ds.train.y)
	x_test = ds.scaler_x.inverse_transform(ds.test.x.reshape(-1))

	# Coverage
	coverage = 1 - alpha
	coverage_name = f'{int(coverage * 100)}% interval'
	
	# The test data must be sorted before plotting
	p = x_test.argsort()
	x_test = x_test[p]
	pred_means = pred_means[p]
	pred_stds = pred_stds[p]
	pred_mixes = pred_mixes[p]

	# Plot the true distribution
	for i, (true_mean, true_sd) in enumerate(ds.test.means_and_sds):
		true_mean = true_mean[p]
		true_std = true_sd[p]
		left, right = norm(true_mean, true_std).interval(1 - alpha)
		axis.plot(x_test, true_mean, color='blue', label='True mean' if i == 0 else None, alpha=0.4)
		axis.plot(x_test, left, color='blue', linestyle='--', label=coverage_name if i == 0 else None, alpha=0.4)
		axis.plot(x_test, right, color='blue', linestyle='--', alpha=0.4)
	
	# Plot the prediction
	im = None
	if plot_type == 'single_gaussian':
		plot_predictions_single_gaussian(axis, x_test, pred_means, pred_stds, alpha, coverage_name)
	elif plot_type == 'gaussian':
		plot_predictions_gaussian(axis, x_test, pred_means, pred_stds, pred_mixes, alpha, coverage_name)
	else:
		im = plot_predictions_density(axis, x_test, y_train, pred_means, pred_stds, pred_mixes)

	# Plot the training data
	axis.scatter(x_train, y_train, s=2, alpha=scatter_alpha, label=f'Training points ({y_train.shape[0]})')

	axis.set_xlim(ds.bounds())
	axis.grid(True)
	axis.set_xlabel('$x$')
	axis.set_ylabel('$y$', rotation=0)
	axis.set_title(name.strip())
	return im

def flip(items, ncol):
	return itertools.chain(*[items[i::ncol] for i in range(ncol)])

class ToyGraphics:
	def __init__(self, size, ncols=3):
		self.ncols = min(ncols, size)
		self.nrows = math.ceil(size / ncols)
		self.fig, self.axes = plt.subplots(self.nrows, self.ncols, squeeze=False, 
				figsize=(self.ncols * 4, self.nrows * 3), dpi=200, sharex=True, sharey=True)
		self.axes = self.axes.flatten()
		for i in range(size, len(self.axes)):
			self.axes[i].set_visible(False)
		self.axis_id = 0
	
	def plot(self, *args, **kwargs):
		im = plot_toy_dataset_gaussian(self.axes[self.axis_id], *args, **kwargs)
		if im is not None:
			divider = make_axes_locatable(self.axes[self.axis_id])
			cax = divider.append_axes('right', size='5%', pad=0.02)
			self.fig.colorbar(im, cax=cax, orientation='vertical')
		self.axis_id += 1
	
	def save(self, path_images, exp_name):
		handles, labels = self.axes[0].get_legend_handles_labels()
		ncol_legend = 3
		self.fig.legend(handles, labels, 
				loc='lower center', bbox_to_anchor=(0.5, 0), frameon=True, ncol=ncol_legend)
		self.fig.tight_layout(rect=[0, 0.15 / self.nrows, 1, 1])
		self.fig.savefig(path_images / f'{exp_name}.png')


def plot_calibration(axis, pred_test, ds, cal_model, model_name, num_points=100):
	pred_dist = make_normal_mixture_from_prediction(pred_test, scaler=ds.scaler_y)
	target = ds.scaler_y.inverse_transform(ds.test.y)

	probs_uncalibrated = cal_model.eval_cdf(pred_dist, target, calibrate=False)
	probs_calibrated = cal_model.eval_cdf(pred_dist, target, calibrate=True)

	axis.set_title(model_name)
	p, emp_freq = empirical_frequencies(probs_uncalibrated, num_points=num_points)
	axis.plot(p, emp_freq, 'b-', label=f'Uncalibrated')
	p, emp_freq = empirical_frequencies(probs_calibrated, num_points=num_points)
	axis.plot(p, emp_freq, 'g-', label=f'Calibrated')
	axis.plot([0, 1], [0, 1], color='black', linestyle='--')
	axis.set_xlabel('Expected confidence level')
	axis.set_ylabel('Observed confidence level')

class CalibrationGraphics:
	def __init__(self, size, ncols=3):
		self.size = size
		if self.size == 0:
			return
		self.ncols = min(ncols, size)
		self.nrows = math.ceil(size / self.ncols)
		self.fig, self.axes = plt.subplots(self.nrows, self.ncols, squeeze=False,
				figsize=(self.ncols * 3, self.nrows * 3), dpi=150, sharex=True, sharey=True)
		self.axes = self.axes.flatten()
		for i in range(size, len(self.axes)):
			self.axes[i].set_visible(False)
		self.axis_id = 0
	
	def plot(self, *args, **kwargs):
		plot_calibration(self.axes[self.axis_id], *args, **kwargs)
		self.axis_id += 1
	
	def save(self, path_images, exp_name):
		if self.size == 0:
			return
		handles, labels = self.axes[0].get_legend_handles_labels()
		ncol_legend = 2
		self.fig.legend(flip(handles, ncol_legend), flip(labels, ncol_legend), 
				loc='lower center', bbox_to_anchor=(0.5, 0), frameon=True, ncol=ncol_legend)
		self.fig.tight_layout(rect=[0, 0.1 / self.nrows, 1, 1])
		self.fig.savefig(path_images / f'{exp_name}_calibration.png')


def plot_learning_curves(axis, history, path=None):
	assert len(history) > 0
	for name, values in history.items():
		domain = np.arange(len(values)) + 0.5
		if name.startswith('val'):
			domain += 0.5
		plt.plot(domain, values, ".-", label=name)
	axis.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
	axis.set_xlim([0, len(domain) + 1])
	axis.legend(fontsize=14)
	axis.set_xlabel("Epochs")
	axis.set_ylabel("Loss")
	axis.grid(True)


def boxplot_metrics(df, metrics_names, path_images, name, n_cols=3):
	models = df['Model'].unique().tolist()
	size = len(metrics_names)
	n_rows = math.ceil(size / n_cols)
	fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, figsize=(n_cols * 4, n_rows * 2.5), dpi=200, sharey=True)
	axes = axes.flatten()
	for i in range(size, axes):
		axes[i].set_visible(False)
	
	for metric, axis in zip(metrics_names, axes.flatten()):
		axis.set_title(metric, fontsize=14)
		if metric == 'PICP':
			axis.axvline(x=0.95, linestyle='--')
		#df_ds.pivot(index='Run', columns='Model', values=metric).boxplot(vert=False, ax=axis)
		df = df.pivot(index='Run', columns='Model', values=metric)
		df = df[models[::-1]] # Keep the original order of the models
		df.boxplot(vert=False, ax=axis)
		plt.setp(axis.get_xticklabels(), rotation=20, horizontalalignment='right')

	fig.tight_layout()
	fig.savefig(path_images / f'{name}_boxplot.png')

