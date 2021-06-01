from scipy.stats import norm
import numpy as np
import random
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from contextlib import contextmanager
from timeit import default_timer
import joblib

def unnormalized_mean_and_std(mean, std, scaler):
	# The prediction and target are rescaled before computing the metrics
	# TODO: CHECK!!! (unnormalization of a normal distribution)
	mean = mean * scaler.scale_ + scaler.mean_
	std = std * scaler.scale_
	return mean, std

def make_normal(mean, std, scaler=None):
	if scaler:
		mean, std = unnormalized_mean_and_std(mean, std, scaler)
	return tfp.distributions.Normal(loc=mean, scale=std)

def make_normal_from_prediction(pred, scaler=None):
	return make_normal(*tf.unstack(pred, axis=1), scaler=scaler)

def make_normal_mixture(means, stds, mixes, scaler=None):
	if scaler:
		means, stds = unnormalized_mean_and_std(means, stds, scaler)
	means = tf.cast(tf.convert_to_tensor(means), tf.float32)
	stds = tf.cast(tf.convert_to_tensor(stds), tf.float32)
	mixes = tf.cast(tf.convert_to_tensor(mixes), tf.float32)

	return tfd.Mixture(
		cat=tfd.Categorical(probs=mixes),
		components=[
			tfd.Normal(loc=mean, scale=std)
			for mean, std in zip(tf.unstack(means, axis=1), tf.unstack(stds, axis=1))
		]
	)

def make_normal_mixture_from_prediction(pred, scaler=None):
	return make_normal_mixture(*tf.unstack(pred, axis=1), scaler=scaler)


def norm_from_quantiles(q1, x1, q2, x2):
	n1 = norm.ppf(q1)
	n2 = norm.ppf(q2)
	mean = (x1 * n2 - x2 * n1) / (n2 - n1)
	std = (x2 - x1) / (n2 - n1)
	std = np.maximum(std, 1e-6)
	return mean, std


def gaussian_mixture_to_gaussian(means, stds, axis=0):
	mean = np.mean(means, axis=axis)
	var = np.mean(means**2 + stds**2, axis=axis) - mean**2
	std = np.sqrt(var)
	return mean, std


def reset_seeds(seed):
	random.seed(seed)
	np.random.seed(seed)
	tf.random.set_seed(seed)


def filter_dict(d, keys):
	return {
		key: d[key]
		for key in keys
		if key in d
	}


@contextmanager
def elapsed_timer():
	start = default_timer()
	elapser = lambda: default_timer() - start
	yield lambda: elapser()
	end = default_timer()
	elapser = lambda: end-start


@contextmanager
def tqdm_joblib(tqdm_object):
	"""Context manager to patch joblib to report into tqdm progress bar given as argument"""
	class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
		def __init__(self, *args, **kwargs):
			super().__init__(*args, **kwargs)

		def __call__(self, *args, **kwargs):
			tqdm_object.update(n=self.batch_size)
			return super().__call__(*args, **kwargs)

	old_batch_callback = joblib.parallel.BatchCompletionCallBack
	joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
	try:
		yield tqdm_object
	finally:
		joblib.parallel.BatchCompletionCallBack = old_batch_callback
		tqdm_object.close()   

def inter(l1, l2):
	return [value for value in l1 if value in l2]
