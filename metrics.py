import tensorflow as tf
import numpy as np
import properscoring
import scipy
from scipy.stats import norm

from utils import (make_normal, make_normal_mixture_from_prediction, 
	gaussian_mixture_to_gaussian
)

# Note: the NLL is computed as a mean and not a sum in order to compare it on datasets of different sizes
def nll(obs, pred_dist):
	return tf.reduce_mean(-pred_dist.log_prob(obs)).numpy()

def crps(obs, pred_dist):
	return np.mean(properscoring.crps_gaussian(obs, mu=pred_dist.loc, sig=pred_dist.scale))

def interval_bounds(alpha, pred_dist):
	interval = [alpha / 2, 1 - alpha / 2]
	interval = np.array(interval)[:, np.newaxis]
	size = pred_dist.batch_shape.as_list()[0]
	quantiles = np.broadcast_to(interval, [len(interval), size])
	lower, upper = pred_dist.quantile(quantiles)
	return lower, upper

def interval_score(alpha, obs, lower, upper):
	score = upper - lower
	score += 2 / alpha * (lower - obs) * tf.cast(tf.where(obs < lower, 1., 0.), score.dtype)
	score += 2 / alpha * (obs - upper) * tf.cast(tf.where(obs > upper, 1., 0.), score.dtype)
	return tf.reduce_mean(score).numpy()

def interval_score_from_gaussian(alpha, obs, pred_dist):
	lower, upper = interval_bounds(alpha, pred_dist)
	return interval_score(alpha, obs, lower, upper)

def mpiw(lower, upper):
	return np.mean(upper - lower)

def mpiw_from_gaussian(alpha, pred_dist):
	lower, upper = interval_bounds(alpha, pred_dist)
	return mpiw(lower, upper)

def picp(alpha, obs, lower, upper):
	captured = (lower < obs) & (obs < upper)
	return np.mean(captured)

def picp_from_gaussian(alpha, obs, pred_dist):
	lower, upper = interval_bounds(alpha, pred_dist)
	return picp(alpha, obs, lower, upper)

def empirical_frequencies(probs, num_points=100):
	p = np.linspace(0., 1., num_points)
	p_broadcast, probs_broadcast = np.broadcast_arrays(p[:, np.newaxis], probs[np.newaxis, :])
	emp_freq = np.mean(np.where(probs_broadcast <= p_broadcast, 1., 0.), axis=1)
	return p, emp_freq

def l2_calibration(probs, num_points=100):
	p, emp_freq = empirical_frequencies(probs, num_points=num_points)
	l2_cal = (emp_freq - p)**2
	return np.mean(l2_cal)

def l2_calibration_from_gaussian(obs, pred_dist, num_points=100):
	probs = pred_dist.cdf(obs).numpy()
	return l2_calibration(probs, num_points=num_points)

def rmse(obs, pred):
	return np.sqrt(np.mean((obs - pred)**2))

def mae(obs, pred):
	return np.mean(np.abs(obs - pred))



gaussian_metrics_names = [
	'Log score',
	'CRPS',
	'Interval score',
	'MPIW',
	'PICP',
	'L2 cal.',
	'L2 cal. (post-hoc)',
	'RMSE',
	'MAE',
]

def gaussian_metrics_values(pred_test, ds, alpha, cal_model):
	means, stds = np.moveaxis(pred_test, 1, 0)[:2]
	# Remove the mixture axis
	mean, std = gaussian_mixture_to_gaussian(means, stds, axis=1)
	pred_dist = make_normal(mean, std, scaler=ds.scaler_y)
	target = ds.scaler_y.inverse_transform(ds.test.y)

	probs_uncalibrated = cal_model.eval_cdf(pred_dist, target, calibrate=False)
	probs_calibrated = cal_model.eval_cdf(pred_dist, target, calibrate=True)
	
	return [
		nll(target, pred_dist),
		crps(target, pred_dist),
		interval_score_from_gaussian(alpha, target, pred_dist),
		mpiw_from_gaussian(alpha, pred_dist),
		picp_from_gaussian(alpha, target, pred_dist),
		l2_calibration(probs_uncalibrated),
		l2_calibration(probs_calibrated),
		rmse(target, pred_dist.loc),
		mae(target, pred_dist.loc),
	]



multimodal_gaussian_metrics_names = [
	'Log score',
	'L2 cal.',
	'L2 cal. (post-hoc)',
]

def print_multimodal(target, pred_dist):
	print(target)
	for component in pred_dist.components:
		print(component.loc)
		print(component.scale)
	print(pred_dist.cat.probs)

def multimodal_gaussian_metrics_values(pred_test, ds, alpha, cal_model):
	pred_dist = make_normal_mixture_from_prediction(pred_test, scaler=ds.scaler_y)
	target = ds.scaler_y.inverse_transform(ds.test.y)

	probs_uncalibrated = cal_model.eval_cdf(pred_dist, target, calibrate=False)
	probs_calibrated = cal_model.eval_cdf(pred_dist, target, calibrate=True)
	
	return [
		nll(target, pred_dist),
		l2_calibration(probs_uncalibrated),
		l2_calibration(probs_calibrated),
	]



"""
Test of numerical integration in order to find CRPS after post-hoc calibration.
It did not gave precise results and could be investigated futher.
"""

def make_calibration_cdf(cal_model, mean, sd):
	dist = norm(mean, sd)
	def cdf(x):
		return cal_model.model.transform(dist.cdf(np.array([x])))[0]
	return cdf

def crps_quadrature(obs, pred_means, pred_stds, cal_model):
	scores = []
	for x, pred_mean, pred_std in zip(obs, pred_means, pred_stds):
		scores.append(properscoring.crps_quadrature(x,
				make_calibration_cdf(cal_model, pred_mean, pred_std),
				xmin=-np.inf, xmax=np.inf, tol=0.))
				# tol is set to 0. because the integral is often not precise
	return np.mean(scores)

calibration_metrics_names = [
	'L2 cal.',
	'L2 cal. (post-hoc)',
	#'CRPS (post-hoc)',
]

def calibration_metrics_values(pred_test, ds, cal_model):
	pred_dist = make_normal_mixture_from_prediction(pred_test, scaler=ds.scaler_y)
	target = ds.scaler_y.inverse_transform(ds.test.y)

	probs_uncalibrated = cal_model.eval_cdf(pred_dist, target, calibrate=False)
	probs_calibrated = cal_model.eval_cdf(pred_dist, target, calibrate=True)

	import warnings
	with warnings.catch_warnings():
		warnings.simplefilter('ignore', scipy.integrate.IntegrationWarning)
	
		metrics = [
			l2_calibration(probs_uncalibrated),
			l2_calibration(probs_calibrated),
			#crps_quadrature(target, pred_mean, pred_std, cal_model),
		]
		return metrics