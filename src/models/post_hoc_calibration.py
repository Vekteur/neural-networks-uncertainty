import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.isotonic import IsotonicRegression

from utils import make_normal_mixture_from_prediction

def normal_dist(pred):
	pred_mean, pred_std = np.moveaxis(pred, 0, 1)
	dist = tfp.distributions.Normal(loc=tf.cast(pred_mean, tf.double), scale=tf.cast(pred_std, tf.double))
	return dist


class PostHocCalibration:
	def fit(self, pred_val, y_val): # Takes the predictions of an already trained model
		x_cal, y_cal = self.make_calibration_dataset(pred_val, y_val)
		self.model = self.make_calibration_model(x_cal, y_cal)
	
	def make_calibration_dataset(self, pred_val, y_val):
		probs = make_normal_mixture_from_prediction(pred_val).cdf(y_val).numpy()
		x_cal = np.sort(probs)
		y_cal = np.linspace(0, 1, len(x_cal))
		return x_cal, y_cal

	def make_calibration_model(self, x, y):
		return IsotonicRegression(out_of_bounds='clip').fit(x, y)

	def eval_cdf(self, pred_dist, y, calibrate=True):
		result = pred_dist.cdf(y).numpy()
		if calibrate:
			result = self.model.transform(result)
		return result
