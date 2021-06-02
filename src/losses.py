import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import Loss
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
import numpy as np

from utils import make_normal_mixture_from_prediction, make_normal_from_prediction


class GaussianNLL(Loss):
	def call(self, obs, pred):
		obs = obs[:, 0]
		# Apparently Keras removes the 3rd dimension if the loss is used as a metric
		if len(pred.shape) == 2:
			pred = pred[:, :, tf.newaxis]
		dist = make_normal_mixture_from_prediction(pred)
		return tf.reduce_mean(-dist.log_prob(obs))


class GaussianNLLSum(Loss):
	def call(self, obs, pred):
		obs = obs[:, 0]
		dist = make_normal_mixture_from_prediction(pred)
		return tf.reduce_sum(-dist.log_prob(obs))


class GaussianNLLFixedStdDev(Loss):
	def __init__(self, std=1.):
		super().__init__()
		self.std = std
	
	def call(self, obs, pred):
		dist = tfp.distributions.Normal(loc=pred, scale=self.std)
		return tf.reduce_sum(-dist.log_prob(obs))


class MultiQuantileLoss(Loss):
	def __init__(self, quantiles):
		super().__init__()
		self.quantiles = np.array(quantiles)

	def call(self, obs, pred):
		pred = pred[:, 0, :]
		diff = obs - pred
		mask = tf.where(diff < 0, 1., 0.)
		loss_per_quantile = diff * (self.quantiles[np.newaxis, :] - mask)
		return tf.reduce_mean(loss_per_quantile, axis=1)


class GaussianCRPS(Loss):
	def call(self, obs, pred):
		assert len(pred.shape) == 3
		assert pred.shape[2] == 1
		obs = obs[:, 0]
		pred = pred[:, :2, 0]
		mean, std = tf.unstack(pred, axis=1)
		dist = tfp.distributions.Normal(loc=0, scale=1)
		z = (obs - mean) / std
		return tf.reduce_mean(std * (2 * dist.prob(z) + z  * (2 * dist.cdf(z) - 1) - 1 / tf.sqrt(np.pi)))



if __name__ == '__main__':
	loss = MultiQuantileLoss([0.1, 0.2])
	print(loss.call(np.array([[0.2, 0.3]]), np.array([[0.3, 0.4]])))