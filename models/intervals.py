import tensorflow as tf
import numpy as np
import logging
from tensorflow.keras.losses import Loss

from models.mlp import mlp
import utils

logger = logging.getLogger('thesis')


class QualityDrivenLoss(Loss):
	def __init__(self, lambda_=0.05, alpha=0.05, soften=10.):
		super().__init__()
		self.lambda_ = lambda_
		self.alpha = alpha
		self.soften = soften

	def call(self, obs, pred):
		batch_size = tf.shape(obs)[0]
		obs = obs[:, 0]
		pred = pred[:, 0, :]
		l = pred[:, 0]
		u = pred[:, 1]
		
		K_HL = tf.maximum(0., tf.sign(obs - l))
		K_HU = tf.maximum(0., tf.sign(u - obs))
		K_H = K_HL * K_HU
		
		K_SL = tf.sigmoid(self.soften * (obs - l))
		K_SU = tf.sigmoid(self.soften * (u - obs))
		K_S = K_SL * K_SU
		
		PICP_S = tf.reduce_mean(K_S)
		MPIW_c = tf.reduce_mean((u - l) * K_H)
		
		Loss_PICP = tf.cast(batch_size, tf.float32) / (self.alpha * (1 - self.alpha)) * tf.maximum(1e-8, (1 - self.alpha) - PICP_S)**2
		Loss_S = MPIW_c + self.lambda_ * Loss_PICP
		return Loss_S


class QualityDrivenModel:
	def __init__(self, args_mlp, alpha):
		self.alpha = alpha
		self.model = mlp(output_size=2, **args_mlp)
		self.model.compile(loss=QualityDrivenLoss(alpha=alpha), optimizer='adam', metrics=[])
	
	def fit(self, x, y, **kwargs):
		return self.model.fit(x, y, verbose=logger.isEnabledFor(logging.DEBUG), **kwargs)

	def predict(self, x):
		return self.model(x)[:, 0, :]


class PredictionIntervalToGaussianModel:
	def __init__(self, model):
		self.model = model
	
	def fit(self, x, y, **kwargs):
		return self.model.fit(x, y, **kwargs)

	def predict(self, x):
		preds = self.model.predict(x)
		low, high = np.moveaxis(preds, 0, 1)
		alpha = self.model.alpha
		mean, std = utils.norm_from_quantiles(alpha / 2, low, 1. - alpha / 2, high)
		return np.stack([mean, std, np.ones(mean.shape)], axis=1)[:, :, np.newaxis]