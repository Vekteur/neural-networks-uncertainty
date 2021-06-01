import tensorflow as tf
from tensorflow.keras import optimizers
import numpy as np
import logging

from models.dense_variational_mean_field import build_dense_variational
from losses import GaussianNLL
from models.mlp import mlp


logger = logging.getLogger('thesis')

class SVISampler:
	def __init__(self, args_mlp, nb_samples):
		self.args_mlp = args_mlp
		self.nb_samples = nb_samples
	
	def fit(self, x, y, batch_size=32, **kwargs):
		num_train_examples = x.shape[0]
		self.model = mlp(add_std_output=True, dense_layer=build_dense_variational(num_train_examples), **self.args_mlp)
		self.model.compile(loss=GaussianNLL(), optimizer=optimizers.Adam(lr=0.001))
		return self.model.fit(x, y, batch_size=batch_size, verbose=logger.isEnabledFor(logging.DEBUG), **kwargs)

	def predict(self, x):
		preds = np.array([
			self.model.predict(x)
			for _ in range(self.nb_samples)
		])
		means, stds = np.moveaxis(preds, 2, 0)
		mean = np.mean(means, axis=0)
		var = np.mean(means**2 + stds**2, axis=0) - mean**2
		std = np.sqrt(var)
		return np.stack([mean, std], axis=1)