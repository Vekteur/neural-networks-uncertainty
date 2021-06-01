import numpy as np
from scipy.stats import norm
import logging

from models.mlp import mlp
from losses import MultiQuantileLoss

logger = logging.getLogger('thesis')


class MultiQuantileModel:
	def __init__(self, args_mlp, quantiles):
		self.model = mlp(output_size=len(quantiles), **args_mlp)
		self.model.compile(loss=MultiQuantileLoss(quantiles), optimizer='adam', metrics=[])
	
	def fit(self, x, y, **kwargs):
		return self.model.fit(x, y, verbose=logger.isEnabledFor(logging.DEBUG), **kwargs)

	def predict(self, x):
		return self.model(x)[:, 0, :]


class IntervalQuantileModel(MultiQuantileModel):
	def __init__(self, args_mlp, alpha):
		super().__init__(args_mlp, [alpha / 2, 1. - alpha / 2])
		self.alpha = alpha