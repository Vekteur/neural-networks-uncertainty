from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
import numpy as np
import logging

logger = logging.getLogger('thesis')

class GaussianProcessModel:
	def __init__(self):
		kernel = 1.**2 * RBF() + WhiteKernel() * RBF()
		self.model = GaussianProcessRegressor(kernel, n_restarts_optimizer=2)
	
	def fit(self, x, y, **kwargs):
		self.model.fit(x, y)
		gp = self.model
		likelihood = gp.log_marginal_likelihood(gp.kernel_.theta)
		logger.info(f'Posterior (kernel: {gp.kernel_})')
		logger.info(f'Log-Likelihood: {likelihood:.3f}')

	def predict(self, x):
		mean, std = self.model.predict(x, return_std=True)
		return np.stack([mean, std], axis=1)
