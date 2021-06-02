from tensorflow.keras.losses import mean_squared_error
import logging

from models.mlp import mlp
from losses import GaussianNLL, GaussianCRPS


logger = logging.getLogger('thesis')

class MLEGaussianModel:
	def __init__(self, args_mlp, dropout_rate=0):
		self.model = mlp(add_std_output=True, dropout_rate=dropout_rate, **args_mlp)
		self.model.compile(loss=GaussianNLL(), optimizer='adam', metrics=[])
	
	def fit(self, x, y, **kwargs):
		return self.model.fit(x, y, verbose=logger.isEnabledFor(logging.DEBUG), **kwargs)

	def predict(self, x):
		return self.model(x).numpy()


class CRPSGaussianModel:
	def __init__(self, args_mlp, dropout_rate=0):
		self.model = mlp(add_std_output=True, dropout_rate=dropout_rate, **args_mlp)
		self.model.compile(loss=GaussianCRPS(), optimizer='adam', metrics=[])
	
	def fit(self, x, y, **kwargs):
		return self.model.fit(x, y, verbose=logger.isEnabledFor(logging.DEBUG), **kwargs)

	def predict(self, x):
		return self.model(x).numpy()