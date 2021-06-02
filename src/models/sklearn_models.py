from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from ngboost import NGBRegressor
from scipy.stats import norm
import numpy as np
import logging

import utils


logger = logging.getLogger('thesis')

class QuantileGradientBoostingModel:
	def __init__(self, alpha):
		self.alpha = alpha
		self.model1 = self.build_model(alpha / 2)
		self.model2 = self.build_model(1. - alpha / 2)
	
	def build_model(self, quantile):
		return GradientBoostingRegressor(loss='quantile', alpha=quantile,
				n_estimators=250, max_depth=3, learning_rate=.1, min_samples_leaf=9,
				min_samples_split=9, verbose=logger.isEnabledFor(logging.DEBUG))
	
	def fit(self, x, y, **kwargs):
		self.model1.fit(x, y)
		self.model2.fit(x, y)
	
	def predict(self, x):
		low = self.model1.predict(x)
		high = self.model2.predict(x)
		return np.stack([low, high], axis=1)


class NGBoostModel:
	def __init__(self):
		self.model = NGBRegressor(Base=DecisionTreeRegressor(max_depth=2), n_estimators=600, learning_rate=0.01, verbose=logger.isEnabledFor(logging.DEBUG))

	def fit(self, x, y, validation_data, **kwargs):
		"""param_grid = {
			'n_estimators': [100, 300, 500],
			'learning_rate': [0.02, 0.01, 0.005],
			'Base': [DecisionTreeRegressor(max_depth=2), DecisionTreeRegressor(max_depth=4)]
		}
		
		grid = GridSearchCV(NGBRegressor(), param_grid, verbose=4 if logger.isEnabledFor(logging.DEBUG) else 0, refit=True, n_jobs=-1)
		grid.fit(x, y)
		logger.debug(f'Best parameters: {grid.best_params_}')
		logger.debug(f'CV results: {grid.cv_results_}')
		self.model = grid.best_estimator_"""
		x_val, y_val = validation_data
		self.model.fit(x, y, x_val, y_val)
	
	def predict(self, x):
		y_dist = self.model.pred_dist(x)
		pred = np.stack([y_dist.loc, y_dist.scale, np.ones(y_dist.loc.shape)], axis=1)
		return pred[:, :, np.newaxis]