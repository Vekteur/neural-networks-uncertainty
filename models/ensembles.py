import numpy as np


class ModelSamplerGaussian:
	def __init__(self, model_builder, nb_samples):
		self.model = model_builder()
		self.nb_samples = nb_samples
	
	def fit(self, x, y, **kwargs):
		return self.model.fit(x, y, **kwargs)

	def predict(self, x):
		preds = np.array([
			self.model.predict(x)
			for _ in range(self.nb_samples)
		])
		# Shape: (ensemble_size, batch_size, 3, mixture_size)
		# Make a bigger mixture with all the predictions
		preds = preds.transpose((1, 2, 0, 3))
		preds = preds.reshape(preds.shape[:2] + (-1,))
		# Shape: (batch_size, 3, ensemble_size * mixture_size)
		# Scale the mixture
		preds[:, 2, :] /= self.nb_samples
		return preds


class ModelSamplerInterval:
	def __init__(self, model_builder, nb_samples, alpha):
		self.model = model_builder()
		self.nb_samples = nb_samples
		self.alpha = alpha
	
	def fit(self, x, y, **kwargs):
		return self.model.fit(x, y, **kwargs)

	def predict(self, x):
		preds = np.array([
			self.model.predict(x)[:, 0]
			for _ in range(self.nb_samples)
		])
		low = np.quantile(preds, self.alpha / 2, 0)
		high = np.quantile(preds, 1. - self.alpha / 2, 0)
		return np.stack([low, high], axis=1)


class EnsembleGaussian:
	def __init__(self, model_builder, ensemble_size):
		self.models = [
			model_builder()
			for _ in range(ensemble_size)
		]
	
	def fit(self, x, y, **kwargs):
		hists = []
		for model in self.models:
			hists.append(model.fit(x, y, **kwargs))
		return hists

	def predict(self, x):
		preds = np.array([
			model.predict(x)
			for model in self.models
		])
		# Shape: (ensemble_size, batch_size, 3, mixture_size)
		# Make a bigger mixture with all the predictions
		preds = preds.transpose((1, 2, 0, 3))
		preds = preds.reshape(preds.shape[:2] + (-1,))
		# Shape: (batch_size, 3, ensemble_size * mixture_size)
		# Scale the mixture
		preds[:, 2, :] /= len(self.models)
		return preds