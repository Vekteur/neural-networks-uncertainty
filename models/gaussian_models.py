from models.ensembles import ModelSamplerGaussian, ModelSamplerInterval, EnsembleGaussian
from models.bayes_by_backprop import BayesByBackpropModelSampler
from models.svi import SVISampler
from models.quantile_models import IntervalQuantileModel
from models.intervals import PredictionIntervalToGaussianModel, QualityDrivenModel
from models.sklearn_models import QuantileGradientBoostingModel, NGBoostModel
from models.gaussian_process import GaussianProcessModel, GPflowHomoscedasticModel, GPflowHeteroscedasticModel
from models.misc import MLEGaussianModel, CRPSGaussianModel


class MLEGaussianEnsembleModel(EnsembleGaussian):
	def __init__(self, args_mlp, ensemble_size):
		super().__init__(lambda: MLEGaussianModel(args_mlp), ensemble_size)


class MLE_MCDropoutModel(ModelSamplerGaussian):
	def __init__(self, args_mlp, dropout_rate, nb_samples):
		super().__init__(lambda: MLEGaussianModel(args_mlp, dropout_rate), nb_samples)


class QuantileBasedGaussianModel(PredictionIntervalToGaussianModel):
	def __init__(self, args_mlp, alpha):
		super().__init__(IntervalQuantileModel(args_mlp, alpha))


class HQPIGaussianModel(PredictionIntervalToGaussianModel):
	def __init__(self, args_mlp, alpha):
		super().__init__(QualityDrivenModel(args_mlp, alpha))


class QuantileGradientBoostingGaussianModel(PredictionIntervalToGaussianModel):
	def __init__(self, alpha):
		super().__init__(QuantileGradientBoostingModel(alpha))


# All models output the mean and standard deviation of a normal distribution
def get_gaussian_model_builders(alpha=0.05, dropout_rate=0.2, ensemble_size=5, nb_samples=100):
	all_gaussian_model_builders = {
		'MLE': lambda args_mlp: MLEGaussianModel(args_mlp),
		'MLE ensemble': lambda args_mlp: MLEGaussianEnsembleModel(args_mlp, ensemble_size),
		'MLE MC dropout': lambda args_mlp: MLE_MCDropoutModel(args_mlp, dropout_rate, nb_samples),
		'CRPS min.': lambda args_mlp: CRPSGaussianModel(args_mlp),
		'Quantile-based PI': lambda args_mlp: QuantileBasedGaussianModel(args_mlp, alpha),
		'HQPI': lambda args_mlp: HQPIGaussianModel(args_mlp, alpha),
		'SVI': lambda args_mlp: SVISampler(args_mlp, nb_samples),
		'Bayes by backprop': lambda args_mlp: BayesByBackpropModelSampler(args_mlp, nb_samples),
		'NGBoost': lambda args_mlp: NGBoostModel(),
		#'Quantile-based PI GB': lambda args_mlp: QuantileGradientBoostingGaussianModel(alpha),
		'Gaussian process': lambda args_mlp: GaussianProcessModel(),
		'GPflow': lambda args_mlp: GPflowHomoscedasticModel(),
	}
	return all_gaussian_model_builders

gaussian_predictors = ['MLE', 'MLE ensemble', 'MLE MC dropout', 'CRPS min.', 'SVI', 
		'Bayes by backprop', 'NGBoost', 'Gaussian process', 'GPflow']
interval_predictors = ['Quantile-based PI', 'HQPI']