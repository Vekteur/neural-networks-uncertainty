import tensorflow as tf
from tensorflow.keras import initializers, activations, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import tensorflow_probability as tfp
from tensorflow.keras.layers import Layer, Input, Flatten, Concatenate, Dense, Dropout
from tensorflow.keras.models import Model
import numpy as np
import logging

from models.mlp import mlp
from losses import GaussianNLLFixedStdDev, GaussianNLL, GaussianNLLSum

logger = logging.getLogger('thesis')


class GaussianLayer(Layer):
	def __init__(self, shape, init_stddev=1., **kwargs):
		super().__init__(**kwargs)
		self.shape = shape
		self.init_stddev = init_stddev
	
	def compute_output_shape(self, input_shape):
		return self.shape
	
	def build(self, input_shape):
		self.mu = self.add_weight(name='mu', shape=self.shape, 
				initializer=initializers.GlorotNormal(), trainable=True)
		self.rho = self.add_weight(name='rho', shape=self.shape,
				initializer=initializers.Constant(-3.), trainable=True)
		super().build(input_shape)
	
	@property
	def sigma(self):
		return tf.math.softplus(self.rho)
	
	def log_prob(self, inputs):
		return tfp.distributions.Normal(self.mu, self.sigma).log_prob(inputs)
	
	def call(self, inputs, training=None):
		return self.mu + self.sigma * tf.random.normal(self.shape)


# Inspired from http://krasserm.github.io/2019/03/14/bayesian-neural-networks/
class DenseVariational(Layer):
	def __init__(self, units, kl_weight, activation=None, prior_std=5., **kwargs):
		super().__init__(**kwargs)
		self.units = units
		self.kl_weight = kl_weight
		self.activation = activations.get(activation)
		self.prior_std = prior_std
		self.prior = tfp.distributions.Normal(0., self.prior_std)

	def compute_output_shape(self, input_shape):
		return input_shape[0], self.units

	def build(self, input_shape):
		self.w = GaussianLayer((input_shape[1], self.units))
		self.b = GaussianLayer((self.units,))
		super().build(input_shape)

	def call(self, inputs, **kwargs):
		sample_w = self.w(())
		sample_b = self.b(())

		w_log_prior = self.prior.log_prob(sample_w)
		b_log_prior = self.prior.log_prob(sample_b)
		log_prior = w_log_prior + b_log_prior

		w_log_post = self.w.log_prob(sample_w)
		b_log_post = self.b.log_prob(sample_b)
		log_post = w_log_post + b_log_post

		self.add_loss(self.kl_weight * K.sum(log_post - log_prior))
		
		return self.activation(K.dot(inputs, sample_w) + sample_b)


class BayesByBackpropModelSampler:
	def __init__(self, args_mlp, nb_samples):
		self.args_mlp = args_mlp
		self.nb_samples = nb_samples
	
	def fit(self, x, y, batch_size=32, **kwargs):
		num_batches = x.shape[0] / batch_size
		def custom_layer(units, **kwargs):
			return DenseVariational(units, kl_weight=1. / num_batches, **kwargs)
		self.model = mlp(add_std_output=True, dense_layer=custom_layer, **self.args_mlp)
		self.model.compile(loss=GaussianNLLSum(), optimizer='adam', metrics=[GaussianNLL(name='GaussianNLL')])
		return self.model.fit(x, y, verbose=logger.isEnabledFor(logging.DEBUG), batch_size=batch_size, **kwargs)

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