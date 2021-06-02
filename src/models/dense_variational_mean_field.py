"""
Source: SVI model from https://github.com/google-research/google-research/tree/master/uq_benchmark_2019
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import logging
import keras
tfd = tfp.distributions


logger = logging.getLogger('thesis')


def _posterior_mean_field(kernel_size, bias_size=0, dtype=None):
	"""Posterior function for variational layer."""
	n = kernel_size + bias_size
	c = np.log(np.expm1(1e-5))
	variable_layer = tfp.layers.VariableLayer(
			2 * n, dtype=dtype,
			initializer=tfp.layers.BlockwiseInitializer([
					keras.initializers.TruncatedNormal(mean=0., stddev=.05, seed=None),
					keras.initializers.Constant(np.log(np.expm1(1e-5)))], sizes=[n, n]))

	def distribution_fn(t):
		scale = 1e-5 + tf.nn.softplus(c + t[Ellipsis, n:])
		return tfd.Independent(tfd.Normal(loc=t[Ellipsis, :n], scale=scale),
													 reinterpreted_batch_ndims=1)
	distribution_layer = tfp.layers.DistributionLambda(distribution_fn)
	return tf.keras.Sequential([variable_layer, distribution_layer])


def _make_prior_fn(kernel_size, bias_size=0, dtype=None):
	del dtype
	loc = tf.zeros(kernel_size + bias_size)
	def distribution_fn(_):
		return tfd.Independent(tfd.Normal(loc=loc, scale=1),
													 reinterpreted_batch_ndims=1)
	return distribution_fn


def build_dense_variational(num_train_examples):
	def dense_variational(units, **kwargs):
		return tfp.layers.DenseVariational(
				units,
				make_posterior_fn=_posterior_mean_field,
				make_prior_fn=_make_prior_fn,
				kl_weight=1./num_train_examples,
				**kwargs)
	return dense_variational
