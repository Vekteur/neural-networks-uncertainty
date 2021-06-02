import gpflow as gpf
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

class GPflowHomoscedasticModel:
	def __init__(self):
		self.kernel = gpf.kernels.RBF() + gpf.kernels.White()
	
	def fit(self, x, y, **kwargs):
		self.model = gpf.models.GPR((x, y[:, np.newaxis]), self.kernel)
		opt = gpf.optimizers.Scipy()
		opt.minimize(self.model.training_loss, self.model.trainable_variables, options=dict(maxiter=100))
	
	def predict(self, x):
		mean, var = self.model.predict_f(x)
		mean, var = mean[:, 0], var[:, 0]
		return np.stack([mean, np.sqrt(var)], axis=1)





"""
Unfinished experiments with GPflow
"""


import matplotlib.pyplot as plt
def plot_distribution(X, Y, loc, scale):
	plt.figure(figsize=(15, 5))
	x = X.squeeze()
	p = x.argsort()
	X = X[p]
	x = x[p]
	Y = Y[p]
	loc = loc[p]
	scale = scale[p]
	for k in (1, 2):
		lb = (loc - k * scale).squeeze()
		ub = (loc + k * scale).squeeze()
		plt.fill_between(x, lb, ub, color="silver", alpha=1 - 0.05 * k ** 3)
	plt.plot(x, lb, color="silver")
	plt.plot(x, ub, color="silver")
	plt.plot(X, loc, color="black")
	plt.scatter(X, Y, color="gray", alpha=0.8)
	plt.show()
	plt.close()

class GPflowHeteroscedasticModel:
	def __init__(self):
		self.kernel = gpf.kernels.RBF()
	
	def fit(self, x, y, **kwargs):
		pass
	
	def predict(self, x):
		likelihood = gpf.likelihoods.HeteroskedasticTFPConditional(
			distribution_class=tfp.distributions.Normal,  # Gaussian Likelihood
			scale_transform=tfp.bijectors.Exp(),  # Exponential Transform
		)
		kernel = gpf.kernels.SeparateIndependent(
			[
				gpf.kernels.SquaredExponential(),  # This is k1, the kernel of f1
				gpf.kernels.SquaredExponential(),  # this is k2, the kernel of f2
			]
		)

		# TEST POINTS
		M = 500  # Number of inducing variables for each f_i

		# Initial inducing points position Z
		Z = np.linspace(x.min(), x.max(), M)[:, None]  # Z must be of shape [M, 1]

		inducing_variable = gpf.inducing_variables.SeparateIndependentInducingVariables(
			[
				gpf.inducing_variables.InducingPoints(Z),  # This is U1 = f1(Z1)
				gpf.inducing_variables.InducingPoints(Z),  # This is U2 = f2(Z2)
			]
		)
		self.model = gpf.models.SVGP(
			kernel=kernel,
			likelihood=likelihood,
			inducing_variable=inducing_variable,
			num_latent_gps=likelihood.latent_dim,
		)
		loss_fn = self.model.training_loss_closure((x, y[:, np.newaxis]))

		gpf.utilities.set_trainable(self.model.q_mu, False)
		gpf.utilities.set_trainable(self.model.q_sqrt, False)

		variational_vars = [(self.model.q_mu, self.model.q_sqrt)]
		natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.1)

		adam_vars = self.model.trainable_variables
		adam_opt = tf.optimizers.Adam(0.01)

		@tf.function
		def optimisation_step():
			natgrad_opt.minimize(loss_fn, variational_vars)
			adam_opt.minimize(loss_fn, adam_vars)

		epochs = 500
		log_freq = 20

		for epoch in range(1, epochs + 1):
			optimisation_step()

			# For every 'log_freq' epochs, print the epoch and plot the predictions against the data
			if epoch % log_freq == 0 and epoch > 0:
				print(f"Epoch {epoch} - Loss: {loss_fn().numpy() : .4f}")
				Ymean, Yvar = self.model.predict_y(x)
				Ymean = Ymean.numpy().squeeze()
				Ystd = tf.sqrt(Yvar).numpy().squeeze()
				#plot_distribution(x, y, Ymean, Ystd)



		"""mean, var = self.model.predict_f(x)
		mean, var = mean[:, 0], var[:, 0]
		return np.stack([mean, np.sqrt(var)], axis=1)"""