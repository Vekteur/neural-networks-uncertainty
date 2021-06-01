from abc import abstractmethod
import numpy as np
from sklearn.preprocessing import StandardScaler


class DatasetBase:
	def __init__(self, name, train_size=1000, val_size=500, test_size=500, scale=True):
		self.name = name
		self.train = DatasetComponent(train_size, self.train_domain, self.f)
		self.valid = DatasetComponent(val_size, self.train_domain, self.f)
		self.test = DatasetComponent(test_size, self.test_domain, self.f)
		self.scale = scale
		if self.scale:
			self.scale_data()
	
	def scale_data(self):
		self.scaler_x = StandardScaler().fit(self.train.x)
		self.scaler_y = StandardScaler().fit(self.train.y.reshape(-1, 1))
		for comp in [self.train, self.valid, self.test]:
			comp.x = self.scaler_x.transform(comp.x)
			comp.y = self.scaler_y.transform(comp.y.reshape(-1, 1)).reshape(-1)
	
	@abstractmethod
	def f(self, x):
		pass

	@abstractmethod
	def train_domain(self, n):
		pass

	def test_domain(self, n):
		return self.train_domain(n)


class DatasetComponent:
	def __init__(self, size, domain, f):
		self.size = size
		self.x = domain(size)
		self.means_and_sds = f(self.x)
		nb_modes = len(self.means_and_sds)
		self.means_and_sds = np.array(self.means_and_sds)
		assert self.means_and_sds.shape == (nb_modes, 2, size)
		cluster = np.random.randint(len(self.means_and_sds), size=size)
		means, sds = self.means_and_sds.transpose(0, 2, 1)[cluster, np.arange(size), :].transpose(1, 0)
		self.y = np.random.normal(loc=means, scale=sds)
		self.x = self.x[:, np.newaxis]


class DatasetSpaced(DatasetBase):
	def __init__(self, *args, **kwargs):
		super().__init__('spaced', *args, **kwargs)
	
	def bounds(self):
		return (-10, 20)
	
	def f(self, x):
		y = 2 * (x / 10 + np.sin(4 * x / 10) + np.sin(13 * x / 10))
		noise_sd = np.full(x.shape, 0.5)
		return [(y, noise_sd)]
	
	def train_domain(self, n):
		#return np.random.normal(loc=5, scale=2, size=n)
		return np.random.uniform(0, 10, n)
		'''left = np.random.uniform(0.5, 2, n // 2)
		right = np.random.uniform(5, 7, n // 2)
		return np.concatenate([left, right])'''
	
	def test_domain(self, n):
		return np.linspace(*self.bounds(), n)


class DatasetHeteroscedastic(DatasetBase):
	def __init__(self, *args, **kwargs):
		super().__init__('heteroscedastic', *args, **kwargs)
	
	def bounds(self):
		return (-1, 2)
	
	def f(self, x):
		y = np.sin(x + 0.7)
		noise_sd = 0.15 * np.abs(np.abs(x) - 1)
		return [(y, noise_sd)]
	
	def train_domain(self, n):
		return np.random.uniform(*self.bounds(), n)


class DatasetHeteroscedastic2(DatasetBase):
	def __init__(self, *args, **kwargs):
		super().__init__('heteroscedastic2', *args, **kwargs)
	
	def bounds(self):
		return (0, 10)
	
	def f(self, x):
		y = 2 * (x / 10 + np.sin(4 * x / 10) + np.sin(13 * x / 10))
		noise_sd = np.sqrt((x + 1) / 10)
		return [(y, noise_sd)]
	
	def train_domain(self, n):
		return np.random.uniform(*self.bounds(), n)
	
	def test_domain(self, n):
		return np.linspace(*self.bounds(), n)


class DatasetSine(DatasetBase):
	def __init__(self, *args, **kwargs):
		super().__init__('sine', *args, **kwargs)
	
	def bounds(self):
		return (-0.5, 0.5)
	
	def f(self, x):
		y = 10 * np.sin(2 * np.pi * (x))
		noise_sd = np.full(x.shape, 1.)
		return [(y, noise_sd)]
	
	def train_domain(self, n):
		return np.random.uniform(*self.bounds(), n)


class DatasetMoreNoise(DatasetBase):
	def __init__(self, *args, **kwargs):
		super().__init__('more_noise', *args, **kwargs)
	
	def bounds(self):
		return (0, 10)
	
	def f(self, x):
		y = x * np.sin(x)
		noise_sd = 1.5 + 1. * np.random.random(x.shape)
		return [(y, noise_sd)]
	
	def train_domain(self, n):
		return np.random.uniform(*self.bounds(), n)
	
	def test_domain(self, n):
		return np.linspace(*self.bounds(), n)


class DatasetBimodal(DatasetBase):
	def __init__(self, *args, **kwargs):
		super().__init__('bimodal', *args, **kwargs)
	
	def bounds(self):
		return (0, 10)
	
	def f(self, x):
		y1 = 2 * (x / 10 + np.sin(4 * x / 10) + np.sin(13 * x / 10))
		noise_sd1 = (x + 1) / 10
		y2 = 2 * (-3 + np.sin(5 * x / 10) + np.sin(19 * x / 10))
		noise_sd2 = (15 - x) / 10
		return [(y1, noise_sd1), (y2, noise_sd2)]
	
	def train_domain(self, n):
		return np.random.uniform(*self.bounds(), n)


def split(x, train_size, valid_size):
	return x[:train_size], x[train_size:train_size + valid_size], x[train_size + valid_size:]