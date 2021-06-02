import numpy as np
import pandas as pd
import openml as oml
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import logging


logger = logging.getLogger('thesis')

def get_data(ds):
	return ds.get_data(dataset_format="dataframe", target=ds.default_target_attribute)

def get_ds_shapes(openml_datasets):
	def ds_shape(ds):
		return get_data(ds)[0].shape
	return {ds.name: ds_shape(ds) for ds in openml_datasets}

def get_dataset_df(openml_datasets):
	ds_shapes = get_ds_shapes(openml_datasets)
	return pd.DataFrame({
		'Name': [name for name in ds_shapes.keys()],
		'Nb of instances': [shape[0] for shape in ds_shapes.values()],
		'Nb of features': [shape[1] for shape in ds_shapes.values()],
	}).set_index('Name').sort_values('Nb of instances')

def get_datasets():
	suite = oml.study.get_suite(269)
	logger.debug(suite)

	openml_datasets = [oml.datasets.get_dataset(ds_id) for ds_id in suite.data]
	ds_shapes = get_ds_shapes(openml_datasets)
	openml_datasets = sorted(openml_datasets, key=lambda ds: ds_shapes[ds.name][0])
	openml_datasets = {ds.name: ds for ds in openml_datasets}
	logger.debug('Loaded datasets: ' + ', '.join(openml_datasets.keys()))
	return openml_datasets


class DatasetComponent:
	def __init__(self, x=None, y=None):
		self.x = x
		self.y = y
		
class Dataset:
	def __init__(self, x, y, val_ratio, test_ratio, name):
		self.train = DatasetComponent()
		self.valid = DatasetComponent()
		self.test = DatasetComponent()
		self.name = name
		
		# Note that shuffling here is very important
		val_size = int(len(x) * val_ratio)
		test_size = int(len(x) * test_ratio)
		self.train.x, self.valid.x, self.train.y, self.valid.y = train_test_split(
			x, y, shuffle=True, test_size=val_size
		)
		self.train.x, self.test.x, self.train.y, self.test.y = train_test_split(
			self.train.x, self.train.y, shuffle=False, test_size=test_size
		)
		
		self.scaler_x = StandardScaler().fit(self.train.x)
		self.scaler_y = StandardScaler().fit(self.train.y.reshape(-1, 1))
		for comp in [self.train, self.valid, self.test]:
			comp.x = self.scaler_x.transform(comp.x)
			comp.y = self.scaler_y.transform(comp.y.reshape(-1, 1)).reshape(-1)

class InvalidDataset(Exception):
	pass

def check_dataset(openml_ds):
	try:
		ds = make_dataset(openml_ds)
	except InvalidDataset:
		return False
	return True

def make_dataset(ds):
	x, y, categorical_indicator, attribute_names = get_data(ds)
	# For some datasets, y is a ndarray instead of a dataframe
	if type(y) == np.ndarray:
		y = pd.DataFrame(y)
	# Only keep numeric features
	x = x.select_dtypes(include='number')
	if x.columns.empty:
		raise InvalidDataset('No valid column')
	x, y = x.fillna(0), y.fillna(0)
	# Categorical data could also be converted to one-hot
	x, y = x.to_numpy('float64'), y.to_numpy('float64')
	return Dataset(x, y, 0.15, 0.15, ds.name)