
from tensorflow.keras.callbacks import EarlyStopping, History
import numpy as np
import logging
from tqdm import tqdm
import joblib


from utils import elapsed_timer, tqdm_joblib


logger = logging.getLogger('thesis')

def get_last_epoch(history):
	losses = history.history['val_loss']
	last_epoch = np.argmin(losses) + 1
	logger.debug(f'Last epoch: {last_epoch}')
	return last_epoch

def fit_model(model, ds, max_epochs=1000, patience=50, batch_size=32):
	callbacks = [EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)]
	history = model.fit(ds.train.x, ds.train.y, batch_size=batch_size, shuffle=True,
			validation_data=(ds.valid.x, ds.valid.y), epochs=max_epochs, callbacks=callbacks)
	
	nb_epoch = None
	if type(history) == History:
		nb_epoch = get_last_epoch(history)
	elif type(history) == list:
		nb_epochs = [get_last_epoch(single_hist) for single_hist in history]
		nb_epoch = sum(nb_epochs)

	return nb_epoch
	
def predict_model(model, ds_test):
	y_pred = model.predict(ds_test.x)
	logger.debug(f'y_test.shape: {ds_test.y.shape}, y_pred.shape: {y_pred.shape}')
	return y_pred

def build_fit_predict(model_name, model_builder, ds, args_mlp, **kwargs):
	# Build
	model = model_builder({**args_mlp, 'input_size': ds.train.x.shape[1]})
	# Fit
	with elapsed_timer() as train_time:
		last_epoch = fit_model(model, ds, **kwargs)
	# Predict
	with elapsed_timer() as prediction_time:
		pred_test = predict_model(model, ds.test)
	pred_val = predict_model(model, ds.valid)
	
	return last_epoch, pred_test, pred_val, train_time(), prediction_time()

def train_models(datasets, model_name, model_builder, args_mlp, nb_jobs, nb_repeat, **kwargs):
	if nb_jobs == 1:
		trained_models = [
			build_fit_predict(model_name, model_builder, ds, args_mlp, **kwargs)
			for ds in tqdm(datasets)
		]
	else:
		with tqdm_joblib(tqdm(desc="Training", total=nb_repeat)) as progress_bar:
			trained_models = joblib.Parallel(n_jobs=nb_jobs)(
				joblib.delayed(build_fit_predict)(model_name, model_builder, ds, args_mlp, **kwargs)
				for ds in datasets
			)
	return trained_models


def train_models_adapted(datasets, model_name, model_builder, args_mlp, nb_jobs, nb_repeat, **kwargs):
	if model_name == 'Bayes by backprop':
		args_mlp = args_mlp.copy()
		args_mlp['nb_hidden'] = 1
		kwargs = kwargs.copy()
		kwargs['patience'] = 30
	return train_models(datasets, model_name, model_builder, args_mlp, nb_jobs, nb_repeat, **kwargs)