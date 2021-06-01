import pandas as pd


class StatsStorage:
	def __init__(self, columns):
		self.columns = columns
		self.data_metrics = []
	
	def add(self, ds_name, model_name, run, metrics_values, epochs, train_time, prediction_time):
		self.data_metrics.append((ds_name, model_name, run, *metrics_values, epochs, train_time, prediction_time))
	
	def build_df(self):
		return pd.DataFrame(self.data_metrics, columns=self.columns)