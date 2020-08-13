""" Prediction module """

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import os

logger = logging.getLogger("fleetmanagement")

class PredictSellPrice:
	""" Class for predicting vehicle prices

	Parameters
	----------
	  target : name of target column

	Methods
	  tune_fit: Tunes parameters and fits model
	  fit: Fits CatBoost model
	  plot_model_fit: plots learning curves
	  predict: Predicts Sell Price
	  plot_predictions: Plots predictions against the two most important features

	Usage

	```python
	model = PredictSellPrice()

	model.tune_fit(df_train)
	model.fit(df_train)
	model.plot_model_fit()
	result = model.predict(df_target)
	model.plot_predictions()
	```

	"""

	target = 'SELLPRICE_CAR'

	def __init__(self, target: str = None):

		"""
		Initializes PredictSellPrice for providing sell car price predictions
		on the specified target dataset fitting a tree-based regression model
		to the predefined training dataset.

		Parameters
		----------
		  target: name of the variable that contains the sell price information

		Returns
		----------
		  PredictSellPrice: Object with car sell price modelling.

		"""
		self.model = CatBoostRegressor(allow_writing_files=False, random_seed=123)
		self.target = target if target else self.target
		self.df_target = None

	def tune_fit(self, df_train: pd.DataFrame):

		grid = {'learning_rate': [0.03, 0.1],
		'depth': [4, 6, 10],
		'l2_leaf_reg': [1, 3, 5, 7, 9]}

		y = df_train[[self.target]]
		X = df_train.loc[:, ~df_train.columns.isin([self.target])]

		logger.info("Starting hyperparameter tunning")
		grid_search_result = self.model.grid_search(grid, X=X, y=y)

		best_params = grid_search_result['params']

		results = pd.DataFrame(grid_search_result['cv_results'])

		logger.info("Best parameters: %s" % best_params)

		self.model = CatBoostRegressor(
				  depth = best_params['depth'],
				  l2_leaf_reg = best_params['l2_leaf_reg'],
				  learning_rate = best_params['learning_rate'],
				  early_stopping_rounds=10,
				  allow_writing_files=False,
				  random_seed=123)

		self.fit(df_train)

	def fit(self, df_train: pd.DataFrame): # TODO: allow to save model

		logger.info("Fitting model on training dataset")

		y = df_train[[self.target]]
		X = df_train.loc[:, ~df_train.columns.isin([self.target])]

		# split between train and test
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
												random_state=42)
		
		self.model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)

	def plot_model_fit(self):

		df_val = pd.DataFrame(self.model.evals_result_['validation'])\
						.rename(columns={'RMSE': 'validation'})
		df_learn = pd.DataFrame(self.model.evals_result_['learn'])\
						.rename(columns={'RMSE': 'learn'})

		df_results = pd.concat([df_learn, df_val], axis=1)

		# get evaluation scores
		iterations = df_results.index
		train = df_results['learn']
		test = df_results['validation']

		# plot learning curves
		plt.figure(figsize=(6,6))
		plt.plot(iterations, test, '-r', label='test')
		plt.plot(iterations, train, '-b', label='train')
		plt.xlabel("n iteration")
		plt.ylabel('RMSE')
		plt.legend(loc='upper right')
		plt.title('Learning curves')

	def predict(self,
				df_target: pd.DataFrame,
				file_name: str = 'sell_price_prediction',
				file_path: str = ''):

		# predict on target data
		y_pred = self.model.predict(df_target)

		# add predictions to dataset
		df_target[self.target] = y_pred

		self.df_target = df_target

		# export to csv
		output_file = os.path.join(file_path, file_name+'.csv')
		df_target[[self.target]].to_csv(output_file)

		return df_target

	def plot_predictions(self):

		ax = sns.scatterplot(x="CATALOG_PRICE_CAR", y="SELLPRICE_CAR", alpha=0.7,
					 hue="DAMAGE_CURRENT", size='TOTALMILEAGEVALUE', sizes=(10, 400),
					 data=self.df_target)
