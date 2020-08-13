""" Pre-processing module """

import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import pandas as pd
import unidecode
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger("carnext")

def get_datasets(
	vehicles_filepath: str = "../data/CASE_VEHICLES.csv",
	damage_filepath: str = "../data/CASE_DAMAGE.csv",
	target: str = 'SELLPRICE_CAR',
	cols_to_drop: list = [
					'MODEL' # too specific
					,'FIRSTUSE' # no data
					,'TYPE' # too many categories
					,'HORSEPOWER' # no data on target set
					,'CYLINDERCAPACITY' # no data on target set
					,'COLOR' # too many categories
					,'SALES_CHANNEL' # will be imposed (B2B)
					],
	feature_selection_threshold: float = 0.01,
	plot_features: bool = True
):

	"""
	Function for reading, cleaning and processing the data needed 
	in the sell price model

	Parameters
	----------
	vehicles_filepath : filepath for vehicles dataset 
	damage_filepath : filepath for damages dataset 
	target : name of target column
	cols_to_drop : list of columns to drop from dataset
	feature_selection_threshold: threshold for feature selection
	plot_features : wether plot features or not

	Returns
	----------
	df_train_red : reduced dataset for training the model
	df_target_red : reduced dataset for making predictions
	df_merged : raw dataset that contains vehicle and damage information

	"""

	logger.info("Reading and processing datasets")

	df_vehicles = pd.read_csv(vehicles_filepath)
	df_damage = pd.read_csv(damage_filepath)
	
	# prepare damage dataset
	df_damage = _preprocess_df_damage(df_damage)

	# prepare vehicle dataset
	df_vehicles = _preprocess_df_vehicles(df_vehicles)

	# join both datasets
	df_merged = pd.merge(df_vehicles, df_damage, how='left')\
	   .set_index(['VEHICLEID','TMD_SRCE_ENTITY'])

	# clean data
	df_clean = df_merged.loc[:, ~df_merged.columns.isin(cols_to_drop)]

	# Process: get dummies for categorical columns
	df_clean = pd.get_dummies(df_clean, prefix_sep='_', drop_first=False)

	# Target dataset
	df_target = df_clean[df_merged['SALES_CHANNEL']=='TO BE DETERMINED']

	# Training/Testing set
	df_train = df_clean[df_merged['SALES_CHANNEL']!='TO BE DETERMINED']

	# Drop NAs from columns except target
	subset_cols = df_train.loc[:,df_train.columns != 'SELLPRICE_CAR'].columns.to_list()
	df_train = df_train.dropna(subset=subset_cols, axis=0)

	# perform feature selection
	df_train_red = _feature_selection(df_train, target, feature_selection_threshold, plot_features)

	# select the same columns for the target dataset
	df_target_red = df_target.loc[:, df_train_red.columns]
	
	# remove target variable
	df_target_red = df_target_red.drop([target], axis=1)

	return df_train_red, df_target_red, df_merged

def _preprocess_df_vehicles(df_vehicles: pd.DataFrame):

	# Applying uppercase to MAKE column
	df_vehicles['MAKE'] = df_vehicles['MAKE'].str.upper()

	# Replace "-" for " "
	df_vehicles['MAKE'] = df_vehicles['MAKE'].str.replace('-',' ')

	# remove punctuation signs
	df_vehicles['MAKE'] = df_vehicles['MAKE'].apply(lambda x: unidecode.unidecode(str(x)))

	return df_vehicles

def _preprocess_df_damage(df_damage: pd.DataFrame):

	# process damage dataset
	df_damage = df_damage.sort_values(['VEHICLEID','TMD_SRCE_ENTITY','ROWNUM'])
	df_damage_agg = df_damage.groupby(['VEHICLEID','TMD_SRCE_ENTITY'], as_index=False).agg({
		'DAMAGE': 'last',
		'ROWNUM': 'max',
		'CNT_DAMAGE_DISTINCT': 'last',
		'CNT_LARGE_DAMAGE': 'last',
		'DAMAGE_MAX': 'last'
	}).rename(columns={
		'DAMAGE': 'DAMAGE_CURRENT',
		'ROWNUM': 'DAMAGE_COUNT',
		'CNT_DAMAGE_DISTINCT': 'DAMAGE_CATEGORY',
		'CNT_LARGE_DAMAGE': 'DAMAGE_LARGE_CATEGORY'
	})

	return df_damage_agg

def _feature_selection(df, target, feature_selection_threshold, plot_features):

	# define target
	y = df[[target]]

	# remove target from training
	X = df.loc[:, ~df.columns.isin([target])]

	# fit tree based model
	logger.info("Number of initial features: %d" % X.shape[1])
	model = RandomForestRegressor()
	model.fit(X,  y)

	# get feature importance
	feat_importances = pd.Series(model.feature_importances_, index=X.columns)

	if plot_features:
		feat_importances.nlargest(20).plot(kind='bar')
		plt.title("Feature importance plot (Top 20 features)")
		plt.ylabel('Importance')
		plt.show()

	# reduced model
	model_reduced = SelectFromModel(model, threshold=feature_selection_threshold, prefit=True)
	X_reduced = model_reduced.transform(X)
	featured_num = X_reduced.shape[1]
	logger.info("Number of features selected: %d" % featured_num)
	
	# select most important features
	selected_features = feat_importances.nlargest(featured_num).index.to_list()
	logger.info("Selected feautres: %s" % selected_features)
	X = X.loc[:,selected_features]

	# add target variable
	df_reduced = pd.merge(X, y, left_index=True, right_index=True)

	return df_reduced






