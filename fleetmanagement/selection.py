""" Selection module """

import logging
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import os

logger = logging.getLogger("fleetmanagement")

def select_cars(
	df_merged: pd.DataFrame,
	demand_vars: list = [
					'MAKE',
					'FUELTYPE',
					'GEARTYPE',
					'VEHICLEKIND'],
	damage_vars: list = [
					'DAMAGE_CURRENT',
					'DAMAGE_MAX',
					'DAMAGE_COUNT',
					'DAMAGE_CATEGORY',
					'DAMAGE_LARGE_CATEGORY'],
	N: int = 50,
	file_name: str = 'vehicles_selection',
	file_path: str = ''
):

	"""
	Function that calculates a rank index based on demand and damage 
	indicators in order to select the optimal N number of vehicles
	for the store in the Netherlads 

	Parameters
	----------
	df_merged : raw dataset that contains vehicle and damage information
	demand_vars : list of variables to use in the calculation of the demand rank index
	damage_vars : list of variables to use in the calculation of the damage rank index
	N : Number of vehicles to select
	file_name : name of the csv file where list of selected vehicles will be stored
	file_path : path to the folder where so export the results
	
	Returns
	----------
	df_selection : dataset with the N selected vehicles

	"""
	logger.info("Selecting optimal %d cars to sell BSC" % N)

	# split datasets
	df_b2c = df_merged[df_merged['SALES_CHANNEL']=='B2C'].reset_index()
	df_target = df_merged[df_merged['SALES_CHANNEL']=='TO BE DETERMINED'].reset_index()

	# calculate demand rank
	df_target['DEMAND_RANK'] = _demand_rank(df_b2c, df_target, demand_vars)

	# calculate damage rank
	df_target['DAMAGE_RANK'] = _damage_rank(df_target, damage_vars)

	# calculate final rank from average between demand and damage ranks
	df_target['RANK'] = df_target[['DEMAND_RANK','DAMAGE_RANK']]\
									.mean(axis=1).rank(pct=True)

	# sort by rank and get first N rows
	df_selection = df_target.sort_values('RANK', ascending=False).head(N)

	# export as csv
	output_file = os.path.join(file_path, file_name+'.csv')
	df_selection.to_csv(output_file)

	return df_selection

def add_sell_price(df_selection, df_prediction):

	df_selection = pd.merge(df_selection.drop('SELLPRICE_CAR', axis=1),
                         df_prediction[['SELLPRICE_CAR']].reset_index(), how='left')
	
	return df_selection

def _demand_rank(df_b2c, df_target, demand_vars):

	df_demand = df_target[demand_vars]
	for v in demand_vars:

		# crosstabs comparing sold cars among countries
		CT = pd.crosstab(df_b2c[v],
				df_b2c['TMD_SRCE_ENTITY'],
				normalize='index').sort_values('LPNL', ascending=False)
		
		df = CT[['LPNL']].reset_index()
		
		df['LPNL'].rank(pct=True)
		
		df.rename(columns={'LPNL': v+'_RANK'}, inplace=True)
		
		df_demand = df_demand.merge(df, how='left')

	demand_rank = df_demand.mean(axis=1).rank(pct=True)

	return demand_rank

def _damage_rank(df_target, damage_vars):

	df_damage = df_target[damage_vars]
	for v in damage_vars:

		df_damage[v+"_RANK"] = df_damage[v].rank(ascending=False, pct=True)
	   
	damage_rank = df_damage.drop(damage_vars, axis=1)\
									.apply(lambda x: x.mean(), axis=1)\
									.rank(pct=True)

	return damage_rank


