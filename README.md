# Carnext Data Science Case

This project contains the code to answer the key question of the data science assignment.

## Description

The purpose of this project is to make predictions on the Sell Price of a subset of 150 vehicles and select the best 50 to sell in a physical store in the Netherlands.

## Getting Started

### Installing

Download or clone the project into your local machine, go to the project folder and run:
```bash
pip install -e .
```
### Executing program

1. Create training and target datasets to train a model and make predictions
```python
from carnext.dataset import get_datasets
df_train, df_target, df_merged = get_datasets()
```
2. Predict sell price of the vehicles contained in the target dataset
```python
from carnext.catboost_model import PredictSellPrice

# initialize class
model = PredictSellPrice()

# tune parameters and fit model
model.tune_fit(df_train)

# plot learning curves
model.plot_model_fit()

# make predictions
predictions = model.predict(df_target)

# plot predictions
model.plot_predictions()
```
3. Select optimal subset of cars
```python
from carnext.selection import select_cars, add_sell_price
selected_vehicles = select_cars(df_merged, N=50)
selected_vehicles = add_sell_price(selected_vehicles, df_predicted)
selected_vehicles.to_csv("selected_vehicles.csv")
```
## Authors

* Pau Vilar (pau.vilar.ribo@gmail.com)
