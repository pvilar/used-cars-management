# Used Cars Management

This project contains the solution to solve a business problem using data science.

## Business Statement

Which 50 cars from a selection of 150 cars should Company X select for its retail delivery store in Utrecht (city in the Netherlands) and what is the business impact?

## Business Context

The 3-4-year-old used car market is a large, growing €65bn market which is highly fragmented and ill-served, with low levels of transparency and consumer trust, making it ripe for disruption.
Company X aims to disrupt the European market for high-quality used cars. This new B2B and B2C digital marketplace will enable its customers to buy, lease, and subscribe to high-quality used cars in Europe.
For its customers, Company X is all about enjoying the ride: we offer the highest trust and convenience with fixed all-in prices, full maintenance history, a 14-days no question asked return guarantee and home delivery. Not all used cars qualify to be sold B2C at a Company X retail delivery store, for example, very specific vehicle types for which no customer demand exists, or cars that are relatively worn out. Moreover, delivery stores by definition are limited in physical space. Cars that are not sold B2C, will be sold in an auction (B2B).

## Data
[In progress]

## Approach

The objective of this project is to make predictions on the Sell Price of a subset of 150 vehicles and select the best 50 to sell in a physical store in the Netherlands.

The business problem is solved in a two step process:

1. **Predict** the sell price of the target 150 subset of cars. This will be done training a regression model on the training data using the relevant features of used cars. Initially there are 79 features that describe each used car, but only the 8 most relevant ones will be used for predicting the sell price. These features are:

['CATALOG_PRICE_CAR', 'TOTALMILEAGEVALUE', 'DAMAGE_CURRENT', 'ENGINECAPACITY', 'DAMAGE_MAX', 'MANUFACTUREYEAR', 'CATALOG_PRICE_ACCESSORIES', 'DAMAGE_CATEGORY']

[This table]('data/CASE_DAMAGE_LABELS_INFO.xlsx') contains a description for some of the features.

2. **Select** the optimal 50 cars from the subset of 150 that will be sold B2C based on demand and damage criteria:

- Filter only B2C historical vehicle sales in all countries
- Chose key variables for calculating a demand rank (MAKE, FUELTYPE, GEARTYPE, VEHICLEKIND) and a damage rank
- For each demand variables, compute the normalized frequency of sold vehicles among countries 
- Rank the frequency using a percentile form
- Select the 50 optimal vehicles based on the average of demand and damage rank criteria

## Code

The code to solve the problem is packaged in a package named fleetmanagement and below there is an explanation on how to use it.

### Installing the Package

Create a new virtual environment, download or clone the project into your local machine, go to the project folder and run:
```bash
pip install -e .
```
### Executing Program

1. Create training and target datasets to train a model and make predictions
```python
from fleetmanagement.dataset import get_datasets
df_train, df_target, df_merged = get_datasets()
```
2. Predict sell price of the vehicles contained in the target dataset
```python
from fleetmanagement.catboost_model import PredictSellPrice

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
from fleetmanagement.selection import select_cars, add_sell_price
selected_vehicles = select_cars(df_merged, N=50)
selected_vehicles = add_sell_price(selected_vehicles, df_predicted)
selected_vehicles.to_csv("selected_vehicles.csv")
```

## Author

* Pau Vilar (pau.vilar.ribo@gmail.com)
