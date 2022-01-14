import matplotlib as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("C:/Users/Marco/Desktop/R/RegressionModels/Used_fiat_500_in_Italy_dataset.csv")

x, y = data.loc[:, "age_in_days"], data.loc[:,"price"]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)
print(x_train.head())
print(x_train.shape)
print(x_test.head())
print(x_test.shape)
print(y_train.head())
print(y_train.shape)
print(y_test.head())
print(y_test.shape)

regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)

regr_y_pred = regr.predict(x_test)
print("coefficients: \n", regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, regr_y_pred))
print("Coefficient of determination: %.2f" % r2_score(y_test, regr_y_pred))