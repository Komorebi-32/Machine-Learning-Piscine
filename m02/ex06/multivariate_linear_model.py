import sys
sys.path.insert(0, "../ex05")

import numpy as np
import pandas as pandas
import matplotlib.pyplot as plt
from my_multivar_linear_regression import MyLinearRegression as MyLR

def univariate_linear_regression(feature):
    if not isinstance(feature, str):
        print("Error: feature input arg should be a string")
    x = np.array(data[feature]).reshape(-1, 1)
    y = np.array(data['Sell_price']).reshape(-1, 1)

    model = MyLR(np.array([[1000], [-3]]), alpha = 2.5e-5, max_iter = 500000)
    model.fit_(x, y)
    model.plot_scatter(x.reshape(-1,), y.reshape(-1,), feature, "Sell price")

def multivariate_linear_regression():
    x = np.array(data[['Age','Thrust_power','Terameters']])
    y = np.array(data[['Sell_price']])
    model = MyLR(np.array([1.0, 1.0, 1.0, 1.0]).reshape(-1, 1), alpha=9e-5, max_iter=500000)
    model.fit_(x, y)
    print(f"theta: {model.thetas}")
    y_pred = model.predict_(x)
    loss = model.loss_(y, y_pred)
    print(f"loss: {loss}")
    model.plot_scatter(x[:, 0], y.reshape(-1, ), "Age", "Sell price")

if __name__ == "__main__":
    data = pandas.read_csv("./spacecraft_data.csv")
    univariate_linear_regression("Age")
    # univariate_linear_regression("Thrust_power")
    # univariate_linear_regression("Terameters")
    # multivariate_linear_regression()