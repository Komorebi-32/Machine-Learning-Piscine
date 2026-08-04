import sys
sys.path.insert(0, "../ex05")

import numpy as np
import pandas as pandas
import matplotlib.pyplot as plt
from my_multivar_linear_regression import MyLinearRegression as MyLR

def univariate_linear_regression(feature):
    if not isinstance(feature, str):
        print("Error: feature input arg should be a string")
    data = pandas.read_csv("./spacecraft_data.csv")
    x = np.array(data[feature]).reshape(-1, 1)
    y = np.array(data['Sell_price']).reshape(-1, 1)

    model = MyLR(np.array([[32], [3]]))
    model.fit_(x, y)
    model.plot_scatter(x.reshape(-1,), y.reshape(-1,), feature, "Sell price")

if __name__ == "__main__":
    univariate_linear_regression("Age")