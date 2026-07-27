from prediction import simple_predict
import numpy as np

x = np.array([1, 2, 3, 4])
theta = np.array([32, 1])
y_hat = simple_predict(x, theta)
print(y_hat)

simple_predict("wrong", theta)

wrong_theta = np.array([[32, 1]])
simple_predict(x, wrong_theta)
