import sys
sys.path.insert(0, "../ex01")
sys.path.insert(0, "../../m00/ex03")

import numpy as np
from vec_gradient import simple_gradient
from tools import add_intercept

class MyLinearRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    fit_(self, x, y),
    • predict_(self, x),
    • loss_elem_(self, y, y_hat),
    • loss_(self, y, y_hat).
    """
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

    def fit_(self, x, y):
        """
        Description:
        Fits the model to the training dataset contained in x and y.
        Args:
        x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
        Returns:
        new_theta: numpy.ndarray, a vector of dimension 2 * 1.
        None if there is a matching dimension problem.
        Raises:
        This function should not raise any Exception.
        """
        if (
            not isinstance(x, np.ndarray)
            or not isinstance(y, np.ndarray)
            or not isinstance(self.theta, np.ndarray)
            or not isinstance(self.alpha, float)
            or not isinstance(self.max_iter, int)
        ):
            print("Error: x and/or y and/or theta should be of type numpy.ndarray")
            return None
        if x.size == 0 or y.size == 0 or self.theta.size == 0:
            print("Error: x and/or y and/or theta should not be empty")
            return None
        if (
            x.ndim != 2 or y.ndim != 2
            or x.shape != y.shape or y.shape[1] != 1
            or self.theta.shape != (2, 1)
        ):
            print("Error: x and/or y and/or theta dimensions are not appropriate.")
            return None
        if self.alpha < 0 or self.alpha > 1:
            print("Error: alpha should be between 0 and 1")
            return None
        if self.max_iter <= 0:
            print("Error: max_iter should be greater than 0")
            return None
        iter = 0
        self.theta = self.theta.astype(float)
        while iter != self.max_iter:
            derivative = simple_gradient(x, y, self.theta)
            self.theta = self.theta - (self.alpha * derivative)
            iter += 1
        return self.theta

    def predict_(self, x):
        """Computes the vector of prediction y_hat from two non-empty numpy.array.
        Args:
        x: has to be an numpy.array, a one-dimensional array of size m.
        theta: has to be an numpy.array, a two-dimensional array of shape 2 * 1.
        Returns:
        y_hat as a numpy.array, a two-dimensional array of shape m * 1.
        None if x and/or theta are not numpy.array.
        None if x or theta are empty numpy.array.
        None if x or theta dimensions are not appropriate.
        Raises:
        This function should not raise any Exceptions.
        """
        if not isinstance(x, np.ndarray) or not isinstance(self.theta, np.ndarray):
            print("Error: x and/or theta should be of type numpy.ndarray")
            return None
        if x.size == 0 or self.theta.size == 0:
            print("Error: x and/or theta should not be empty")
            return None
        if x.ndim != 1 or self.theta.shape != (2, 1):
            print("Error: x or theta dimensions are not appropriate.")
            return None
        X = add_intercept(x)
        y_hat = np.dot(X, self.theta)
        return y_hat


    def loss_elem_(self, y, y_hat):
        """
        Description:
        Calculates all the elements (y_pred - y)^2 of the loss function.
        Args:
        y: has to be an numpy.array, a two-dimensional array of shape m * 1.
        y_hat: has to be an numpy.array, a two-dimensional array of shape m * 1.
        Returns:
        J_elem: numpy.array, a array of dimension (number of the training examples, 1).
        None if there is a dimension matching problem.
        None if any argument is not of the expected type.
        Raises:
        This function should not raise any Exception.
        """
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
            print("Error: y and/or y_hat should be of type numpy.ndarray")
            return None
        if y.size == 0 or y_hat.size == 0:
            print("Error: y and/or y_hat should not be empty")
            return None
        if y.ndim != 2 or y_hat.ndim != 2 or y.shape != y_hat.shape or y.shape[1] != 1:
            print("Error: y and/or y_hat dimensions are not appropriate.")
            return None
        J_elem = (y_hat - y) ** 2
        return J_elem

    def loss_(self, y, y_hat):
        """
        Description:
        Calculates the value of loss function.
        Args:
        y: has to be an numpy.array, a two-dimensional array of shape m * 1.
        y_hat: has to be an numpy.array, a two-dimensional array of shape m * 1.
        Returns:
        J_value : has to be a float.
        None if there is a dimension matching problem.
        None if any argument is not of the expected type.
        Raises:
        This function should not raise any Exception.
        """
        elem = self.loss_elem_(y, y_hat)
        if elem is None:
            return None
        J_value = np.sum(elem) / (2 * y.shape[0])
        return J_value