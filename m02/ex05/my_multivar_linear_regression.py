import sys
sys.path.insert(0, "../ex03")
sys.path.insert(0, "../../m00/ex03")


import numpy as np
import matplotlib.pyplot as plt
from gradient_multivar import gradient
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
        x: has to be a numpy.array, a matrix of dimension m * n:
        (number of training examples, number of features).
        y: has to be a numpy.array, a vector of dimension m * 1:
        (number of training examples, 1).
        theta: has to be a numpy.array, a vector of dimension (n + 1) * 1:
        (number of features + 1, 1).
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
        Return:
        new_theta: numpy.array, a vector of dimension (number of features + 1, 1).
        None if there is a matching dimension problem.
        None if x, y, theta, alpha or max_iter is not of expected type.
        Raises:
        This function should not raise any Exception.
        """
        if (
            not isinstance(x, np.ndarray)
            or not isinstance(y, np.ndarray)
            or not isinstance(self.thetas, np.ndarray)
            or not isinstance(self.alpha, float)
            or not isinstance(self.max_iter, int)
        ):
            print("fit() Error: x and/or y and/or theta should be of type numpy.ndarray")
            return None
        if x.size == 0 or y.size == 0 or self.thetas.size == 0:
            print("fit() Error: x and/or y and/or theta should not be empty")
            return None
        if (
            x.ndim != 2
            or y.ndim != 2
            or self.thetas.ndim != 2
        ):
            print("fit() Error: x, y and theta should be 2D arrays.")
            return None
        if (
            x.shape[0] != y.shape[0]
            or y.shape[1] != 1
            or x.shape[1] + 1 != self.thetas.shape[0]
            or self.thetas.shape[1] != 1
        ):
            print("fit() Error: x and/or y and/or theta dimensions are not appropriate.")
            return None
        if self.alpha < 0 or self.alpha > 1:
            print("fit() Error: alpha should be between 0 and 1")
            return None
        if self.max_iter <= 0:
            print("fit() Error: max_iter should be greater than 0")
            return None
        iter = 0
        self.thetas = self.thetas.astype(float)
        while iter != self.max_iter:
            derivative = gradient(x, y, self.thetas)
            self.thetas = self.thetas - (self.alpha * derivative)
            iter += 1
        return self.thetas

    def predict_(self, x):
        """Computes the prediction vector y_hat from two non-empty numpy.array.
        Args:
        x: has to be an numpy.array, a vector of dimensions m * n.
        theta: has to be an numpy.array, a vector of dimensions (n + 1) * 1.
        Return:
        y_hat as a numpy.array, a vector of dimensions m * 1.
        None if x or theta are empty numpy.array.
        None if x or theta dimensions are not appropriate.
        None if x or theta is not of expected type.
        Raises:
        This function should not raise any Exception.
        """
        if not isinstance(x, np.ndarray) or not isinstance(self.thetas, np.ndarray):
            print("predict_() Error: x and/or theta should be of type numpy.ndarray")
            return None
        if x.size == 0 or self.thetas.size == 0:
            print("predict_() Error: x and/or theta should not be empty")
            return None
        if x.ndim == 1 and self.thetas.shape != (2, 1):
            print("predict_ Error: x or theta dimensions are not appropriate.")
            return None
        if x.ndim > 1 and (x.shape[1] + 1 != self.thetas.shape[0] or self.thetas.shape[1] != 1):
            print("predict_() Error: x or theta dimensions are not appropriate.")
            return None
        x = add_intercept(x)
        y_hat = x @ self.thetas
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
            print("loss_elem_ Error: y and/or y_hat should be of type numpy.ndarray")
            return None
        if y.size == 0 or y_hat.size == 0:
            print("loss_elem_ Error: y and/or y_hat should not be empty")
            return None
        if y.ndim != 2 or y_hat.ndim != 2 or y.shape != y_hat.shape or y.shape[1] != 1:
            print("loss_elem_ Error: y and/or y_hat dimensions are not appropriate.")
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

    def plot_scatter(self, x, y, xlabel, ylabel):
        """Plot the data and prediction line from three non-empty numpy.array.
        Args:
        x: has to be an numpy.array, a one-dimensional array of size m.
        y: has to be an numpy.array, a one-dimensional array of size m.
        theta: has to be an numpy.array, a two-dimensional array of shape 2 * 1.
        Returns:
        Nothing.
        Raises:
        This function should not raise any Exceptions.
        """
        if (
            not isinstance(x, np.ndarray)
            or not isinstance(self.thetas, np.ndarray)
            or not isinstance(y, np.ndarray)
            or not isinstance(xlabel, str)
            or not isinstance(ylabel, str)
        ):
            print("Error: x and/or y and/or theta should be of type numpy.ndarray")
            print("xlabel and ylabel should be of type string")
            return None
        if x.size == 0 or y.size == 0 or self.thetas.size == 0:
            print("Error: x and/or y and/or theta should not be empty")
            return None
        if x.ndim != 1 or y.ndim != 1 or self.thetas.shape != (2, 1):
            print("plot: Error: x and/or y and/or theta dimensions are not appropriate.")
            return None

        # Generate prediction values for the regression line
        y_pred = self.predict_(x)

        # Create the visualization
        plt.figure(figsize=(8, 5))

        # Plot matrix data points
        plt.scatter(x, y, color='blue', label='Matrix Data Points', zorder=5)

        # Plot predicted points
        plt.scatter(x, y_pred, color='red', zorder=5, 
                label='Regression Line')

        # Customize the chart aesthetics
        title = "Prediction of " + ylabel + " according to " + xlabel
        title += ". Loss function value: " + str(self.loss_(y.reshape(-1, 1), y_pred))
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        # Display the plot
        plt.show()