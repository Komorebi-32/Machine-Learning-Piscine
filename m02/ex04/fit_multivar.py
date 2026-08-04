import sys
sys.path.insert(0, "../ex03")

import numpy as np
from gradient_multivar import gradient

def fit_(x, y, theta, alpha, max_iter):
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
        or not isinstance(theta, np.ndarray)
    ):
        print("fit() Error: x and/or y and/or theta should be of type numpy.ndarray")
        return None
    if x.size == 0 or y.size == 0 or theta.size == 0:
        print("fit() Error: x and/or y and/or theta should not be empty")
        return None
    if (
        x.ndim != 2
        or y.ndim != 2
        or theta.ndim != 2
    ):
        print("fit() Error: x, y and theta should be 2D arrays.")
        return None
    if (
        x.shape[0] != y.shape[0]
        or y.shape[1] != 1
        or x.shape[1] + 1 != theta.shape[0]
        or theta.shape[1] != 1
    ):
        print("fit() Error: x and/or y and/or theta dimensions are not appropriate.")
        return None
    if alpha < 0 or alpha > 1:
        print("Error: alpha should be between 0 and 1")
        return None
    if max_iter <= 0:
        print("Error: max_iter should be greater than 0")
        return None
    iter = 0
    theta = theta.astype(float)
    while iter != max_iter:
        derivative = gradient(x, y, theta)
        theta = theta - (alpha * derivative)
        iter += 1
    return theta