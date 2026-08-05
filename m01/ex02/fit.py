import sys
sys.path.insert(0, "../ex01")

import numpy as np
from vec_gradient import simple_gradient

def fit_(x, y, theta, alpha: float, max_iter):
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
        or not isinstance(theta, np.ndarray)
        or not isinstance(alpha, float)
        or not isinstance(max_iter, int)
    ):
        print("Error: x and/or y and/or theta should be of type numpy.ndarray")
        return None
    if x.size == 0 or y.size == 0 or theta.size == 0:
        print("Error: x and/or y and/or theta should not be empty")
        return None
    if (
        x.ndim != 2 or y.ndim != 2
        or x.shape != y.shape or y.shape[1] != 1
        or theta.shape != (2, 1)
    ):
        print("Error: x and/or y and/or theta dimensions are not appropriate.")
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
        derivative = simple_gradient(x, y, theta)
        theta = theta - (alpha * derivative)
        iter += 1
    return theta
    