import sys
sys.path.insert(0, "../ex03")

import numpy as np
from tools import add_intercept 

def predict_(x, theta):
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
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        print("Error32: x and/or theta should be of type numpy.ndarray")
        return None
    if x.size == 0 or theta.size == 0:
        print("Error: x and/or theta should not be empty")
        return None
    if x.ndim != 1 or theta.shape != (2, 1):
        print("Error: x or theta dimensions are not appropriate.")
        return None
    X = add_intercept(x)
    y_hat = np.dot(X, theta)
    return y_hat