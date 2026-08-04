import sys
sys.path.insert(0, "../../m00/ex03")

import numpy as np
from tools import add_intercept

def predict_(x, theta):
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
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        print("simple_predict() Error: x and/or theta should be of type numpy.ndarray")
        return None
    if x.size == 0 or theta.size == 0:
        print("simple_predict() Error: x and/or theta should not be empty")
        return None
    if x.shape[1] + 1 != theta.shape[0] or theta.shape[1] != 1:
        print("simple_predict() Error: x or theta dimensions are not appropriate.")
        return None
    x = add_intercept(x)
    y_hat = x @ theta
    return y_hat