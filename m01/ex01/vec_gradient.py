import sys
sys.path.insert(0, "../../m00/ex03")

import numpy as np
from tools import add_intercept

def simple_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.arrays, without any for loop.
    The three arrays must have compatible shapes.
    Args:
    x: has to be a numpy.array, a vector of shape m * 1.
    y: has to be a numpy.array, a vector of shape m * 1.
    theta: has to be a numpy.array, a 2 * 1 vector.
    Return:
    The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
    None if x, y, or theta is an empty numpy.ndarray.
    None if x, y and theta do not have compatible dimensions.
    Raises:
    This function should not raise any Exception.
    """
    if (
        not isinstance(x, np.ndarray)
        or not isinstance(y, np.ndarray)
        or not isinstance(theta, np.ndarray)
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
    X = add_intercept(x)
    XT = np.transpose(X)
    J = (XT @ (X @ theta - y)) / y.shape[0]
    return J