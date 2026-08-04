import sys
sys.path.insert(0, "../../m00/ex03")

import numpy as np
from tools import add_intercept

def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, without any for-loop.
    The three arrays must have the compatible dimensions.
    Args:
    x: has to be an numpy.array, a matrix of dimension m * n.
    y: has to be an numpy.array, a vector of dimension m * 1.
    theta: has to be an numpy.array, a vector (n +1) * 1.
    Return:
    The gradient as a numpy.array, a vector of dimensions n * 1,
    containg the result of the formula for all j.
    None if x, y, or theta are empty numpy.array.
    None if x, y and theta do not have compatible dimensions.
    None if x, y or theta is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    if (
        not isinstance(x, np.ndarray)
        or not isinstance(y, np.ndarray)
        or not isinstance(theta, np.ndarray)
    ):
        print("gradient() Error: x and/or y and/or theta should be of type numpy.ndarray")
        return None
    if x.size == 0 or y.size == 0 or theta.size == 0:
        print("gradient() Error: x and/or y and/or theta should not be empty")
        return None
    if (
        x.ndim != 2
        or y.ndim != 2
        or theta.ndim != 2
    ):
        print("gradient() Error: x, y and theta should be 2D arrays.")
        return None
    if (
        x.shape[0] != y.shape[0]
        or y.shape[1] != 1
        or x.shape[1] + 1 != theta.shape[0]
        or theta.shape[1] != 1
    ):
        print("gradient() Error: x and/or y and/or theta dimensions are not appropriate.")
        return None
    X = add_intercept(x)
    XT = np.transpose(X)
    J = (XT @ (X @ theta - y)) / y.shape[0]
    return J