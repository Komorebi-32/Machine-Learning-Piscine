import sys
sys.path.insert(0, "../../m00/ex04")
sys.path.insert(0, "../../m00/ex03")

import numpy as np
from prediction import predict_

def simple_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.arrays, with a for-loop.
    The three arrays must have compatible shapes.
    Args:
    x: has to be an numpy.array, a vector of shape m * 1.
    y: has to be an numpy.array, a vector of shape m * 1.
    theta: has to be an numpy.array, a 2 * 1 vector.
    Return:
    The gradient as a numpy.array, a vector of shape 2 * 1.
    None if x, y, or theta are empty numpy.array.
    None if x, y and theta do not have compatible shapes.
    None if x, y or theta is not of the expected type.
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
    x = x.reshape(-1)
    h = predict_(x, theta)
    if h is None:
        return None
    J0 = 0
    J1 = 0
    for hi, yi, xi in zip(h.flatten(), y.flatten(), x):
        J0 += hi - yi
        J1 += (hi - yi) * xi
    J0 /= y.shape[0]
    J1 /= y.shape[0]
    J = np.array([[J0], [J1]])
    return J