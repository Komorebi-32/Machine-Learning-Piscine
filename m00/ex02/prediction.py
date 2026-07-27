import numpy as np

def simple_predict(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
    x: has to be an numpy.ndarray, a one-dimensional array of size m.
    theta: has to be an numpy.ndarray, a one-dimensional array of size 2.
    Returns:
    y_hat as a numpy.ndarray, a one-dimensional array of size m.
    None if x or theta are empty numpy.ndarray.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        print("Error: x and/or theta should be of type numpy.ndarray")
        return None
    if x.size == 0 or theta.size == 0:
        print("Error: x and/or theta should not be empty")
        return None
    # if x.shape != (len(x), ): would work for most cases but Exception with 0-dimensional
    # arrays
    if x.ndim != 1 or theta.ndim != 1 or theta.shape != (2, ):
        print("Error: x or theta dimensions are not appropriate.")
        return None
    y_hat = theta[0] + theta[1] * x
    return y_hat
