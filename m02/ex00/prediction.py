import numpy as np

def simple_predict(x, theta):
    """Computes the prediction vector y_hat from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a matrix of dimension m * n.
    theta: has to be an numpy.array, a vector of dimension (n + 1) * 1.
    Return:
    y_hat as a numpy.array, a vector of dimension m * 1.
    None if x or theta are empty numpy.array.
    None if x or theta dimensions are not matching.
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
    y_hat = np.empty((x.shape[0], 1))
    y_hat = x.dot(theta[1:]) + theta[0]
    y_hat = y_hat.reshape(-1, 1) # transform (m, ) into (m, 1)
    return y_hat