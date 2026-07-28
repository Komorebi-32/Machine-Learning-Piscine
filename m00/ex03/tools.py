import numpy as np

def add_intercept(X):
    """Adds a column of 1’s to the non-empty numpy.array x.
    Args:
    x: has to be a numpy.array. x can be a one-dimensional (m * 1) or two-dimensional (m * n) array.
    Returns:
    X, a numpy.array of dimension m * (n + 1).
    None if x is not a numpy.array.
    None if x is an empty numpy.array.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(X, np.ndarray):
        print("Error: X should be of type numpy.ndarray")
        return None
    if X.size == 0:
        print("Error: X should not be empty")
        return None
    if X.ndim > 2:
        print("Error: X dimensions are not appropriate.")
        return None
    n,m = X.shape
    X0 = np.ones((n,1))
    Xnew = np.hstack((X0, X))
    return Xnew