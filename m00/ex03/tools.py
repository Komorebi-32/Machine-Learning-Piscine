import numpy as np

def add_intercept(x):
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
    if not isinstance(x, np.ndarray):
            print("Error: x should be of type numpy.ndarray")
            return None
    if x.size == 0:
        print("Error: x should not be empty")
        return None
    if x.ndim > 2:
        print("Error: x dimensions are not appropriate.")
        return None