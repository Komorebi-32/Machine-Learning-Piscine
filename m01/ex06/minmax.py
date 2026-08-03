import numpy as np

def minmax(x):
    """Computes the normalized version of a non-empty numpy.ndarray using the min-max standardization.
    Args:
    x: has to be an numpy.ndarray, a vector.
    Returns:
    x’ as a numpy.ndarray.
    None if x is an empty numpy.ndarray or not a numpy.ndarray.
    Raises:
    This function shouldn’t raise any Exception.
    """
    if not isinstance(x, np.ndarray):
        print("Error: x should be of type numpy.ndarray")
        return None
    if x.size == 0:
        print("Error: x should not be empty")
        return None
    x = (x - x.min()) / (x.max() - x.min())
    return x