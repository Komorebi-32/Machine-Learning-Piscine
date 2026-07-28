import numpy as np

def loss_elem_(y, y_hat):
    """
    Description:
    Calculates all the elements (y_pred - y)^2 of the loss function.
    Args:
    y: has to be an numpy.array, a two-dimensional array of shape m * 1.
    y_hat: has to be an numpy.array, a two-dimensional array of shape m * 1.
    Returns:
    J_elem: numpy.array, a array of dimension (number of the training examples, 1).
    None if there is a dimension matching problem.
    None if any argument is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        print("Error: x and/or y_hat should be of type numpy.ndarray")
        return None
    if y.size == 0 or y_hat.size == 0:
        print("Error: x and/or y_hat should not be empty")
        return None
    if y.ndim != 2 or y_hat.ndim != 2 or y.shape != y_hat.shape or y.shape[1] != 1:
        print("Error: x or y_hat dimensions are not appropriate.")
        return None
    J_elem = (y_hat - y) ** 2
    return J_elem

def loss_(y, y_hat):
    """
    Description:
    Calculates the value of loss function.
    Args:
    y: has to be an numpy.array, a two-dimensional array of shape m * 1.
    y_hat: has to be an numpy.array, a two-dimensional array of shape m * 1.
    Returns:
    J_value : has to be a float.
    None if there is a dimension matching problem.
    None if any argument is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    elem = loss_elem_(y, y_hat)
    if elem is None:
        return None
    J_value = np.sum(elem) / (2 * y.shape[0])
    return J_value