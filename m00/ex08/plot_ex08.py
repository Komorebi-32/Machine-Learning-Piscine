import sys
sys.path.insert(0, "../ex04")

import numpy as np
import matplotlib.pyplot as plt
from prediction import predict_
from plot import plot

def plot_with_loss(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.ndarray.
    Args:
    x: has to be an numpy.ndarray, one-dimensional array of size m.
    y: has to be an numpy.ndarray, one-dimensional array of size m.
    theta: has to be an numpy.ndarray, one-dimensional array of size 2.
    Returns:
    Nothing.
    Raises:
    This function should not raise any Exception.
    """
    if (
        not isinstance(x, np.ndarray)
        or not isinstance(theta, np.ndarray)
        or not isinstance(y, np.ndarray)
    ):
        print("Error: x and/or theta should be of type numpy.ndarray")
        return None
    if x.size == 0 or y.size == 0 or theta.size == 0:
        print("Error: x and/or theta should not be empty")
        return None
    if x.ndim != 1 or y.ndim != 1 or theta.ndim != 1 or theta.size != 2:
        print("Error: x or theta dimensions are not appropriate.")
        return None
    
    # transform theta into a 2D array of shape 2 * 1
    theta = theta.reshape(-1, 1)

    plot(x, y, theta)

    y_hat = predict_(x, theta)
    y_hat = y_hat.reshape(-1) # reshape(-1) to convert y_hat from (m, 1) to (m,)
    # so that plt.plot() can work well. Entering the for makes browse through scalars
    # and not single element arrays
    for xi, yi, ypi in zip(x, y, y_hat):
        plt.plot([xi, xi], [yi, ypi], color='pink', linewidth=3, linestyle='dotted')

    plt.show()