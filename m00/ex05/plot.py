import sys
sys.path.insert(0, "../ex04")

import numpy as np
import matplotlib.pyplot as plt
from prediction import predict_

def plot(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a one-dimensional array of size m.
    y: has to be an numpy.array, a one-dimensional array of size m.
    theta: has to be an numpy.array, a two-dimensional array of shape 2 * 1.
    Returns:
    Nothing.
    Raises:
    This function should not raise any Exceptions.
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
    if x.ndim != 1 or y.ndim != 1 or theta.shape != (2, 1):
        print("Error: x or theta dimensions are not appropriate.")
        return None

    # Generate prediction values for the regression line
    y_pred = predict_(x, theta)

    # Create the visualization
    plt.figure(figsize=(8, 5))

    # Plot matrix data points
    plt.scatter(x, y, color='blue', label='Matrix Data Points', zorder=5)

    # Plot the regression line
    plt.plot(x, y_pred, color='red', linewidth=2, 
            label='Regression Line')

    # Customize the chart aesthetics
    plt.title('NumPy Matrix Data with Linear Regression Line')
    plt.xlabel('X (Independent Variable)')
    plt.ylabel('Y (Dependent Variable)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Display the plot
    plt.show()