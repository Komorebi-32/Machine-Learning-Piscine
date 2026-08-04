import sys
sys.path.insert(0, "../ex05")

import numpy as np
from my_multivar_linear_regression import MyLinearRegression as MyLR

def main():
    model = MyLR(np.array([[32], [3]]))
    