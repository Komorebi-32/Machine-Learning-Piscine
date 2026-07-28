from tools import add_intercept
import numpy as np

X = np.random.uniform(size=(10,1))
Xnew = add_intercept(X)
print(Xnew)