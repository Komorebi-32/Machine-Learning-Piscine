from tools import add_intercept
import numpy as np

X = np.random.uniform(size=(10,1))
print(X)
Xnew = add_intercept(X)
print(Xnew)

x = np.arange(1,6)
x_new = add_intercept(x)
print(x_new)