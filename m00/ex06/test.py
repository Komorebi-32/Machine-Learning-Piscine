import sys
sys.path.insert(0, "../ex04")

import numpy as np
from loss import loss_elem_
from loss import loss_
from prediction import predict_

x1 = np.array([0., 1., 2., 3., 4.])
theta1 = np.array([[2.], [4.]])
print("y_hat1 predict_")
y_hat1 = predict_(x1, theta1)
y1 = np.array([[2.], [7.], [12.], [17.], [22.]])
# Example 1:
loss_elem_(y1, y_hat1)
# Output:
# array([[0.], [1], [4], [9], [16]])

# Example 2:
loss_(y1, y_hat1)
# Output:
# 3.0

x2 = np.array([0, 15, -9, 7, 12, 3, -21]).reshape(-1, 1)
theta2 = np.array(np.array([[0.], [1.]]))
y_hat2 = predict_(x2, theta2)
y2 = np.array([2, 14, -13, 5, 12, 4, -19]).reshape(-1, 1)

# Example 3:
loss_(y2, y_hat2)
# Output:
# 2.142857142857143

# Example 4:
loss_(y2, y2)
# Output:
# 0.0