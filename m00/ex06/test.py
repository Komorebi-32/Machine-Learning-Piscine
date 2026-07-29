import numpy as np
from loss import loss_elem_
from loss import loss_
from prediction import predict_

x1 = np.array([0., 1., 2., 3., 4.])
print(f"x1: {x1}")
theta1 = np.array([[2.], [4.]])
print(f"theta1: {theta1}")
print("y_hat1 predict_")
y_hat1 = predict_(x1, theta1)
y1 = np.array([[2.], [7.], [12.], [17.], [22.]])
print("Example 1:")
print("Expected result: array([[0.], [1], [4], [9], [16]])")
print(f"Got: {loss_elem_(y1, y_hat1)}")

# Output:
# array([[0.], [1], [4], [9], [16]])

# Example 2:
print("Example 2:")
print("Expected result: 3.0")
print(f"Got: {loss_(y1, y_hat1)}")

# Output:
# 3.0

# x2 = np.array([0, 15, -9, 7, 12, 3, -21]).reshape(-1, 1)
x2 = np.array([0, 15, -9, 7, 12, 3, -21])
print(f"x2: {x2}")
theta2 = np.array([[0.], [1.]])
print(f"theta2: {theta2}")
y_hat2 = predict_(x2, theta2)
y2 = np.array([2, 14, -13, 5, 12, 4, -19]).reshape(-1, 1)
# y2 = np.array([2, 14, -13, 5, 12, 4, -19])

# Example 3:
print("Example 3:")
print("Expected result: 2.142857142857143")
print(f"Got: {loss_(y2, y_hat2)}")

# Output:
# 2.142857142857143

# Example 4:
print("Example 4:")
print("Expected result: 0.0")
print(f"Got: {loss_(y2, y2)}")
# Output:
# 0.0