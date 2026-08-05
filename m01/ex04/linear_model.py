import sys
sys.path.insert(0, "../ex03")
sys.path.insert(0, "../../m00/ex05")
sys.path.insert(0, "../../m00/ex04")

import pandas as pandas
import numpy as np
from my_linear_regression import MyLinearRegression as MyLR
from plot import plot
import matplotlib.pyplot as plt

def main():
    data = pandas.read_csv("./are_blue_pills_magic.csv")
    Xpill = np.array(data['Micrograms']).reshape(-1, 1)
    Yscore = np.array(data['Score']).reshape(-1,1)

    linear_model1 = MyLR(np.array([[89.0], [-8]]))
    linear_model2 = MyLR(np.array([[89.0], [-6]]))
    
    new_theta1 = linear_model1.fit_(Xpill, Yscore)
    new_theta2 = linear_model2.fit_(Xpill, Yscore)
    Y_model1 = linear_model1.predict_(Xpill.reshape(-1,))
    Y_model2 = linear_model2.predict_(Xpill.reshape(-1,))
    plot(Xpill.reshape(-1,), Yscore.reshape(-1,), new_theta1)
    plot(Xpill.reshape(-1,), Yscore.reshape(-1,), new_theta2)
    print(f"loss function value model 1: {linear_model1.loss_(Yscore, Y_model1)}")
    print(f"loss function value model 2: {linear_model2.loss_(Yscore, Y_model2)}")

    # trying to plot the function J in function of the theta values
    plt.figure(figsize=(8, 5))
    i = 10
    linear_model = []
    while i > 0:
        linear_model[i] = MyLR(np.array([[i], [89]]))
        theta1 = np.linspace(-20, 20, 4000)
        J = np.empty(theta1.shape)
        for elem in J:
            elem = linear_model1[i].loss_(Yscore, Y_model1)

    
    plt.plot()

if __name__ == "__main__":
    main()