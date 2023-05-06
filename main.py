import time
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

from CustomAi import LinearRegression

if __name__ == '__main__':
    data = pd.read_csv("./data/Fish.csv")
    data = data[data["Species"] == "Perch"]

    input_data = data["Length"].to_numpy()
    target_data = data["Weight"].to_numpy()

    model = LinearRegression()

    model.fit(input_data, target_data, 3000, 0.00001)

    x = np.arange(min(input_data), max(input_data))
    y = x * model.weights + model.bias

    plt.plot(x, y, "r--")
    plt.scatter(input_data, target_data)

    plt.show()
