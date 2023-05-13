import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import cv2

from CustomAi import LinearRegression

if __name__ == '__main__':
    data = pd.read_csv("./data/Fish.csv")
    data = data[data["Species"] == "Perch"]

    input_data = data["Length"].to_numpy().reshape(-1, 1)
    target_data = data["Weight"].to_numpy().reshape(-1, 1)

    model = LinearRegression()
    model.fit(input_data, target_data, epoch=3000)
    loss = model.get_loss(input_data, target_data)

    plt.xlim(0, 50)
    plt.ylim(0, 1500)

    weight = model.parameters[0]
    bias = model.parameters[1]

    x = np.arange(0, 50)
    y = x * weight + bias

    plt.scatter(input_data, target_data)
    plt.plot(x, y, "r--")
    plt.text(2, 1400, f"y = {round(weight, 2)} * X + {round(bias, 2)}, (loss : {round(loss, 2)})", color="red")
    plt.show()
