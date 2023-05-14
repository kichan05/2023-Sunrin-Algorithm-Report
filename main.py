import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import cv2
import os

from CustomAi import LinearRegression


def simulation():
    model = LinearRegression()
    epoch = 5000
    parameters = np.array([10, 10], dtype="float64")

    for i in range(epoch):
        slops = model.get_slope(input_data, target_data, parameters)
        parameters -= slops * np.array([0.002, 0.02])

        weight = parameters[0]
        bias = parameters[1]

        loss = model.get_loss(input_data, target_data, parameters)

        x = np.arange(0, 50, 0.1)
        y = x * weight + bias

        plt.xlim(0, 50)
        plt.ylim(0, 1500)


        if(bias >= 0):
            graph = f"y = {round(weight, 2)} * x + {round(bias, 2)} (loss : {round(loss, 2)}, epoch : {i})"
        else:
            graph = f"y = {round(weight, 2)} * x {round(bias, 2)} (loss : {round(loss, 2)}, epoch : {i})"

        if(i % 3 == 0):
            plt.text(3, 1400, graph, color = "red")
            plt.scatter(input_data, target_data)
            plt.plot(x, y, "r--")

            plt.savefig(f"./image/{i}.png")
            plt.cla()

            image = cv2.imread(f"./image/{i}.png")
            cv2.imshow("Image", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.waitKey(0)


def train_data():
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

if __name__ == '__main__':
    data = pd.read_csv("./data/Fish.csv")
    data = data[data["Species"] == "Perch"]

    input_data = data["Length"].to_numpy().reshape(-1, 1)
    target_data = data["Weight"].to_numpy().reshape(-1, 1)


    # simulation() # 학습 과정 시뮬레이션
    train_data() # 모델 학습