import pandas as pd
from CustomAi import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

from CustomAi import LinearRegression

if __name__ == '__main__':
    data = pd.read_csv("./data/Fish.csv")
    data = data[data["Species"] == "Perch"]

    input_data = data["Length"].to_numpy()
    target_data = data["Weight"].to_numpy()

    # plt.scatter(input_data, target_data)
    # plt.show()



    model = LinearRegression()

    cost = []
    weight = np.arange(-100, 100, 0.1)

    cost_min = model.get_cost(input_data, target_data, 0, 0)
    cost_min_weight = 0

    for i in weight:
        current_cost = model.get_cost(input_data, target_data, i, 0)
        cost.append(current_cost)

        if(cost_min > current_cost):
            cost_min = current_cost
            cost_min_weight = i

    plt.plot(weight, cost)
    plt.scatter([cost_min_weight], [cost_min])
    plt.xlabel("Weidht")
    plt.ylabel("Cost")
    plt.savefig("1.png")
    plt.show()
