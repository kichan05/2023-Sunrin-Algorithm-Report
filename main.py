import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    data = pd.read_csv("./data/Fish.csv")
    data = data[data["Species"] == "Perch"]

    input_data = data["Length"].to_numpy().reshape(-1, 1)
    target_data = data["Weight"].to_numpy().reshape(-1, 1)

    plt.scatter(input_data, target_data)
    plt.show()

    # target_data_set = list(set(target_data))
    # target_data_set.sort()
    # target_index = {i : n for n, i in enumerate(target_data_set) }
    #
    # target_data[:] = np.vectorize(target_index.get)(target_data[:])



    # model = LinearRegression()
    # model.predict()