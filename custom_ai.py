class LinearRegression:
    import numpy as np

    def __init__(self):
        print("Hello World")

    def predict(self):
        pass

    def fit(self):
        pass

    def evaluate(self):
        pass

class Util:
    def test_split_train(x, y, test_ratio, random_seed = None):
        import random

        if(x.shape[0] != y.shape[0]):
            raise Exception("Test_split_train")

        if(random_seed != None):
            random.seed(random_seed)