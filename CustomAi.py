import numpy as np

class LinearRegression:
    def __init__(self):
        self.parameters = np.array([10, 10], dtype="float64")

    def get_loss(self, x, y, weight = None):
        """
        평균 제곱 오차 방식을 사용해서 오차를 구한다.
        :param x:
        :param y:
        :param weight:
        :param bias:
        :return:
        """

        predict = self.predict(x, weight)
        loss = np.sum((predict - y) ** 2) / len(y) / 2
        return loss

    def get_slope(self, x, y, parameter, d_x=0.0001):
        """
        측정 가중치에서의 오차 함수의 경사를 구한다.
        :param x:
        :param y:
        :param weight:
        :param bias:
        :param d_x:
        :return:
        """

        glops = np.zeros_like(parameter)

        for index, weight in enumerate(parameter):
            parameter[index] = weight + d_x
            y1 = self.get_loss(x, y, parameter)

            parameter[index] = weight - d_x
            y2 = self.get_loss(x, y, parameter)

            parameter[index] = weight
            glops[index] = (y1 - y2) / d_x / 2

        return glops

    def predict(self, x, parameters = None):
        """
        :param x:
        :param parameters:
        :return:
        """

        if(parameters is None):
            weights = self.parameters[:-1]
            bias = self.parameters[-1]
        else:
            weights = parameters[:-1]
            bias = parameters[-1]


        return x * weights + bias

    def fit(self, x, y, epoch=300, learning_rate=np.array([0.001, 0.01])):
        """
        학습을 진행하는 함수
        :param x:
        :param y:
        :param epoch:
        :param learning_rate:
        :return:
        """

        def print_loading_bar(current, mx):
            percent = round(current / mx * 100, 2)
            percent_10 = int(percent) // 10
            print(f"{'■' * percent_10}{'□' * (10 - percent_10)} {current}/{epoch}({percent}%%)")

        for i in range(epoch):
            slop = self.get_slope(x, y, self.parameters)
            self.parameters -= slop * learning_rate

            print_loading_bar(i + 1, epoch)

    def evaluate(self):
        # Todo("평가 함수 작성")
        pass


class Util:
    def test_split_train(x, y, test_ratio, random_seed=None):
        import random

        if (x.shape[0] != y.shape[0]):
            raise Exception("Test_split_train")

        if (random_seed != None):
            random.seed(random_seed)
