import numpy as np


class LinearRegression:
    import numpy as np

    def __init__(self):
        self.weights = 100
        self.bias = 0

    def get_loss(self, x, y, weight, bias):
        """
        평균 제곱 오차 방식을 사용해서 오차를 구한다.
        :param x:
        :param y:
        :param weight:
        :param bias:
        :return:
        """

        return sum((x * weight + bias - y) ** 2) / len(y) / 2

    def get_slope(self, x, y, weight, bias, d_x=0.0001):
        """
        측정 가중치에서의 오차 함수의 경사를 구한다.
        :param x:
        :param y:
        :param weight:
        :param bias:
        :param d_x:
        :return:
        """
        return (self.get_loss(x, y, weight + d_x, bias) - self.get_loss(x, y, weight, bias)) / d_x

    def predict(self, x):
        """
        여측하는 함수
        :param x: 예측하는 독립 변수
        :param weight: 예측에 사용하는 가중치, 입력하지 않는다면 모델에 저장된 가중치를 사용
        :param bias: 예측에 사용하는 편향, 입력하지 않는다면 모델에 저장된 편향을 사용
        :return:
        """
        return x * self.weights + self.bias

    def fit(self, x, y, epoch=300, learning_rate=0.001):
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
            slop = self.get_slope(x, y, self.weights, self.bias)
            self.weights -= slop * learning_rate

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
