class LinearRegression:
    import numpy as np

    def __init__(self):
        self.weight = 0
        self.bias = 0
    def get_cost(self, x, y, weight, bias):
        """평균 오차 제곱의 방식으로 오차를 구한다."""
        if(x.shape[0] != y.shape[0]):
            raise Exception("") #Todo(에러의 이름 적기)

        return sum((self.predict(x, weight, bias) - y) ** 2) / len(x)

    def predict(self, x, weight = None, bias = None):
        """
        :param x: 예측하는 독립 변수
        :param weight: 예측에 사용하는 가중치, 입력하지 않는다면 모델에 저장된 가중치를 사용
        :param bias: 예측에 사용하는 편향, 입력하지 않는다면 모델에 저장된 편향을 사용
        :return:
        """

        w = weight
        if(weight is None):
            w = self.weight

        b = bias
        if(bias is None):
            b = self.bias

        return x * w + b

    def fit(self, x, y, epoch, ):
        pass

    def evaluate(self):
        # Todo("평가 함수 작성")
        pass


class Util:
    def test_split_train(x, y, test_ratio, random_seed = None):
        import random

        if(x.shape[0] != y.shape[0]):
            raise Exception("Test_split_train")

        if(random_seed != None):
            random.seed(random_seed)