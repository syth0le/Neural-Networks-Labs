from typing import List

import numpy as np


class Perceptron:

    def __init__(self, size: int, i: int = 0, h: float = 0.75) -> None:
        self.i = i
        self.h = h
        self.weights = self._get_weights(size)

    def train(self, data: List[float], nu: float = 1, epochs: int = 1) -> None:
        target = self.__get_target_data(0)
        for epoch in range(epochs):
            sum_local = self.__get_sum_for_weight(data) + ((-1) * self.h)
            res = self.__get_activation_function(sum_local)
            error = self.__get_error(target, res)
            self.__normalize_weights(data, nu, error)
            self.h += error * (-1) * nu
            print(f'>epoch={epoch}, lrate={nu}')

    def fit(self):
        pass

    def predict(self, data: List[float]):
        sum_local = self.__get_sum_for_weight(data) - self.h
        return self.__get_activation_function(sum_local)

    @staticmethod
    def _get_weights(size: int) -> np.array:
        return np.random.uniform(-1, 1, size)

    @staticmethod
    def __get_activation_function(sum_local: float) -> int:
        return 1 if sum_local > 0 else 0

    def __get_sum_for_weight(self, data: List[float]) -> float:
        net_y = 0
        length = len(data)
        if length != len(self.weights):
            raise Exception
        for i in range(length):
            net_y += data[i] * self.weights[i]
        return net_y

    @staticmethod
    def __get_error(target: float, res: float) -> float:
        return target - res

    def __normalize_weights(self, data: List[float],  nu: float, error: float) -> None:
        for i, _ in enumerate(self.weights):
            self.weights[i] = error * nu * data[i]

    @staticmethod
    def __get_target_data(number: int) -> float:
        return 0
