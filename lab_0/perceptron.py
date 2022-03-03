from typing import List

import numpy as np

from reader import TrainData


class Perceptron:

    def __init__(self, size: int, h: float = 0.75) -> None:
        self.h = h
        self.weights = self._get_weights(size)

    def train(self, train_data: List[TrainData], nu: float = 000.1, epochs: int = 10) -> None:
        for epoch in range(epochs):
            for item in train_data:
                sum_local = self.__get_sum_for_weight(item.data)
                res = self.__get_activation_function(sum_local)
                error = self.__get_error(item.target, res)
                self.__normalize_weights(item.data, nu, error)

    def predict(self, test_data: TrainData) -> int:
        sum_local = self.__get_sum_for_weight(test_data.data)
        return self.__get_activation_function(sum_local)

    @staticmethod
    def _get_weights(size: int) -> np.array:
        return np.random.uniform(-1, 1, size)

    @staticmethod
    def __get_activation_function(sum_local: float) -> int:
        return 1 if sum_local > 1 else 0

    def __get_sum_for_weight(self, data: List[float]) -> float:
        net_y = 0
        length = len(data)
        if length != len(self.weights):
            raise Exception
        for i in range(length):
            net_y += data[i] * self.weights[i]
        return net_y

    @staticmethod
    def __get_error(target: int, res: float) -> float:
        return target - res

    def __normalize_weights(self, data: List[float],  nu: float, error: float) -> None:
        for i in range(len(self.weights)):
            self.weights[i] += error * nu * data[i]
