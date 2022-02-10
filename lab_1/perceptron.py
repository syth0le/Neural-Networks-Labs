from typing import List

import numpy as np


class Perceptron:

    def __init__(self, size: int):
        self.weights = self._get_weights(size)

    def train(self, data: List[float], nu: float, epochs: int = 1):
        for epoch in range(epochs):
            sum_local = self.__get_sum_for_weight(data)
            res = self.__get_activation_function(sum_local)
            error = self.__get_error(res)
            self.__normalize_weights(data, nu, error)

    def fit(self):
        pass

    def predict(self, data: List[float]):
        sum_local = self.__get_sum_for_weight(data)
        return self.__get_activation_function(sum_local)

    @staticmethod
    def _get_weights(size: int) -> np.array:
        return np.random.uniform(-1, 1, size)

    @staticmethod
    def __get_activation_function(sum: float) -> int:
        return 1 if sum > 0 else 0

    def __get_sum_for_weight(self, data: List[float]) -> float:
        res_array = []
        for i in range(len(data)):
            res_array.append(data[i] * self.weights[i])
        return sum(res_array)

    @staticmethod
    def __get_error(res: float) -> float:
        return self.number - res

    def __normalize_weights(self, data: List[float],  nu: float, error: float) -> None:
        for i, _ in enumerate(self.weights):
            self.weights[i] = error * nu * data[i]
