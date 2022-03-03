from typing import List

from lab_1.reader import TrainData
from perceptron import Perceptron


class Layer:
    def __init__(self, size: int, h: float = 0.75) -> None:
        self.outputs = 10
        self.perceptrons = [Perceptron(size=size, h=h) for _ in range(self.outputs)]

    def train(self, train_data: List[TrainData], nu: float = 000.1, epochs: int = 10) -> None:
        for i in range(self.outputs):
            self.perceptrons[i].train(train_data=train_data, target_number=i, epochs=epochs, nu=nu)

    def predict(self, input_vector: TrainData) -> int:
        answers = [perceptron.predict(input_vector) for perceptron in self.perceptrons]
        return answers.index(1)
