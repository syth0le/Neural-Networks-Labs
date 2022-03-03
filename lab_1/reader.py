import dataclasses
import json

from typing import List


@dataclasses.dataclass
class TrainData:
    number: int
    data: List[int]


class DataReader:
    def __init__(self) -> None:
        self.train_file = 'train_data.json'
        self.test_file = 'test_data.json'

    def get_train_data(self) -> List[TrainData]:
        return self.__get_data(self.train_file)

    def get_test_data(self) -> List[TrainData]:
        return self.__get_data(self.test_file)

    @staticmethod
    def __get_data(file: str) -> List[TrainData]:
        with open(file) as f:
            data = json.load(f)

        result_list = []
        for item in data:
            result_list.append(TrainData(number=int(item), data=data[item]))
        return result_list

