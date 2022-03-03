import dataclasses
import json

from typing import List


@dataclasses.dataclass
class TrainData:
    target: int
    number: int
    data: List[int]


class DataReader:
    def __init__(self, target_number: int = 0) -> None:
        self.file = 'train_data.json'
        if 0 <= target_number <= 9:
            self.target_number = str(target_number)
        else:
            raise AttributeError
        self.train_data = []

    def get_train_data(self) -> List[TrainData]:
        with open(self.file) as f:
            self.train_data = json.load(f)

        result_list = []
        for item in self.train_data:
            target = 0
            if self.target_number == item:
                target = 1
            result_list.append(TrainData(target=target, number=int(item), data=self.train_data[item]))
        return result_list

