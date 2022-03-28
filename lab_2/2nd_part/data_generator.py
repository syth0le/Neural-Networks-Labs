import dataclasses
import json
import random
from multiprocessing import Pool
from typing import Callable


@dataclasses.dataclass
class RepresentedData:
    number: int
    data: list[int]


class DataGenerator:
    def __init__(self, train_data: int, test_data: int) -> None:
        self.__data_file = 'data/data.json'
        self.__train_file = 'data/train_data.json'
        self.__test_file = 'data/test_data.json'
        self.train_data = train_data
        self.test_data = test_data
        self.temp_data = self.__get_data(file=self.__data_file)

    @staticmethod
    def __get_data(file: str) -> dict:
        with open(file) as f:
            # print(json.load(f))
            return json.load(f)["data"]

    @staticmethod
    def __decorator_represent_data_as_dataclass(func: Callable) -> list[RepresentedData]:
        def wrapper(self):
            temp_data = []
            for item in func(self):
                key = list(item.keys())[0]
                value = list(item.values())[0]
                temp_data.append(RepresentedData(number=int(key), data=value))
            return temp_data

        return wrapper

    def __generate_data(self, file: str, iterations: int = 10) -> None:
        with open(file, 'r+') as wf:
            file_data = json.load(wf)
            # is_bool = True
            # with Pool(10) as pool:
            #     print(pool)
            # for _ in pool.map(func, B)
            for _ in range(iterations):
                for value in self.temp_data:
                    length = len(list(value.values())[0])
                    random_positions = self.__get_random_positions_to_change(length=length, amount=3)
                    # print(random_positions)
                    for index in random_positions:
                        data_values = list(value.values())[0]
                        # print(index, data_values[index])
                        data_values[index] = 1 if data_values[index] == 0 else 0
                        # print(index, data_values[index])
                        value[list(value.keys())[0]] = data_values
                    # print(value)
                    # is_bool = False
                    file_data['data'].append(value)
                    wf.seek(0)
                    json.dump(file_data, wf, indent=4)

    @staticmethod
    def __get_random_positions_to_change(length: int = 35, amount: int = 3) -> list[int]:
        return [random.randint(0, length - 1) for _ in range(amount)]

    def generate_train_data(self) -> None:
        self.__generate_data(file=self.__train_file, iterations=30)

    def generate_test_data(self) -> None:
        self.__generate_data(file=self.__test_file, iterations=10)

    @__decorator_represent_data_as_dataclass
    def get_train_data(self) -> list[RepresentedData]:
        return self.__get_data(self.__train_file)

    @__decorator_represent_data_as_dataclass
    def get_test_data(self) -> list[RepresentedData]:
        return self.__get_data(self.__test_file)


if __name__ == '__main__':
    generator = DataGenerator(test_data=100, train_data=500)
    generator.generate_train_data()
    data = generator.get_train_data()
    print(len(data))
