from perceptron import Perceptron
from reader import DataReader

NU = 000.1
H = 0.7
# learning_rate - это скорость обучения


def main():
    train_data = DataReader(4).get_train_data()
    perceptron = Perceptron(size=len(train_data[0].data), h=H)

    perceptron.train(train_data=train_data, nu=NU)

    for item in train_data:
        print(f'Prediction for number {item.number} is:  ', perceptron.predict(item))


if __name__ == '__main__':
    main()
