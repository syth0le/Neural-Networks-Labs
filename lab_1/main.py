from random import shuffle

from layer import Layer
from reader import DataReader

NU = 000.1  # learning_rate - это скорость обучения
H = 0.7


def main():
    reader = DataReader()
    train_data = reader.get_train_data()
    size = len(train_data[0].data)

    layer = Layer(size=size, h=H)
    layer.train(train_data=train_data, epochs=30, nu=NU)

    print('THE END OF TRAINING\n')
    print('PREDICTIONS:')
    test_data = reader.get_test_data()
    shuffle(test_data)
    for item in test_data:
        try:
            print(f'This is output for ({item.number}). Answer is: {layer.predict(item)}')
        except ValueError:
            print('Cannot recognize the number')


if __name__ == '__main__':
    main()
