from perceptron import Perceptron
from utils.reader import Reader

NU = 000.1
H = 0.7


def main():
    reader = Reader()
    data = reader.read_train_data()
    size = len(data[0])
    layers = []
    for i in range(10):
        perceptron = Perceptron(size=size, i=i, h=H)
        layers.append(perceptron)

    for number in range(len(data)):
        print(f'Number: {number} in train data')
        target = reader.get_target_data(number=number)
        for i in range(10):
            perceptron = layers[i]
            perceptron.train(data=data[number], target=target, epochs=1000, nu=NU)
            print(f'Perceptron: {i}, {perceptron.predict(data[number])}')

    print('THE END OF TRAINING\n')

    for perceptron in layers:
        test_data = reader.read_test_data()
        result = perceptron.predict(data=test_data)
        print(f'This is output for {perceptron}: {result}')


if __name__ == '__main__':
    main()
