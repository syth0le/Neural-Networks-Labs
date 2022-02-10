from perceptron import Perceptron
from utils.reader import Reader


def main():
    data = Reader.read_data()
    perceptron = Perceptron(size=10)
    perceptron.train(data, epochs=10, nu=0.7)
    result = perceptron.predict(data)
    print(f'This is output: {result}')


if __name__ == '__main__':
    main()
