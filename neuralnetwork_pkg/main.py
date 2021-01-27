import numpy as np
from neuralnetwork import *
import random
# import tensorflow as tf

# mnist = tf.keras.datasets.mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = tf.keras.utils.normalize(x_train, axis=1).reshape(len(x_train), 784)
# x_test = tf.keras.utils.normalize(x_test, axis=1).reshape(len(x_test), 784)


# nn = NeuralNetwork(784)
# nn.add(Layer(784, 128, activation='relu'))
# nn.add(Layer(128, 128, activation='relu'))
# nn.add(Layer(128, 10, activation='softmax'))

training_data = [((0,0), 0), ((0,1), 1), ((1,0), 1), ((1,1), 0)]

# nn = NeuralNetwork(2, [2], 1)
nn = NeuralNetwork(2)
nn.add(Layer(2, 2))
nn.add(Layer(2, 2))
nn.add(Layer(2, 1))

for _ in range(100000):
  inputs, targets = random.choice(training_data)
  nn.train(np.array(inputs), np.array(targets))


for inputs, targets in training_data:
  print(f"{inputs} -> {targets}: nn: {nn.feedforward(np.array(inputs))}")