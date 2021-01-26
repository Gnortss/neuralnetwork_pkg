import numpy as np
from neuralnetwork import *
import random

training_data = [((0,0), (0)), ((0,1), (1)), ((1,0), (1)), ((1,1), (0))]

nn = NeuralNetwork(2, [2], 1)

for _ in range(100000):
  inputs, targets = random.choice(training_data)
  nn.train(np.array(inputs), np.array(targets))


for inputs, targets in training_data:
  print(f"{inputs} -> {targets}: nn: {nn.feedforward(np.array(inputs))}")