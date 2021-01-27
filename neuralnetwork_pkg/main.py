import numpy as np
from neuralnetwork import *
import random
from layers import Dense, Activation

model = NeuralNetwork() \
  .add(Dense(2, 4)) \
  .add(Activation(4)) \
  .add(Dense(4, 2)) \
  .add(Activation(2, activation='softmax'))

print(model.feedforward([-2, 2]))

# training_data = [((0,0), 0), ((0,1), 1), ((1,0), 1), ((1,1), 0)]

# # nn = NeuralNetwork(2, [2], 1)
# nn = NeuralNetwork(2)
# nn.add(Layer(2, 2))
# nn.add(Layer(2, 2))
# nn.add(Layer(2, 1))

# for _ in range(100000):
#   inputs, targets = random.choice(training_data)
#   nn.train(np.array(inputs), np.array(targets))


# for inputs, targets in training_data:
#   print(f"{inputs} -> {targets}: nn: {nn.feedforward(np.array(inputs))}")