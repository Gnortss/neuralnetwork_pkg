import numpy as np
from neuralnetwork import *

nn = NeuralNetwork(2, [3, 4, 3], 2)
# print(nn.feedforward(np.array([1,1])))

# print('layer outputs')
# for layer in nn.layers:
#   print(layer.outputs, layer.act_d(layer.outputs))

nn.train(np.array([1, 1]), np.array([0, 1]))