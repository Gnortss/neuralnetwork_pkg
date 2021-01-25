import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal
from neuralnetwork_pkg.neuralnetwork import NeuralNetwork

class TestNeuralNetwork(unittest.TestCase):
  def test_feedforward(self):
    def sigmoid(x):
      return 1/(1+np.exp(-x))

    def calc_layer(a, w, b):
      tmp = []
      for row, bias in zip(w, b):
        sum = bias
        for x1, x2 in zip(row, a):
          sum += x1*x2
        tmp.append(sum)
      ret = []
      for x in tmp:
        ret.append(sigmoid(x))
      return np.array(ret)
      


    nn = NeuralNetwork(2,[3,4,3],2)
    a0 = np.array([1,1])
    a1 = calc_layer(a0, nn.layers[0].w, nn.layers[0].b)
    a2 = calc_layer(a1, nn.layers[1].w, nn.layers[1].b)
    a3 = calc_layer(a2, nn.layers[2].w, nn.layers[2].b)
    a4 = calc_layer(a3, nn.layers[3].w, nn.layers[3].b)
    
    assert_array_almost_equal(a4, nn.feedforward(a0))

if __name__ == '__main__':
    unittest.main()