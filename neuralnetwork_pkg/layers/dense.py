from layers.layer import Layer
import numpy as np

class Dense(Layer):
  def __init__(self, input_shape, nodes):
    super().__init__(input_shape)
    self.name = f"Dense\t\t({nodes})"
    self.nodes = nodes

    self.w = np.random.randn(nodes, input_shape)
    self.b = np.zeros(nodes)

  def forward(self, input):
    """ Applies weights and biases to given input """
    return (self.w @ input) + self.b