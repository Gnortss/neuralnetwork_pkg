import numpy as np
import itertools

class Layer:
  """A class to represent a layer in NeuralNetwork"""
  def __init__(self, input_nodes, nodes, activation='sigmoid'):
    self.w = np.random.randn(nodes, input_nodes)
    self.b = np.zeros(nodes)
    self.activation=activation

class NeuralNetwork:
  def __init__(self, input_nodes, hidden_layer_format, output_nodes):
    self.input_nodes = input_nodes
    self.hidden_layer_format = hidden_layer_format
    self.output_nodes = output_nodes

    self.layers = []
    self.build_layers()


  def build_layers(self):
    layers = [self.input_nodes] + self.hidden_layer_format + [self.output_nodes]
    for input_nodes, hidden_nodes in zip(layers, layers[1:]):
      self.layers.append(Layer(input_nodes, hidden_nodes))

  def feedforward(self, a):
    def calc_layer(layer_index, a):
      def sigmoid(x):
        return 1/(1 + np.exp(-x))
      if layer_index >= len(self.layers):
        return a
      return calc_layer(layer_index+1, sigmoid((self.layers[layer_index].w @ a) + self.layers[layer_index].b))

    return calc_layer(0, a)