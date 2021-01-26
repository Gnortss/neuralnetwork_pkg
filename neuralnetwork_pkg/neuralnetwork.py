import numpy as np
import itertools

class Layer:
  """A class to represent a layer in NeuralNetwork"""
  def __init__(self, input_nodes, nodes, activation='sigmoid'):
    self.w = np.random.randn(nodes, input_nodes)
    self.b = np.zeros(nodes)
    self.activation=activation
    self.act_f = lambda x: 1/(1 + np.exp(-x))
    self.act_d = lambda x: x*(1-x) # Expected x is output which is already a sigmoid

  def calc_outputs(self, input):
    self.outputs = self.act_f((self.w @ input) + self.b)
    return self.outputs

class NeuralNetwork:
  def __init__(self, input_nodes, hidden_layer_format, output_nodes):
    self.input_nodes = input_nodes
    self.hidden_layer_format = hidden_layer_format
    self.output_nodes = output_nodes

    self.layers = []
    self.build_layers()

    self.learning_rate = 0.1

  def build_layers(self):
    layers = [self.input_nodes] + self.hidden_layer_format + [self.output_nodes]
    for input_nodes, hidden_nodes in zip(layers, layers[1:]):
      self.layers.append(Layer(input_nodes, hidden_nodes))

  def feedforward(self, a):
    outputs = self.layers[0].calc_outputs(a)
    for layer in self.layers[1:]:
      outputs = layer.calc_outputs(outputs)
    return outputs

  def train(self, inputs, targets):
    def backpropagate(layer_index, errors):
      l = self.layers[layer_index]
      l_1 = self.layers[layer_index-1]
      gradient = l.act_d(l.outputs) * errors * self.learning_rate
      delta_w = gradient.reshape(len(gradient), 1) @ np.atleast_2d(l_1.outputs)
      l.w = l.w + delta_w
      if layer_index == 1:
        return
      n_errors = np.transpose(l.w) @ errors
      backpropagate(layer_index-1, n_errors)

    outputs = self.feedforward(inputs)
    output_errors = targets-outputs
    backpropagate(len(self.layers)-1, output_errors)