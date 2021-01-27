import numpy as np
import itertools

class Layer:
  """A class to represent a layer in NeuralNetwork"""
  def __init__(self, input_nodes : int, nodes : int, activation='sigmoid'):
    self.w = np.random.randn(nodes, input_nodes)
    self.b = np.zeros(nodes)
    self.activation=activation
    self.setup_activation(activation)

  def calc_outputs(self, input):
    self.z = (self.w @ input) + self.b
    self.outputs = self.act_f(self.z)
    return self.outputs

  def setup_activation(self, activation):
    if activation == 'sigmoid':
      self.act_f = lambda x: 1/(1 + np.exp(-x))
      self.act_d = lambda x: self.act_f(x)*(1-self.act_f(x)) 
    elif activation == 'relu':
      self.act_f = lambda x: np.maximum(0, x)
      self.act_d = lambda x: 1*(x > 0)
    # elif activation == 'softmax':
    #   def softmax(x):
    #     exps = np.exp(x - x.max())
    #     return exps / np.sum(exps)
    #   def softmax_derivative(x):
    #     x = softmax(x)
    #     s = x.reshape(-1,1)
    #     return np.diagflat(s) - np.dot(s, s.T)
    #   self.act_f = softmax
    #   self.act_d = softmax_derivative
      

class NeuralNetwork:
  def __init__(self, input_nodes : int, hidden_layer_format : list, output_nodes : int):
    self.input_nodes = input_nodes
    self.hidden_layer_format = hidden_layer_format
    self.output_nodes = output_nodes

    self.layers = []
    self.build_layers()

    self.learning_rate = 0.1

  def __init__(self, input_nodes):
    self.input_nodes = input_nodes
    self.learning_rate = 0.1
    self.layers = []

  def add(self, layer : Layer):
    self.output_nodes = len(layer.b)
    self.layers.append(layer)

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

      derivative = l.act_d(l.z)
      gradients = derivative * errors * self.learning_rate
      if layer_index > 0:
        delta_w = gradients.reshape(-1, 1) @ np.atleast_2d(l_1.outputs)
      else:
        delta_w =  gradients.reshape(-1,1 ) @ np.atleast_2d(inputs)
      # Adjust weights and biases by their deltas
      l.w = l.w + delta_w
      l.b = l.b + gradients
      if layer_index == 0:
        return
      n_errors = np.transpose(l.w) @ errors
      backpropagate(layer_index-1, n_errors)

    outputs = self.feedforward(inputs)
    output_errors = targets-outputs
    backpropagate(len(self.layers)-1, output_errors)