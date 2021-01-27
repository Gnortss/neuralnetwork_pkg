from layers.layer import Layer
import numpy as np

class Activation(Layer):
  def __init__(self, input_shape, activation='relu'):
    super().__init__(input_shape=input_shape)
    self.name = f"Activation\t({activation})"
    self.activation=activation
    self.setup()

  def forward(self, input):
    """ Applies activation function and forwards it"""
    return self.act_f(input)

  def setup(self):
    if self.activation == 'relu':
      self.act_f = lambda x: np.maximum(0, x)
      self.act_d = lambda x: 1*(x > 0)
    elif self.activation == 'sigmoid':
      self.act_f = lambda x: 1/(1 + np.exp(-x))
      self.act_d = lambda x: self.act_f(x)*(1-self.act_f(x)) 
    elif self.activation == 'softmax':
      def softmax(x):
        exps = np.exp(x - x.max())
        return exps / np.sum(exps)
      self.act_f = softmax