import numpy as np

class Layer:
  def __init__(self, input_shape):
    self.name = 'Abstract layer'
    self.input_shape = input_shape

  def forward(self, input):
    """ Feedforward input """
    return input

  def __str__(self) -> str:
    return self.name