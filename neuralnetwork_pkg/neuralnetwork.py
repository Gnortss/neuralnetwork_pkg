from layers import Layer

class NeuralNetwork:
  def __init__(self):
    self.layers : list[Layer] = []

  def add(self, layer : Layer):
    '''Appends the layer to layers of this Neural Network\n
        TODO: check if output and input shapes match
    '''
    self.layers.append(layer)
    return self

  def feedforward(self, inputs):
    outputs = self.layers[0].forward(inputs)
    for layer in self.layers[1:]:
      outputs = layer.forward(outputs)
    return outputs

  def configure(self, optimizer, loss):
    '''Configures the model
    
    '''
    pass

  def print_layers(self):
    print(f"Number of layers: {len(self.layers)}")
    for i, l in enumerate(self.layers):
      print(f"  {i}\t| {str(l)}")