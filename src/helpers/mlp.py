import random
from helpers.value import Value
class Neuron:
    def __init__(self, number_of_input_weights):
        self.W = [Value(random.uniform(-1, 1)) for _ in range(number_of_input_weights)]
        self.b = Value(random.uniform(-1, 1))
    def __call__(self, X):
        return sum((Xi * Wi for Xi, Wi in zip(X, self.W)), self.b)
    def parameters(self):
        return self.W + [self.b]
class Layer:
    def __init__(self, number_of_input_weights, number_of_neurons):
        self.neurons = [Neuron(number_of_input_weights) for _ in range(number_of_neurons)]
    def __call__(self, X):
        outs = [neuron(X) for neuron in self.neurons]
        return outs[0] if len(outs) == 1 else outs   # to return single output for the last layer
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
class MLP:
    def __init__(self, number_of_inputs, neuron_count_per_layer):
        neuron_counts = [number_of_inputs] + neuron_count_per_layer
        self.layers = [Layer(neuron_counts[i], neuron_counts[i+1]) for i in range(len(neuron_count_per_layer))]
    def __call__(self, X):
        # Feed the output of each layer sequentially into the following layer,
        # ultimately producing a single output from the final layer.
        for layer in self.layers:
            X = layer(X)
        return X
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
        