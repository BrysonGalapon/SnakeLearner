import random
import numpy as np

def relu(x):
    return x if x > 0 else 0

class NeuralNetwork(object):
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.weight = np.zeros( (num_inputs, num_outputs) )
        self.bias = np.zeros( (1, num_outputs) )

    '''
    Returns a deep copy of this NN
    '''
    def deepCopy(self):
        pass

    '''
    Mutates this NN (in-place)

    One of the following NN mutations is performed uniformly at random
    '''
    def mutate(self):
        pass

    '''
    Perform an NN calculation. 

    Takes in a (1, NUM_IN) numpy array representing
        the input of the NN calculation
    
    Returns a (1, NUM_OUT) numpy array representing
        the output of the NN calculation.
    '''
    def output(self, inp):
        z = np.dot(inp, self.weight)+self.bias
        return np.apply_along_axis(relu, axis=0, arr=z)
