import random
import numpy as np

class NeuralNetwork(object):
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.weight = np.zeros( (num_inputs, num_outputs) )


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
        pass

    
