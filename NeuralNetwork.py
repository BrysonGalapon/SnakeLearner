import random
import numpy as np
from Mutation import Mutation

# really rough guestimates at mean and std -- accuracy isn't really important here,
#   since we just want to get values somewhat close to 0
INP_MEAN = np.array([0.5, 0.5, 0.5, 0.5, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 0])
INP_STD = np.array([1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 10])

NUM_SNAKE_OUTPUTS = 4

NEW_LINK_WEIGHT_LIMIT = 2
SCALE_WEIGHT_LIMIT = 2
TOGGLE_FREQ = 0.2

def relu(x):
    return x if x > 0 else 0

class NeuralNetwork(object):
    def __init__(self):
        self.num_inputs = INP_MEAN.shape[0]
        self.num_outputs = NUM_SNAKE_OUTPUTS
        self.num_hidden = 2*self.num_inputs

        # first layer
        self.w1 = np.zeros( (self.num_inputs, self.num_hidden) )
        self.b1 = np.zeros( (1, self.num_hidden) )

        # output layer
        self.w2 = np.zeros( (self.num_hidden, self.num_outputs) )
        self.b2 = np.zeros( (1, self.num_outputs) )

        # percentage of weights accounted for by bias vector
        #self.BIAS_FRAC = self.num_outputs/(self.num_inputs*self.num_outputs+self.num_outputs)

    '''
    Saves the state of a given NN to a specified location
    '''
    @staticmethod
    def save(nn, savePath):
        # save the weights of all the matrices
        np.save(savePath+"w1.npy", nn.w1)
        np.save(savePath+"b1.npy", nn.b1)
        np.save(savePath+"w2.npy", nn.w2)
        np.save(savePath+"b2.npy", nn.b2)

    '''
    Loads the state of a given NN from a specified location and returns it
    '''
    @staticmethod
    def load(loadPath):
        # extract matrices from files
        w1 = np.load(loadPath+"w1.npy")
        b1 = np.load(loadPath+"b1.npy")
        w2 = np.load(loadPath+"w2.npy")
        b2 = np.load(loadPath+"b2.npy")

        # initialize the NN with correct weights
        nn = NeuralNetwork()
        nn.w1 = w1
        nn.b1 = b1
        nn.w2 = w2
        nn.b2 = b2
        return nn

    '''
    Returns a deep copy of this NN
    '''
    def deepCopy(self):
        nn_copy = NeuralNetwork()
        nn_copy.w1 = np.copy(self.w1)
        nn_copy.b1 = np.copy(self.b1)
        nn_copy.w2 = np.copy(self.w2)
        nn_copy.b2 = np.copy(self.b2)

        return nn_copy

    '''
    Mutates this NN (in-place)

    One of the following NN mutations is performed uniformly at random:
        - Add a new link with weight between [-NEW_LINK_WEIGHT_LIMIT, NEW_LINK_WEIGHT_LIMIT]
        - Toggle links at a rate of TOGGLE_FREQ
        - Scale the link weight by a random float between [0, SCALE_WEIGHT_LIMIT]
    '''
    def mutate(self):
        mutation = random.choice([Mutation.ADD_NEW_LINK, 
                                  Mutation.TOGGLE_LINKS, 
                                  Mutation.SCALE_LINK])

        if mutation == Mutation.ADD_NEW_LINK:
            # flip coin to mutate bias link or weight link
            if random.uniform(0, 1) < self.BIAS_FRAC: # mutate bias link
                j = random.randrange(self.num_outputs)
                # random weight
                rw = random.uniform(-1*NEW_LINK_WEIGHT_LIMIT, NEW_LINK_WEIGHT_LIMIT)
                # assign weight
                self.bias[0][j] = rw
            else: # mutate weight link
                # random link
                i, j = random.randrange(self.num_inputs), random.randrange(self.num_outputs)
                # random weight
                rw = random.uniform(-1*NEW_LINK_WEIGHT_LIMIT, NEW_LINK_WEIGHT_LIMIT)
                # assign weight
                self.weight[i][j] = rw

        elif mutation == Mutation.TOGGLE_LINKS:
            # toggle weight links
            for i in range(self.num_inputs):
                for j in range(self.num_outputs):
                    # flip coin
                    if random.uniform(0, 1) < TOGGLE_FREQ:
                        if self.weight[i][j] == 0:
                            # turn link on with random weight
                            rw = random.uniform(-1*NEW_LINK_WEIGHT_LIMIT, NEW_LINK_WEIGHT_LIMIT)
                            # assign weight
                            self.weight[i][j] = rw
                        else:
                            self.weight[i][j] = 0

            # toggle bias links
            for j in range(self.num_outputs):
                # flip coin
                if random.uniform(0, 1) < TOGGLE_FREQ:
                    # turn link on with random weight
                    rw = random.uniform(-1*NEW_LINK_WEIGHT_LIMIT, NEW_LINK_WEIGHT_LIMIT)
                    # assign weight
                    self.bias[0][j] = rw

        elif mutation == Mutation.SCALE_LINK:
            # flip coin to mutate bias link or weight link
            if random.uniform(0, 1) < self.BIAS_FRAC: # mutate bias link
                j = random.randrange(self.num_outputs)
                # random scale factor
                sf = random.uniform(0, SCALE_WEIGHT_LIMIT)
                # scale weight
                self.bias[0][j] *= sf
            else:
                # random link
                i, j = random.randrange(self.num_inputs), random.randrange(self.num_outputs)
                # random scale factor
                sf = random.uniform(0, SCALE_WEIGHT_LIMIT)
                # scale weight
                self.weight[i][j] *= sf

        else:
            raise Error("Unexpected mutation choice: {}".format(mutation))

    '''
    Performs a cross-over between this NN and another NN, and returns the crossed over NN.
        The genes of this NN are prioritized in the cross over
    '''
    def cross(self, other):
        # prioritize the current genes by copying them over first
        child = self.deepCopy()

        # cross over weights
        for i in range(self.num_inputs):
            for j in range(self.num_outputs):
                # only express other genes if they haven't already been set
                if other.weight[i][j] != 0 and child.weight[i][j] == 0:
                    child.weight[i][j] = other.weight[i][j]

        # cross over biases
        for j in range(self.num_outputs):
            # only express other genes if they haven't already been set
            if other.bias[0][j] != 0 and child.bias[0][j] == 0:
                child.bias[0][j] = other.bias[0][j]
        return child

    '''
    Perform an NN calculation. 

    Takes in a (1, NUM_IN) numpy array representing
        the input of the NN calculation
    
    Returns a (1, NUM_OUT) numpy array representing
        the output of the NN calculation.
    '''
    def output(self, inp):
        norm_inp = self.normalize(inp)

        # apply first hidden layer
        z1 = np.dot(norm_inp, self.w1)+self.b1
        fz1 = np.apply_along_axis(relu, axis=0, arr=z1)

        # apply output layer
        z2 = np.dot(fz1, self.w2)+self.b2

        return z2

    def normalize(self, x):
        return (x-INP_MEAN)/INP_STD

    def __str__(self):
        outStr = ""
        outStr += "Num Input Nodes: {}\n".format(self.num_inputs)
        outStr += "Num Output Nodes: {}\n".format(self.num_outputs)
        outStr += "W1 Matrix: \n"
        outStr += str(self.w1) + "\n"
        outStr += "B1 Matrix: \n"
        outStr += str(self.b1) + "\n"
        outStr += "W2 Matrix: \n"
        outStr += str(self.w2) + "\n"
        outStr += "B2 Matrix: \n"
        outStr += str(self.b2) + "\n"

        return outStr
