import random
import numpy as np
from Mutation import Mutation
from LinkType import LinkType

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
            linkType = self.selectUniformLink()

            if linkType == LinkType.W1:
                i, j = NeuralNetwork.selectRandom2DCoord(*self.w1.shape)
                # assign random weight
                self.w1[i][j] = random.uniform(-1*NEW_LINK_WEIGHT_LIMIT, NEW_LINK_WEIGHT_LIMIT)
            elif linkType == LinkType.B1:
                i, j = NeuralNetwork.selectRandom2DCoord(*self.b1.shape)
                # assign random weight
                self.b1[i][j] = random.uniform(-1*NEW_LINK_WEIGHT_LIMIT, NEW_LINK_WEIGHT_LIMIT)
            elif linkType == LinkType.W2:
                i, j = NeuralNetwork.selectRandom2DCoord(*self.w2.shape)
                # assign random weight
                self.w2[i][j] = random.uniform(-1*NEW_LINK_WEIGHT_LIMIT, NEW_LINK_WEIGHT_LIMIT)
            elif linkType == LinkType.B2:
                i, j = NeuralNetwork.selectRandom2DCoord(*self.b2.shape)
                # assign random weight
                self.b2[i][j] = random.uniform(-1*NEW_LINK_WEIGHT_LIMIT, NEW_LINK_WEIGHT_LIMIT)
            else:
                raise Error("Unexpected link choice: {}\n".format(linkType))

        elif mutation == Mutation.TOGGLE_LINKS:
            # toggle all links at the given TOGGLE_FREQ
            NeuralNetwork.toggleMatrix(self.w1)
            NeuralNetwork.toggleMatrix(self.b1)
            NeuralNetwork.toggleMatrix(self.w2)
            NeuralNetwork.toggleMatrix(self.b2)

        elif mutation == Mutation.SCALE_LINK:
            linkType = self.selectUniformLink()

            if linkType == LinkType.W1:
                i, j = NeuralNetwork.selectRandom2DCoord(*self.w1.shape)
                # scale by random weight
                self.w1[i][j] *= random.uniform(0, SCALE_WEIGHT_LIMIT)
            elif linkType == LinkType.B1:
                i, j = NeuralNetwork.selectRandom2DCoord(*self.b1.shape)
                # scale by random weight
                self.b1[i][j] *= random.uniform(0, SCALE_WEIGHT_LIMIT)
            elif linkType == LinkType.W2:
                i, j = NeuralNetwork.selectRandom2DCoord(*self.w2.shape)
                # scale by random weight
                self.w2[i][j] *= random.uniform(0, SCALE_WEIGHT_LIMIT)
            elif linkType == LinkType.B2:
                i, j = NeuralNetwork.selectRandom2DCoord(*self.b2.shape)
                # scale by random weight
                self.b2[i][j] *= random.uniform(0, SCALE_WEIGHT_LIMIT)
            else:
                raise Error("Unexpected link choice: {}\n".format(linkType))

        else:
            raise Error("Unexpected mutation choice: {}\n".format(mutation))

    '''
    Performs a cross-over between this NN and another NN, and returns the crossed over NN.
        The genes of this NN are prioritized in the cross over
    '''
    def cross(self, other):
        # prioritize the current genes by copying them over first
        child = self.deepCopy()

        # cross all matrices
        NeuralNetwork.crossMatrix(child.w1, other.w1)
        NeuralNetwork.crossMatrix(child.b1, other.b1)
        NeuralNetwork.crossMatrix(child.w2, other.w2)
        NeuralNetwork.crossMatrix(child.b2, other.b2)
        return child

    '''
    Mutates zero entries in m1 with corresponding entries in m2 
    '''
    @staticmethod
    def crossMatrix(m1, m2):
        num_rows, num_cols = m1.shape

        for i in range(num_rows):
            for j in range(num_cols):
                if m1[i][j] == 0 and m2[i][j] != 0:
                    m1[i][j] = m2[i][j]

    '''
    Selects link type uniformly at random -- one of:
        - W1 link
        - B1 link
        - W2 link
        - B2 link
    and returns the link type
    '''
    def selectUniformLink(self):
        # enumerate each cell as an integer in the flat range [0, totCells)
        numW1cells = NeuralNetwork.numCells(self.w1)
        numB1cells = NeuralNetwork.numCells(self.b1)
        numW2cells = NeuralNetwork.numCells(self.w2)
        numB2cells = NeuralNetwork.numCells(self.b2)

        totCells = numW1cells+numB1cells+numW2cells+numB2cells
        
        # choose a cell (integer) uniformly in the flat range
        cellInt = random.randrange(totCells)

        # cut the flat range into chunks corresponding to link type, and
        #   return the chunk that the chosen cell lies in
        flatRangeSpace = [(LinkType.W1, numW1cells),
                          (LinkType.B1, numB1cells),
                          (LinkType.W2, numW2cells),
                          (LinkType.B2, numB2cells)]

        for (linkType, numLinks) in flatRangeSpace:
            if cellInt < numLinks:
                # chosen integer fell in chunk range
                return linkType
            else:
                # translate flat range by excluded chunk size
                cellInt -= numLinks

        raise Error("Unable to select a uniform link -- chose a cell outside of possible range")

    '''
    Counts the number of cells in the given matrix
    '''
    @staticmethod
    def numCells(m):
        num_rows, num_cols = m.shape
        return num_rows*num_cols

    '''
    Selects a random (i, j) cell
    '''
    @staticmethod
    def selectRandom2DCoord(num_rows, num_cols):
        i = random.randrange(num_rows)
        j = random.randrange(num_cols)
        return i, j
    
    '''
    Toggles all elements of the matrix according to the TOGGLE_FREQ. To 'toggle' means
        that if an element is zero, it is assigned a random value, and 
        if an element is nonzero, it is assigned a zero value
    '''
    @staticmethod
    def toggleMatrix(m):
        num_rows, num_cols = m.shape

        for i in range(num_rows):
            for j in range(num_cols):
                # flip coin
                if random.uniform(0, 1) < TOGGLE_FREQ:
                    NeuralNetwork.toggleMatrixLoc(m, i, j)

    '''
    Toggles the value at location (i, j) in matrix m
    '''
    @staticmethod
    def toggleMatrixLoc(m, i, j):
        if m[i][j] == 0:
            # assign random weight
            m[i][j] = random.uniform(-1*NEW_LINK_WEIGHT_LIMIT, NEW_LINK_WEIGHT_LIMIT)
        else:
            m[i][j] = 0

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
