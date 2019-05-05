import random
from SnakeAI import SnakeAI

class NN_Manager(object):
    '''
    Takes as input two NNs
    '''
    def __init__(self, adam, eve):
        # each generation is a list of (NN, fitness) tuples
        gen0 = [[adam, None], [eve, None]]

        # the number of NNs in the latest nonzero generation
        self.POPULATION_SIZE = 50

        # a list of nn generations
        self.generations = []

    '''
    Performs a cross-over between two NNs, not modifying either NNs,
        and returns the crossed-over NN
    '''
    def cross(self, nn_1, nn_2):
        pass

    '''
    Obtains the fitness score for each NN
    '''
    def play(self, i):
        for nn_tuple in self.generations[i]:
            nn, _ = nn_tuple
            ai = SnakeAI(nn)

            # get and assign NN fitness 
            fitness = ai.play()
            nn_tuple[1] = fitness

    '''
    Emulates "survival of the fittest". Removes NNs in generation i that are "unfit"
    '''
    def selection(self, i):
        pass

    '''
    Breeds all NNs left in generation i, creating generation i+1 with 
        POPULATION_SIZE NNs
    '''
    def breed(self, i):
        pass

    '''
    Simulates evolution for a fixed number of generations
    '''
    def evolve(self, numGenerations=10):
        # breed the first generation of NNs
        self.breed(0)

        for i in range(1, numGenerations+1):
            print("Generation {}: Playing ...".format(i))
            self.play(i)
            print("Generation {}: Killing off the weak ...".format(i))
            self.selection(i)
            print("Generation {}: Rewarding the strong ;) ...".format(i))
            self.breed(i)

