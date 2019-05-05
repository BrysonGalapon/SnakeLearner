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
        # the fraction of top-performing NNs that are selected
        self.SELECTION_TOP_FRAC = 0.25
        # approximate frequency of bottom-performing NNs that are selected
        self.SELECTION_BOT_FREQ = 0.2

        # a list of nn generations
        self.generations = [gen0]

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
        # keep track of selected NNs
        chosen_ones = []
        num_top_perf = int(self.POPULATION_SIZE*self.SELECTION_TOP_FRAC)

        # sort by fitness in decreasing order
        nn_rankings = sorted(self.generations[i], key=lambda x:x[1], reverse=True)
        for i, nn_tuple in enumerate(nn_rankings):
            # select a certain fraction of top-performers
            if i < num_top_perf:
                chosen_ones.append(nn_tuple)
                continue

            # randomly select bottom performers
            if np.uniform(0, 1) < self.SELECTION_BOT_FREQ:
                chosen_ones.append(nn_tuple)

        # overwrite generation with only selected NNs
        self.generations[i] = chosen_ones

    '''
    Breeds all NNs left in generation i, creating generation i+1 with 
        POPULATION_SIZE NNs
    '''
    def breed(self, i):
        curr_gen = self.generations[i]
        next_gen = self.generations[i+1]

        while len(next_gen) < self.POPULATION_SIZE:
            # select 2 parents uniformly at random
            [(p1, f1), (p2, f2)] = np.sample(curr_gen, 2)

            # bebe making ;), while prioritizing the more fit parent
            if f1 > f2:
                child = p1.cross(p2)
            else:
                child = p2.cross(p1)

            # bebes gotta be different than each other!
            child.mutate()

            next_gen.append( (child, None) )

    '''
    Simulates evolution for a fixed number of generations
    '''
    def evolve(self, numGenerations=10):
        # allocate space for numGenerations elements (1-indexed)
        while len(self.generations) <= numGenerations:
            self.generations.append([])

        # breed the first generation of NNs
        self.breed(0)

        # note that i is 1-indexed
        for i in range(1, numGenerations+1):
            print("Generation {}: Playing ...".format(i))
            self.play(i)
            print("Generation {}: Killing off the weak ...".format(i))
            self.selection(i)
            print("Generation {}: Rewarding the strong ;) ...".format(i))
            self.breed(i)

