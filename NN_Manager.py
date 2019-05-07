import random
from SnakeAI import SnakeAI
import numpy as np
import global_vars
import time

class NN_Manager(object):
    '''
    Takes as input two NNs
    '''
    def __init__(self, adam, eve):
        # each generation is a list of (NN, fitness) tuples
        gen0 = [[adam, 0], [eve, 0]]

        # the number of NNs in the latest nonzero generation
        self.POPULATION_SIZE = 1000
        # the fraction of top-performing NNs that are selected
        self.SELECTION_TOP_FRAC = 0.25
        # approximate frequency of bottom-performing NNs that are selected
        self.SELECTION_BOT_FREQ = 0.2
        # the number of past generations to keep (throw away all other generations)
        self.GENERATION_WINDOW = 3
        # number of seconds to sleep at the end of each generation
        self.GENERATION_SLEEP_TIME = 0.4
        # NN score to exceed to be considered a noteworthy record-breaker
        self.RECORD_BREAKER_THRESH = 1000

        # a list of nn generations
        self.generations = [gen0]

    '''
    Obtains the fitness score for each NN
    '''
    def play(self, i):
        for nn_tuple in self.generations[i]:
            nn, _ = nn_tuple
            ai = SnakeAI(nn, i)

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
        for index, nn_tuple in enumerate(nn_rankings):
            # select a certain fraction of top-performers
            if index < num_top_perf:
                chosen_ones.append(nn_tuple)
                continue

            # randomly select bottom performers
            if random.uniform(0, 1) < self.SELECTION_BOT_FREQ:
                chosen_ones.append(nn_tuple)

        # overwrite generation with only selected NNs
        self.generations[i] = chosen_ones

    '''
    Obtains the strongest NN in generation i
    '''
    def getStrongest(self, i):
        nn_rankings = sorted(self.generations[i], key=lambda x:x[1], reverse=True)
        return nn_rankings[0]

    '''
    Display the strongest game from generation i
    '''
    def showBestNNFromGen(self, i):
        nn, score = self.getStrongest(i)
        ai = SnakeAI(nn, i)
        ai.play(show=True)

    '''
    Display game from given nn
    '''
    def showGame(self, nn_tuple, nn_generation):
        [nn, fitness] = nn_tuple
        ai = SnakeAI(nn, nn_generation)
        ai.play(show=True, fitness=fitness)

    '''
    Breeds all NNs left in generation i, creating generation i+1 with 
        POPULATION_SIZE NNs
    '''
    def breed(self, i):
        curr_gen = self.generations[i]
        next_gen = self.generations[i+1]

        while len(next_gen) < self.POPULATION_SIZE:
            # select 2 parents uniformly at random
            [[p1, f1], [p2, f2]] = random.sample(curr_gen, 2)

            # bebe making ;), while prioritizing the more fit parent
            if f1 > f2:
                child = p1.cross(p2)
            else:
                child = p2.cross(p1)

            # bebes gotta be different than each other!
            child.mutate()

            next_gen.append( [child, None] )

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
        for i in range(1, numGenerations):
            # don't kill CPU -- sleep for a bit
            time.sleep(self.GENERATION_SLEEP_TIME)

            # release 'old' NNs that occur before the GENERATION WINDOW
            if i > self.GENERATION_WINDOW:
                self.generations[i-self.GENERATION_WINDOW] = []

            print("Generation {}: Playing ...".format(i))
            self.play(i)
            print("Generation {}: Killing off the weak ...".format(i))
            self.selection(i)
            print("Generation {}: Rewarding the strong ;) ...".format(i))
            self.breed(i)

            nn_tuple = self.getStrongest(i)
            bestNN, bestGenScore = nn_tuple
            if bestGenScore - global_vars.best_total_score > self.RECORD_BREAKER_THRESH: 
                print("Generation {}: Significant Record Breaker! Showing game ...".format(i))
                self.showGame(nn_tuple, i)

                global_vars.best_total_score = bestGenScore
                global_vars.record_breakers.append(nn_tuple)

            if i % 50 == 0:
                print("Generation {}: Mandatory  game display ...".format(i))
                self.showGame(nn_tuple, i)

        # select best NN from last generation
        print("Generation {}: Playing ...".format(numGenerations))
        self.play(numGenerations)
        print("Generation {}: Selecting the absolute champion ...".format(numGenerations))
        best_nn_tuple = self.getStrongest(numGenerations)
        print("Generation {}: Showing champion NN ...".format(numGenerations))
        self.showGame(best_nn_tuple, numGenerations)

