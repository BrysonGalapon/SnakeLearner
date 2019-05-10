import os
import sys
from NN_Manager import NN_Manager 
from NeuralNetwork import NeuralNetwork

# assume training a new batch of NNs
continueTraining = False
# number of generations to evolve NN
numGenerations = 200000

def parseCLI():
    global continueTraining
    args = sys.argv[1:]

    if "--continue-training" in args:
        continueTraining = True

def isDirEmpty(directory):
    return len(os.listdir(directory)) == 0

def getBestGenFolder():
    genFolders = os.listdir("./NNs")

    def extractGen(genFolder):
        # removes the 'gen' from 'gen40' and converts the remaining string to an int
        return int(genFolder[3:])

    # sort folders in decreasing gen order
    sortedGenFolder = sorted(genFolders, key=extractGen, reverse=True)
    return sortedGenFolder[0]

def createAdam():
    if continueTraining and not isDirEmpty("./NNs"):
        # adam was a smart boi in a previous lyfe
        return NeuralNetwork.load("./NNs/{}".format(getBestGenFolder()))
    else:
        # default to a stupid adam
        return NeuralNetwork()

def createEve(adam):
    eve = adam.deepCopy()
    # sorry eve
    eve.mutate()
    eve.mutate()
    eve.mutate()
    eve.mutate()
    eve.mutate()
    return eve

def createAdamAndEve():
    adam = createAdam()
    eve  = createEve(adam)
    return adam, eve

if __name__ == "__main__":
    parseCLI()

    # Create the grandfaher NNs
    adam, eve = createAdamAndEve()

    nn_manager = NN_Manager(adam, eve)

    # leave NNs for a while
    nn_manager.evolve(numGenerations)
