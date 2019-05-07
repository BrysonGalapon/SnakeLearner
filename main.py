from NN_Manager import NN_Manager 
from NeuralNetwork import NeuralNetwork

if __name__ == "__main__":
    num_generations = 200000

    # Create the grandfaher NNs
    adam = NeuralNetwork()
    eve = NeuralNetwork()

    nn_manager = NN_Manager(adam, eve)

    # leave NNs for a while
    nn_manager.evolve(num_generations)
