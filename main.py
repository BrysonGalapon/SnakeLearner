from NN_Manager import NN_Manager 
from NeuralNetwork import NeuralNetwork

if __name__ == "__main__":
    num_inputs = 11
    num_outputs = 4
    num_generations = 200000

    # Create the grandfaher NNs
    adam = NeuralNetwork(num_inputs, num_outputs)
    eve = NeuralNetwork(num_inputs, num_outputs)

    nn_manager = NN_Manager(adam, eve)

    # leave NNs for a while
    nn_manager.evolve(num_generations)
