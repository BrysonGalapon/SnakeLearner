# Expected usage -- `python3 showNN.py [gen]` where gen is the generation of the NN to show a game by
#   Example: >$ python3 showNN.py 3

import sys
from NeuralNetwork import NeuralNetwork
from SnakeAI import SnakeAI

NUM_EXPECTED_ARGS = 1

if __name__ == "__main__":
    if len(sys.argv[1:]) != NUM_EXPECTED_ARGS:
        print("Must provide a single argument [generation] to show a NN game")
        print("Example: >$ python3 showNN.py 3")
        sys.exit(1)
    
    # first argument is 1-indexed
    gen = int(sys.argv[1])

    try:
        loadPath = "./NNs/gen{}/".format(gen)
        nn = NeuralNetwork.load(loadPath)
        ai = SnakeAI(nn, gen)
        ai.play(show=True)
    except Exception as e:
        print(e)
        sys.exit(1)


