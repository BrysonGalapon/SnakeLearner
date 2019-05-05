from Snake import Snake
from Action import Action
import numpy as np
import time

class SnakeAI(object):
    def __init__(self, nn, width=10, height=10):
        self.nn = nn
        self.snake = Snake(width, height)

    '''
    Plays a game of Snake. Returns game score.
    '''
    def play(self, show=False):
        while self.snake.alive():
            if show:
                print(self.snake)
                time.sleep(0.3)

            state = self.snake.getCurrentState()
            out = self.nn.output(state)
            if show:
                print("Output: ", out)
            action = self.translateOutput(out)
            if show:
                print("NN chose: ", action)
            self.snake.step(action)

        return self.snake.score()

    '''
    Converts the NN output into the specified Action to perform
    '''
    def translateOutput(self, out):
        max_index = np.argmax(out)

        if max_index == 0:
            return Action.UP
        elif max_index == 1:
            return Action.LEFT
        elif max_index == 2:
            return Action.RIGHT
        elif max_index == 3:
            return Action.DOWN
        else:
            raise Error("Unexpected max_index: {} in output array: {} -- there should only be 4 possible actions".format(max_index, out))


