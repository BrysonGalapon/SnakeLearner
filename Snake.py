import random
import sys
from Action import Action
import numpy as np

class Snake():
    def __init__(self, width, height):
        self.text = "O"
        # whether we lost the game or not
        self.lose = False
        # set width and length of snake screen
        self.width = width
        self.height = height
        # count number of moves made
        self.game_counter = 0
        # positions where snake is
        self.snake_pos = [(2,0),(1,0),(0,0)]
        # current direction snake is heading
        self.snake_dir = Action.DOWN
        # current apple position
        self.apple_pos = self.random_apple_pos()
        # time since last apple capture
        self.lac = 0

    def getCurrentState(self):
        pass

    def score(self):
        pass

    def getCurrentActions(self):
        actions = []
        for act in Action:
            actions.append(act)
        return actions

    def perform(self, action):
        if self.step(action):
            return (1.0/ ( 0.1 + self.headDist()) ) + (10.0*self.snake_length() - 2e-2*self.lac)
        else:
            return -1*sys.maxsize

    # Manhattan distance to apple
    def headDist(self):
        head_x, head_y = self.snake_pos[0]
        app_x, app_y = self.apple_pos

        return abs(head_x-app_x) + abs(head_y-app_y)

    def snake_length(self):
        return len(self.snake_pos)

    def counter(self):
        return self.game_counter

    def alive(self):
        return not self.lose

    def step(self, inp):
        # don't do anything if we lost
        if self.lose:
            return False

        # change direction of snake, if given input
        if inp in Action:
            self.snake_dir = inp
        
            # move snake forward in current direction
            self.move_snake()

            # increment game counter
            self.game_counter += 1
            return True
        else:
            return False

    def move_snake(self):
        # save the tail position, just in case we eat an apple
        tail = self.snake_pos[-1]

        # update snake positions

        # calculate new head, wrapping around if needed
        head = self.snake_pos[0]
        new_head = self.update_pos(head)
        # prepend new head to positions
        self.snake_pos.insert(0, new_head)
        # remove tail of snake
        del self.snake_pos[-1]

        # check if we ran into ourselves (end condition)
        if self.has_duplicates(self.snake_pos):
            self.lose = True
            return

        # check if snake just moved to eat apple
        head = self.snake_pos[0]            
        if head == self.apple_pos:
            # increase length of snake
            self.snake_pos.append(tail)
            # place apple in a new location
            self.apple_pos = self.random_apple_pos()
            # reset apple counter
            self.lac = 0
        else:
            # didn't eat apple -- increment counter
            self.lac += 1

    def update_pos(self, pos):
        (x, y) = pos
        if self.snake_dir == Action.UP:
            # decrement y, wrapping around if neded
            x_new = x
            if y == 0:
                y_new = self.height-1
            else:
                y_new = y-1
        elif self.snake_dir == Action.DOWN:
            # increment y, wrapping around if neded
            x_new = x
            if y == self.height-1:
                y_new = 0
            else:
                y_new = y+1
        elif self.snake_dir == Action.LEFT:
            # decrement x, wrapping around if neded
            y_new = y
            if x == 0:
                x_new = self.width-1
            else:
                x_new = x-1
        elif self.snake_dir == Action.RIGHT:
            # increment x, wrapping around if neded
            y_new = y
            if x == self.width-1:
                x_new = 0
            else:
                x_new = x+1
        return (x_new, y_new)

    def has_duplicates(self, A):
        # A has duplicates if the set version of it is smaller in size
        return len(set(A)) < len(A)

    def random_apple_pos(self):
        # randomize apple position, until we find a spot where the snake isnt
        apple_x = random.randint(0, self.width-1)
        apple_y = random.randint(0, self.height-1)
        while (apple_x, apple_y) in self.snake_pos:
            apple_x = random.randint(0, self.width-1)
            apple_y = random.randint(0, self.height-1)
        return (apple_x, apple_y)

    def getChar(self, i):
        if i >= len(self.text):
            return "."
        else:
            return self.text[i]

    def __str__(self):
        # 2D array to represent board
        board = [[" "]*self.width for i in range(self.height)]

        # insert apple into board
        apple_x, apple_y = self.apple_pos
        board[apple_y][apple_x] = "*"

        # insert snake into board
        for i, (x,y) in enumerate(self.snake_pos):
            char = self.getChar(i)
            board[y][x] = char

        # translate board to string
        output = "Enter 1,2,3,4 to move UP, DOWN, LEFT, or RIGHT. Press q to exit. \n"
        output += "-"*(self.width+2) + "\n"
        for y in range(len(board)):
            output += "|"
            for x in range(len(board[0])):
                output += board[y][x]
            # break to new line
            output += "|\n"
        output += "-"*(self.width+2)
        return output


