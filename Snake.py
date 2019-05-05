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
        # number of steps before snake starvs
        self.max_snake_starve = 2*(width*height)

    '''
    Returns a numpy array with 1 row and the following columns (in order):
        - isHeadGoingUp (0 or 1)
        - isHeadGoingDown (0 or 1)
        - isHeadGoingLeft (0 or 1)
        - isHeadGoingRight (0 or 1)
        - horizontal distance from head to apple
        - vertical distance from head to apple
        - upDistanceToObstacle
        - downDistanceToObstacle
        - leftDistanceToObstacle
        - rightDistanceToObstacle
        - length of snake
    '''
    def getCurrentState(self):
        head_x, head_y = self.snake_pos[0]
        apple_x, apple_y = self.apple_pos
        # ignore head and tail, since we can't move to hit head, and
        #   we can't hit the tail if there is a straight-line path to the tail
        #   on this turn
        snakeBody = set(self.snake_pos[1:-1])

        # direction state
        isHeadGoingUp = self.snake_dir == Action.UP
        isHeadGoingDown = self.snake_dir == Action.DOWN
        isHeadGoingLeft = self.snake_dir == Action.LEFT
        isHeadGoingRight = self.snake_dir == Action.RIGHT

        # horizontal distance from head to apple (positive dist -> apple is to the right)
        hdha = apple_x-head_x
        # horizontal distance from head to apple (positive dist -> apple is above)
        vdha = -1*(apple_y-head_y)

        # up distance to obstacle
        curr_x, curr_y = head_x, head_y
        while (curr_x, curr_y) not in snakeBody and curr_y > 0:
            curr_y -= 1
        udo = -1*(curr_y-head_y)

        # down distance to obstacle
        curr_x, curr_y = head_x, head_y
        while (curr_x, curr_y) not in snakeBody and curr_y < self.height-1:
            curr_y += 1
        ddo = -1*(curr_y-head_y)

        # left distance to obstacle
        curr_x, curr_y = head_x, head_y
        while (curr_x, curr_y) not in snakeBody and curr_x > 0:
            curr_x -= 1
        ldo = curr_x-head_x

        # left distance to obstacle
        curr_x, curr_y = head_x, head_y
        while (curr_x, curr_y) not in snakeBody and curr_x < self.width-1:
            curr_x += 1
        rdo = curr_x-head_x

        length = len(self.snake_pos)

        state_vec = [isHeadGoingUp,
                     isHeadGoingDown,
                     isHeadGoingLeft,
                     isHeadGoingRight,
                     hdha,
                     vdha,
                     udo,
                     ddo,
                     ldo,
                     rdo,
                     length]

        return np.array([state_vec])

    def score(self, length_weight=1000, starve_weight=-1, game_len_weight=-1):
        return length_weight*len(self.snake_pos) + starve_weight*self.lac + game_len_weight*self.game_counter

    def getCurrentActions(self):
        actions = []
        for act in Action:
            actions.append(act)
        return actions

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
        return not self.lose and self.lac < self.max_snake_starve

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
            # decrement y
            x_new = x
            if y == 0:
                # ran into top wall
                self.lose = True
                # pretend to wrap around
                y_new = self.height-1
            else:
                y_new = y-1
        elif self.snake_dir == Action.DOWN:
            # increment y
            x_new = x
            if y == self.height-1:
                # ran into bottom wall
                self.lose = True
                # pretend to wrap around
                y_new = 0
            else:
                y_new = y+1
        elif self.snake_dir == Action.LEFT:
            # decrement x
            y_new = y
            if x == 0:
                # ran into left wall
                self.lose = True
                # pretend to wrap around
                x_new = self.width-1
            else:
                x_new = x-1
        elif self.snake_dir == Action.RIGHT:
            # increment x
            y_new = y
            if x == self.width-1:
                # ran into left wall
                self.lose = True
                # pretend to wrap around
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


