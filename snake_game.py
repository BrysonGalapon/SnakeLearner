from Snake import Snake
from Action import Action

width = 10
height = 10
snake = Snake(width=width, height=height)

while snake.alive():
    print(snake)
    inp = input()
    if inp == "q":
        inp = ""
        break
    elif inp == "1":
        inp = Action.UP
    elif inp == "2":
        inp = Action.DOWN
    elif inp == "3":
        inp = Action.LEFT
    elif inp == "4":
        inp = Action.RIGHT
    else:
        raise Exception("WAAA, BAD INPUT")

    snake.step(inp)
