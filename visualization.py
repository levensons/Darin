import numpy as np
import random
import pygame
from tkinter import *

root = Tk()

size = 225
sz = 15
mx_x = size // sz + 1
mx_y = size // sz + 1

canv = Canvas(root, width=size, height=size, bg='green')
canv.pack()


def make_move(game, turn, side):
    turn[0] -= 1
    turn[1] -= 1
    if turn[0] < 0 or turn[1] < 0 or turn[0] > 14 or turn[1] > 14:
        print('Wrong coordinates!')
        return -1

    game[turn[0], turn[1]] = side
    return 0


def show(turn, player):
    if player == 1:
        canv.create_rectangle(sz * turn[1], sz * (turn[0] + 1), sz * (turn[1] + 1), sz * turn[0], fill="black")
    else:
        canv.create_rectangle(sz * turn[1], sz * (turn[0] + 1), sz * (turn[1] + 1), sz * turn[0], fill="white")

    root.update()


def play():
    game = np.zeros(shape=(15, 15), dtype=np.int8)
    player = 1
    while 1:
        in_str = input()
        if in_str == 'End':
            print('Over!')
            break

        turn = list(map(int, input().split()))

        if make_move(game, turn, player) == -1:
            break

        print(turn)
        show(turn, player)
        player *= -1

    root.update()


for i in range(size // sz):
    canv.create_line(i * sz, 0, i * sz, size, width=0.5)

for i in range(size // sz):
    canv.create_line(0, i * sz, size, i * sz, width=0.5)

root.update()
play()

root.mainloop()