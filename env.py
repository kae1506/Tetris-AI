import pygame
import random
import math
import numpy as np
import shapes_list as sh
import time
import cv2

# creating the data structure for pieces
# setting up global vars
# functions
# - create_grid
# - draw_grid
# - draw_window
# - rotating shape in main
# - setting up the main

"""
10 x 20 square grid
shapes: S, Z, I, O, J, L, T
represented in order by 0 - 6
"""

pygame.font.init()

# GLOBALS VARS
s_width = 400
s_height = 600
block_size = 20
rows, cols = int(s_width/block_size), int(s_height/block_size)

class Piece:
    """
    Class for each piece.
    Has functionality.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.shape_func = np.random.choice(sh.shapes)
        self.rotations = self.shape_func(self.x, self.y)
        ind = random.randint(0,len(self.rotations)-1)
        self.shape = self.rotations[ind]
        self.index = ind
        self.color = sh.shape_colors[self.index]
        self.height = self.shape[-1]["vals"][1]+1 - self.shape[-1]["vals"][0]

    def refresh_shapes(self):
        self.rotations = self.shape_func(self.x, self.y)
        self.shape = self.rotations[self.index]

    def check_rotate(self,grid):
        dummy = self.shape.copy()
        dummy_y = self.y
        dummy_x = self.x
        index = self.index+1
        if index > len(self.rotations)-1:
            index = 0
        dummy = self.rotations[index]

        least_dist = 0
        for pos in dummy[::-1]:
            y = pos["vals"][1]
            x = pos["vals"][0]
            least_dist = max(least_dist, ((y+1)-rows))


        dummy_y -= least_dist

        self.rotations = self.shape_func(dummy_x, dummy_y)
        dummy = self.rotations[index]

        for i in grid:
            for j in dummy[::-1]:
                if i[0] == j["vals"][0] and i[1] == j["vals"][1]:
                    return False

        return True

    def check_move_down(self, grid):
        dummy = self.shape.copy()
        dummy_y = self.y
        dummy_x = self.x
        index = self.index

        dummy_y += 1

        self.rotations = self.shape_func(dummy_x, dummy_y)
        dummy = self.rotations[index].copy()

        for i in grid:
            for j in dummy[::-1]:
                if i[0] == j["vals"][0] and i[1] == j["vals"][1]:
                    return False

        for pos in self.shape[::-1]:
            if pos["vals"][1] >= ((s_height - block_size) / block_size):
                return False

        return True

    def show(self, surface):
        for i in self.shape:
            x = i["vals"][0]
            y = i["vals"][1]

            pygame.draw.rect(surface, self.color, (x*block_size, y*block_size, block_size, block_size))

        x = self.shape[-1]["vals"][0]
        y =  self.shape[-1]["vals"][1]

        # pygame.draw.rect(surface, (255,255,255), (x*block_size, y*block_size, block_size, block_size))


    def rotate(self,grid):
        self.index = self.index+1
        if self.index > len(self.rotations)-1:
            self.index = 0
        self.shape = self.rotations[self.index]

        least_dist = 0
        for pos in self.shape[::-1]:
            y = pos["vals"][1]
            x = pos["vals"][0]
            least_dist = max(least_dist, ((y+1)-rows))

        self.y -= least_dist

        self.refresh_shapes()

    def check_move(self, grid, num):
        dummy = self.shape.copy()
        dummy_y = self.y
        dummy_x = self.x
        index = self.index

        dummy_y += num

        self.rotations = self.shape_func(dummy_x, dummy_y)
        dummy = self.rotations[index].copy()

        for i in grid:
            for j in dummy[::-1]:
                if i[0] == j["vals"][0] and i[1] == j["vals"][1]:
                    return False
        
        return True

    def move_down(self):
        self.y += 1
        self.refresh_shapes()

    def offscreen(self):
        for i in self.shape:
            if i["vals"][0] < 0 or (i["vals"][0]+1)>rows:
                return True
        return False

class TetrisEnv:
    def __init__(self):
        self.clock = pygame.time.Clock()
        self.win = pygame.display.set_mode((s_width, s_height))
        self.render = True
        self.hole_score = -1
        self.high_reward = 5

        self.piece = Piece(random.randint(0, rows - 1), 0)
        self.grid = []
        self.done = False

        self.reset()

    def reset(self):
        self.piece = Piece(random.randint(0,rows-1), 0)
        self.grid = []
        self.done = False

    def add_to_grid(self):
        for pos in self.piece.shape:
            x = pos["vals"][0]
            y = pos["vals"][1]

            self.grid.append([x,y])
        self.piece = Piece(random.randint(0, rows-5), 0)

    def check_how_many(self, array, val):
        cntr = 0
        for i in array:
            if i == val:
                cntr += 1
        return cntr

    def get_state(self,v):
        state = np.zeros((2, rows, cols))
        for i in self.grid:
            x = i[0]
            y = i[1]
            state[0][x][y] = 1

        for i in self.piece.shape:
            x = i["vals"][0]
            y = i["vals"][1]
            state[1][x-(rows-x)][y-(cols-y)] = 1

        return state


    def remove_row(self, row, gridy, gridx):
        top_ys = []
        for i in gridy:
            if i < row:
                top_ys.append(i)

        while not self.check_how_many(gridy, row) == 0:
            ind = gridy.index(row)
            del self.grid[ind]
            del gridy[ind]

        if len(top_ys) > 0:
            while len(top_ys) > 0:
                i = top_ys[-1]
                #breakpoint()
                ind = gridy.index(i)
                #breakpoint()
                self.grid[ind][1] += 1
                gridy[ind] = -math.inf
                del top_ys[-1]

                pygame.draw.rect(self.win, (255,0,0), (self.grid[ind][0]*block_size, self.grid[ind][1]*block_size, block_size, block_size))


    def check_grid(self):
        rew = 0
        gridx = [i[0] for i in self.grid]
        gridy = [i[1] for i in self.grid]
        for i in range(cols):
            if self.check_how_many(gridy, i) == rows:
                rew += self.high_reward
                self.remove_row(i, gridy, gridx)

        if 0 < rew < 4:
            rew = 1
        if rew >= 4:
            rew = 8

        return rew


    def game_over(self):

        for i in self.grid:
            if i[1] <= 0:
                return True
        return False

    def find_holes(self):
        for i in self.piece.shape[::-1]:
            i = i["vals"]
            top = self.check_how_many([i[1] for i in self.grid], i[1]-1) > 0
            bottom = self.check_how_many([i[1] for i in self.grid], i[1]+1) > 0
            right = self.check_how_many([i[0] for i in self.grid], i[0]+1) > 0
            left = self.check_how_many([i[0] for i in self.grid], i[0]-1) > 0

            if top and bottom and right and left:
                return self.hole_score

        return 1




    def play(self):

        self.win.fill((0, 0, 0))
        self.clock.tick(5)
        if self.piece.check_move_down(self.grid):
            self.piece.move_down()

        if not self.piece.check_move_down(self.grid):
            self.add_to_grid()

        # self.piece.rotate(self.grid)
        rem_p = False

        for pos in self.piece.shape[::-1]:
            if pos["vals"][1] >= ((s_height - block_size) / block_size):
                rem_p = True
            if rem_p:
                break

        if rem_p:
            self.add_to_grid()

        events = pygame.event.get()

        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    if self.piece.check_rotate(self.grid):
                        self.piece.rotate(self.grid)
                if event.key == pygame.K_LEFT:
                    if self.piece.check_move(self.grid, -1):
                        self.piece.x -= 1
                        self.piece.show(self.win)
                elif event.key == pygame.K_RIGHT:
                    if self.piece.check_move(self.grid, 1):
                        self.piece.x += 1
                        self.piece.show(self.win)
                elif event.key == pygame.K_DOWN:
                    if self.piece.check_move_down(self.grid):
                        self.piece.move_down()


        self.check_grid()

        if self.game_over():
            print("the pieces are above screen")
            print("SO BOO YA SUCK!!!")
            time.sleep(1)
            pygame.quit()
            print(self.get_state())
            quit()

        if self.piece.offscreen():
            print("u just drove another piece off")
            print("SO BOO YA SUCK!!!")
            time.sleep(1)
            pygame.quit()
            print(self.get_state())
            quit()

        self.piece.show(self.win)

        for piece in self.grid:
            x = piece[0]
            y = piece[1]
            pygame.draw.rect(self.win, (255, 255, 255), (x * block_size, y * block_size, block_size, block_size))
        pygame.display.update()

    def step(self, actions):
        action = actions.index(1)

        reward = 0
        info = {
            "Piece": self.piece.shape,
            "Boo": "Ya suck."
        }


        self.win.fill((0,0,0))
        self.clock.tick(200)
        if self.piece.check_move_down(self.grid):
            self.piece.move_down()

        if not self.piece.check_move_down(self.grid):
            self.add_to_grid()

        # self.piece.rotate(self.grid)
        rem_p = False

        for pos in self.piece.shape[::-1]:
            if pos["vals"][1] >= ((s_height-block_size)/block_size):
                    rem_p = True
            if rem_p:
                break
        
        if rem_p:
            self.add_to_grid()
            
        events = pygame.event.get()

        if action == 0:
            if self.piece.check_rotate(self.grid):
                self.piece.rotate(self.grid)
        if action == 1:
            if self.piece.check_move(self.grid, -1):
                self.piece.x -= 1
                self.piece.show(self.win)
        elif action == 2:
            if self.piece.check_move(self.grid, 1):
                self.piece.x += 1
                self.piece.show(self.win)
        elif action == 3:
            if self.piece.check_move_down(self.grid):
                self.piece.move_down()
        elif action == 4:
            pass

        reward += self.check_grid()

        if self.game_over():
            time.sleep(1)
            self.done = True

        if self.piece.offscreen():
            time.sleep(1)
            self.done = True

        if self.render:
            self.piece.show(self.win)

            for piece in self.grid:
                x = piece[0]
                y = piece[1]
                pygame.draw.rect(self.win, (255,255,255), (x*block_size, y*block_size, block_size, block_size))

        pygame.display.update()

        state = self.get_state(0) if not self.done else None
        reward = reward if not self.done else -1
        return state, reward, self.done, info


if __name__ == "__main__":
    game = TetrisEnv()
    done = False
    score = 0
    while not done:
        action = random.randint(0,3)
        a = [0 for i in range(4)]
        a[action] = 1
        state, reward, done, info = game.step(a)
        print(info)
        print(reward)
        score += reward

    print("Score: {}".format(score))
    while True:
        game.play()
