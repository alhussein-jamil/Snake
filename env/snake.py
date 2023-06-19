import pygame 
import numpy as np 

class Snake:
    def __init__(self, screen_width, screen_height, block_size):
        self.head = (0, 0)
        self.body = [(0, 0)]
        self.direction = (1, 0)
        self.grow = False
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.block_size = block_size


    def move(self):

        self.body.append((self.body[-1][0] + self.direction[0] * self.block_size, self.body[-1][1] + self.direction[1] * self.block_size))
        if self.grow:
            self.grow = False
        else:
            self.body.pop(0)
        self.head = self.body[-1]


    def change_direction(self, dir ):
        direction = np.argmax(dir)

        if ( direction == 0) and (self.direction != (0, 1) or len(self.body) == 1):
            self.direction = (0, -1)
        elif (direction == 1) and (self.direction != (0, -1) or len(self.body) == 1):
            self.direction = (0, 1)
        elif (direction == 2) and (self.direction != (1, 0) or len(self.body) == 1):
            self.direction = (-1, 0)
        elif ( direction == 3) and (self.direction != (-1, 0) or len(self.body) == 1):
            self.direction = (1, 0)


    def draw(self, surface):
        for block in self.body:
            rect = pygame.Rect(block[0], block[1], self.block_size, self.block_size)
            pygame.draw.rect(surface, (0, 255, 0), rect)
