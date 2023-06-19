import pygame
class Apple():
    def __init__(self, x, y, block_size):
        self.position = (x, y)
        self.block_size = block_size
        self.color = (255, 0, 0)
    def draw(self, screen):
    
        pygame.draw.rect(screen, self.color, (self.position[0], self.position[1], self.block_size, self.block_size))