import numpy as np

from constants import *


class SlimeAgent:
    def __init__(self, x, y):
        self.position = np.array([x, y], dtype=float)
        self.direction = np.random.rand() * 2 * np.pi  # Random direction in radians
        self.speed = 1.0  # Speed of movement

    def move(self):
        # position = self.position + np.array([np.cos(self.direction), np.sin(self.direction)]) * self.speed
        # if position[0] < 0 or position[0] >= screen_width or position[1] < 0 or position[1] >= screen_height:
        #    self.direction = np.random.rand() * 2 * np.pi
        #    self.move()
        # else:
        #    self.position = position
        # Update position based on the current direction
        self.position += np.array([np.cos(self.direction), np.sin(self.direction)]) * self.speed

        # Keep the agent within the screen bounds
        self.position[0] = self.position[0] % screen_width
        self.position[1] = self.position[1] % screen_height

    def update_direction(self, new_direction):
        self.direction = new_direction
