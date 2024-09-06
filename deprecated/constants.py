import numpy as np
import pygame
__all__ = ["screen_width", "screen_height", "pheromone_grid", "evaporation_rate", "diffusion_rate", "agent_count", "pheromones_surface"]

screen_width, screen_height = 320, 360
# Define the pheromone grid
pheromone_grid = np.zeros((screen_width, screen_height))

# Initialize the pheromones surface
pheromones_surface = pygame.Surface((screen_width, screen_height))
pheromones_surface.set_colorkey((0, 0, 0))

# Parameters for pheromone diffusion and evaporation
evaporation_rate = 0.97
diffusion_rate = 0.5
pheromone_weight = 1

#agent settings
agent_count = 2000
