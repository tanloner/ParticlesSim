import numpy as np

screen_width, screen_height = 320, 360
# Define the pheromone grid
pheromone_grid = np.zeros((screen_width, screen_height))

# Parameters for pheromone diffusion and evaporation
evaporation_rate = 0.97
diffusion_rate = 0.5
