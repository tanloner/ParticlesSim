import pygame
import sys
from slime_agent import SlimeAgent
from constants import screen_width, screen_height, evaporation_rate, diffusion_rate, pheromone_grid
import numpy as np
import scipy.ndimage
from scipy.ndimage import uniform_filter, gaussian_filter
import random
import matplotlib.pyplot as plt

# Initialize Pygame
pygame.init()

screen = pygame.display.set_mode((screen_width, screen_height))

# Set the window title
pygame.display.set_caption('Slime Simulation')

agents = [SlimeAgent(random.randint(0, screen_width), random.randint(0, screen_height)) for _ in range(2000)]

def deposit_pheromone(agent):
    x, y = int(agent.position[0]), int(agent.position[1])
    pheromone_grid[x, y] += 1

def diffuse_and_evaporate_using_convolve():
    global pheromone_grid
    
    kernel = np.array([[0, diffusion_rate, 0],
                       [diffusion_rate, 1 - 4 * diffusion_rate, diffusion_rate],
                       [0, diffusion_rate, 0]])
    pheromone_grid = scipy.ndimage.convolve(pheromone_grid, kernel, mode='constant')
    pheromone_grid *= evaporation_rate

def diffuse_and_evaporate():
    global pheromone_grid
    blurred = uniform_filter(pheromone_grid, size=3, mode='constant')
    pheromone_grid = blurred * evaporation_rate

def diffuse_and_evaporate_using_gaussian():
    global pheromone_grid
    blurred = gaussian_filter(pheromone_grid, sigma=0.2)
    pheromone_grid = blurred * evaporation_rate

def evaporate():
    global pheromone_grid
    pheromone_grid *= evaporation_rate

def sense_pheromones(agent):
    # Positions for forward, left, and right sensors
    forward_x = int(agent.position[0] + np.cos(agent.direction) * 3)
    forward_y = int(agent.position[1] + np.sin(agent.direction) * 3)

    left_x = int(agent.position[0] + np.cos(agent.direction + np.pi / 4) * 3)
    left_y = int(agent.position[1] + np.sin(agent.direction + np.pi / 4) * 3)

    right_x = int(agent.position[0] + np.cos(agent.direction - np.pi / 4) * 3)
    right_y = int(agent.position[1] + np.sin(agent.direction - np.pi / 4) * 3)

    # Sense pheromone levels at each sensor
    forward_pheromone = pheromone_grid[forward_x % screen_width, forward_y % screen_height]
    left_pheromone = pheromone_grid[left_x % screen_width, left_y % screen_height]
    right_pheromone = pheromone_grid[right_x % screen_width, right_y % screen_height]

    # Adjust direction based on the pheromone concentration
    if forward_pheromone > left_pheromone and forward_pheromone > right_pheromone:
        agent.update_direction(agent.direction)  # Keep moving forward
    elif left_pheromone > right_pheromone:
        agent.update_direction(agent.direction + 0.1)  # Turn slightly left
    else:
        agent.update_direction(agent.direction - 0.1)  # Turn slightly right

    # Add some randomness to avoid getting stuck
    agent.update_direction(agent.direction + (np.random.rand() - 0.5) * 0.1)



# Initialize the pheromones surface
pheromones_surface = pygame.Surface((screen_width, screen_height))
pheromones_surface.set_colorkey((0, 0, 0))  # Make black transparent

def update_pheromone_visuals_with_loops():
    # Only update the pheromones surface where needed
    global pheromone_grid
    for x in range(screen_width):
        for y in range(screen_height):
            pheromone_intensity = pheromone_grid[x, y]
            if pheromone_intensity > 0:
                intensity = min(int(pheromone_intensity * 255), 255)
                pheromones_surface.set_at((x, y), (intensity, 100, 100))

def update_pheromone_visuals():
    global pheromone_grid
    # Create a mask of non-zero pheromone intensities
    mask = pheromone_grid > 0

    # Calculate intensities only for non-zero elements
    intensities = np.minimum(pheromone_grid[mask] * 255, 255).astype(np.uint8)

    # Get coordinates of non-zero elements
    coords = np.argwhere(mask)

    # Create color array
    colors = np.column_stack((intensities, np.full_like(intensities, 0), np.full_like(intensities, 0)))

    # Get the pixel array from the surface
    pixel_array = pygame.surfarray.pixels3d(pheromones_surface)

    # Update pheromones surface
    pixel_array[coords[:, 0], coords[:, 1]] = colors

    # Delete the pixel array reference
    del pixel_array

    # Update the surface
    #pygame.display.update(pheromones_surface.get_rect())

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))

    # Draw pheromones
    update_pheromone_visuals()
    #configure numpy so it prints the full array
    #np.set_printoptions(threshold=sys.maxsize)
    #print(pheromone_grid)
    screen.blit(pheromones_surface, (0, 0))

    # Update and draw each agent
    for agent in agents:
        #agent.sense_pheromones()
        sense_pheromones(agent)
        agent.move()
        #agent.deposit_pheromone()
        deposit_pheromone(agent)
        #pygame.draw.circle(screen, (0, 255, 0), agent.position.astype(int), 2)
    #diffuse_and_evaporate()
    diffuse_and_evaporate_using_gaussian()
    #plt.imshow(pheromone_grid, cmap='hot', interpolation='nearest')
    #plt.colorbar()
    #plt.show()  # This is optional; use it if you want to see the heatmap before saving
    #evaporate()

    pygame.display.flip()

pygame.quit()
sys.exit()