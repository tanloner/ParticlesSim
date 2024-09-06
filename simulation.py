import sys

import numpy as np
import pygame

from map import Map
from settings import Settings


class Simulation:

    def __init__(self):
        self.settings = Settings()
        self.pheromones_surface = pygame.Surface((self.settings.width, self.settings.height))
        self.pheromones_surface.set_colorkey((0, 0, 0))
        self.map = Map(self.settings)
        self.map.initialize_agents()

    def _init_pygame(self):
        pygame.init()
        screen = pygame.display.set_mode((self.settings.width, self.settings.height))
        pygame.display.set_caption('Initializing SlimeSim...')
        return screen

    def update_pheromone_visuals_beta(self):
        masks = self.map.pheromone_grid > 0 #700 700 2
        #intensities = self.map.pheromone_grid
        intensities = [np.minimum(self.map.pheromone_grid[:,:,index][masks[:,:,index]] * 255, 255).astype(np.int16) for index in range(masks.shape[2])]
        coords = np.argwhere(masks)
        if coords.size > 0:
            pixel_array = pygame.surfarray.pixels3d(self.pheromones_surface)
            for index, species in enumerate(self.settings.species):
                colors = species.calculate_colors(intensities[index])
                co_coords = coords[coords[:, 2] == index]
                pixel_array[co_coords[:, 0], co_coords[:, 1]] = colors




    def update_pheromone_visuals(self):
        # Mask for the first dimension (where dim=0 is to be red and dim=1 is to be blue)
        red_mask = self.map.pheromone_grid[:, :, 0] > 0
        # blue_mask = self.pheromones_grid[:, :, 1] > 0

        # Red intensities for the first dimension
        red_intensities = np.minimum(self.map.pheromone_grid[red_mask, 0] * 255, 255).astype(np.uint8)
        red_coords = np.argwhere(red_mask)
        # Blue intensities for the second dimension
        # blue_intensities = np.minimum(self.pheromones_grid[blue_mask, 1] * 255, 255).astype(np.uint8)
        # blue_coords = np.argwhere(blue_mask)

        # Only update when changes occur
        if red_coords.size > 0:  # or blue_coords.size > 0:
            pixel_array = pygame.surfarray.pixels3d(self.pheromones_surface)

            # Update red pixels
            if red_coords.size > 0:
                red_colors = np.column_stack(
                    (red_intensities, np.zeros_like(red_intensities), np.zeros_like(red_intensities)))
                pixel_array[red_coords[:, 0], red_coords[:, 1]] = red_colors

            # Update blue pixels
            #    if blue_coords.size > 0:
            #        blue_colors = np.column_stack(
            #            (np.zeros_like(blue_intensities), np.zeros_like(blue_intensities), blue_intensities))
            #        pixel_array[blue_coords[:, 0], blue_coords[:, 1]] = blue_colors

            del pixel_array

    def run(self):
        running = True
        screen = self._init_pygame()
        pygame.display.set_caption("SlimeSim")
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill((0, 0, 0))
            self.update_pheromone_visuals_beta()
            screen.blit(self.pheromones_surface, dest=(0, 0))
            # for id, agent in enumerate(self.agents):
            #   self.update(agent, 1, id)
            self.map.update_agents(1)
            self.map.diffuse_and_evaporate_using_gaussian()
            pygame.display.flip()

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    sim = Simulation()
    sim.run()
