import math
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pygame
from scipy.ndimage import gaussian_filter, uniform_filter

from agent import Agent
from settings import Settings, SpawnMode


def hash_state(state: int) -> int:
    state ^= 2747636419
    state *= 2654435769
    state ^= state >> 16
    state *= 2654435769
    state ^= state >> 16
    state *= 2654435769
    return state & 0xFFFFFFFF  # Begrenze den Wert auf 32-Bit, um Überlauf zu vermeiden


def scale_to_range_01(state: int) -> float:
    return state / 4294967295.0


class Simulation:

    def __init__(self):
        self.settings = Settings()
        self.pheromones_surface = pygame.Surface((self.settings.width, self.settings.height))
        self.pheromones_surface.set_colorkey((0, 0, 0))
        self.pheromones_grid = np.zeros((self.settings.width, self.settings.height, len(self.settings.species)))

        self.agents = []
        for _ in range(self.settings.num_agents):
            center = self.settings.width/2, self.settings.height/2
            match Settings.spawn_mode:
                case SpawnMode.RANDOM:
                    start_pos = np.random.rand() * self.settings.width, np.random.rand() * self.settings.height
                    angle = np.random.rand() * 2 * np.pi
                case SpawnMode.CENTER:
                    start_pos = center
                    angle = np.random.rand() * 2 * np.pi
                case SpawnMode.INWARD_CIRCLE:
                    start_pos, angle = self._random_point_in_circle(center, 100)
                    angle -= 180
                case SpawnMode.OUTWARD_CIRCLE:
                    start_pos, angle = self._random_point_in_circle(center, 50)
                case SpawnMode.LINE:
                    start_pos = self.settings.width/2, np.random.rand() * self.settings.height
                    angle = np.random.rand() * 2 * np.pi
                case SpawnMode.RANDOM_CIRCLE:
                    start_pos, angle = self._random_point_in_circle(center, 50)
                    angle = np.random.rand() * 2 * np.pi
                case _:
                    start_pos = (0, 0)
                    angle = np.random.rand() * 2 * np.pi
            species = random.choices(self.settings.species, self.settings.species_probabilities)[0]
            self.agents.append(Agent(*start_pos, angle, species))

    def _random_point_in_circle(self, center, radius):
        r = radius * math.sqrt(random.uniform(0, 1))
        return self._random_point_on_circle(center, r)

    def _random_point_on_circle(self, center, radius):
        angle = random.uniform(0, 2*math.pi)
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        return (x, y), angle

    def _init_pygame(self):
        pygame.init()
        screen = pygame.display.set_mode((self.settings.width, self.settings.height))
        pygame.display.set_caption('Slime Simulation')
        return screen

    # Update-Funktion
    def update(self, agent, deltaTime, id):
        time = datetime.now().timestamp()
        pos = agent.pos
        width = self.settings.width
        height = self.settings.height
        # Random Hash
        random = hash_state(int(pos[1] * width + pos[0] + hash_state(id + int(time * 100000))))

        # Sensordaten auswerten
        sensor_angle_rad = agent.species.sensor_angle_spacing
        weight_forward = self.sense(agent, 0, width, height)
        weight_left = self.sense(agent, sensor_angle_rad, width, height)
        weight_right = self.sense(agent, -sensor_angle_rad, width, height)

        random_steer_strength = scale_to_range_01(random)
        turn_speed = agent.species.turn_speed * 2 * np.pi

        # Steuerungslogik basierend auf den Sensordaten
        if weight_forward > weight_left and weight_forward > weight_right:
            agent.angle += 0
        elif weight_forward < weight_left and weight_forward < weight_right:
            agent.angle += (random_steer_strength - 0.5) * 2 * turn_speed * deltaTime
        elif weight_right > weight_left:
            agent.angle -= random_steer_strength * turn_speed * deltaTime
        elif weight_left > weight_right:
            agent.angle += random_steer_strength * turn_speed * deltaTime

        # Position aktualisieren
        direction = np.array([np.cos(agent.angle), np.sin(agent.angle)])
        new_pos = agent.pos + direction * deltaTime * agent.species.move_speed

        # Position begrenzen und bei Randkollision neuen Zufallswinkel wählen
        if new_pos[0] < 0 or new_pos[0] >= width or new_pos[1] < 0 or new_pos[1] >= height:
            random = hash_state(random)
            random_angle = scale_to_range_01(random) * 2 * np.pi

            new_pos[0] = min(width - 1, max(0, new_pos[0]))
            new_pos[1] = min(height - 1, max(0, new_pos[1]))
            agent.angle = random_angle
        else:
            coord = tuple(new_pos.astype(int))
            old_trail = self.pheromones_grid[coord]
            self.pheromones_grid[coord] = min(1, old_trail + agent.species.sense_weight * deltaTime)

        agent.pos = new_pos

    def update_agents(self, deltaTime, id_offset=0, agents=None):
        # Zeitstempel und Breite/Höhe des Gitters
        if agents is None:
            agents = self.agents
        time = datetime.now().timestamp()
        width, height = self.settings.width, self.settings.height
        # Extrahiere alle Positionen, Winkel und Arten der Agenten in Arrays
        positions = np.array([agent.pos for agent in agents])
        angles = np.array([agent.angle for agent in agents])
        species = np.array([agent.species for agent in agents])

        # Berechne Hashes für alle Agenten
        ids = np.arange(len(agents))
        ids += id_offset
        hashed_states = hash_state((positions[:, 1] * width + positions[:, 0] + ids + int(time * 100000)).astype(int))
        # Sensordaten verarbeiten (hierfür können wir vektorisierte Sensorlogik nutzen)
        sensor_angle_rad = np.array([s.sensor_angle_spacing for s in species])
        weight_forward = np.array([self.sense(agent, 0, width, height) for agent in agents])
        weight_left = np.array(
            [self.sense(agent, angle, width, height) for agent, angle in zip(agents, sensor_angle_rad)])
        weight_right = np.array(
            [self.sense(agent, -angle, width, height) for agent, angle in zip(agents, sensor_angle_rad)])

        # Berechne zufällige Lenkstärken für alle Agenten
        random_steer_strength = scale_to_range_01(hashed_states)
        turn_speed = np.array([s.turn_speed for s in species]) * 2 * np.pi

        # Steuerungslogik in einem Schritt anwenden
        mask_forward = (weight_forward > weight_left) & (weight_forward > weight_right)
        mask_random = (weight_forward < weight_left) & (weight_forward < weight_right)
        mask_left = (weight_left > weight_forward) & (weight_left > weight_right) & (weight_right < weight_forward)
        mask_right = (weight_right > weight_forward) & (weight_right > weight_left) & (weight_left < weight_forward)
        # Agenten drehen abhängig von der Sensorinformation
        angles[mask_random] += (random_steer_strength[mask_random] - 0.5) * 2 * turn_speed[mask_random] * deltaTime
        angles[mask_left] += (random_steer_strength[mask_left])* turn_speed[mask_left] * deltaTime
        angles[mask_right] -= (random_steer_strength[mask_right])* turn_speed[mask_right] * deltaTime
        angles[mask_forward] += 0
        # Berechne die neuen Positionen
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        new_positions = positions + directions * deltaTime * np.array([s.move_speed for s in species]).reshape(-1, 1)

        # Randkollisionen überprüfen und Positionen anpassen
        out_of_bounds_mask = (new_positions[:, 0] < 0) | (new_positions[:, 0] >= width) | (new_positions[:, 1] < 0) | (
                new_positions[:, 1] >= height)
        if np.any(out_of_bounds_mask):
            hashed_states[out_of_bounds_mask] = hash_state(hashed_states[out_of_bounds_mask])
            random_angles = scale_to_range_01(hashed_states[out_of_bounds_mask]) * 2 * np.pi

            # Begrenze die Position auf die gültigen Bereiche
            new_positions[:, 0] = np.clip(new_positions[:, 0], 0, width - 1)
            new_positions[:, 1] = np.clip(new_positions[:, 1], 0, height - 1)

            # Setze neue Winkel für Agenten, die den Rand getroffen haben
            angles[out_of_bounds_mask] = random_angles

        # Pheromone aktualisieren (vektorisiert) und Koordinaten nach dem Runden erneut clippen
        coords = np.round(new_positions).astype(int)

        # Clip die Koordinaten, falls sie außerhalb der Grid-Grenzen geraten
        coords[:, 0] = np.clip(coords[:, 0], 0, width - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, height - 1)

        for i, (x, y) in enumerate(coords):
            self.pheromones_grid[x, y] = np.minimum(1, self.pheromones_grid[x, y] + species[i].sense_weight * deltaTime)

        # Aktualisiere die Positionen und Winkel der Agenten
        for i, agent in enumerate(agents):
            agent.pos = new_positions[i]
            agent.angle = angles[i]

    def sense(self, agent, sensor_angle_offset, width, height):
        sensor_angle = agent.angle + sensor_angle_offset
        sensor_dir = np.array([np.cos(sensor_angle), np.sin(sensor_angle)])
        sensor_pos = agent.pos + sensor_dir * agent.species.sensor_offset_distance

        sensor_x = np.clip(int(sensor_pos[0]), 0, width - 1)
        sensor_y = np.clip(int(sensor_pos[1]), 0, height - 1)

        # Nutze direkte NumPy-Operationen anstatt Schleifen
        sensor_area = self.pheromones_grid[
                      sensor_x - agent.species.sensor_size: sensor_x + agent.species.sensor_size + 1,
                      sensor_y - agent.species.sensor_size: sensor_y + agent.species.sensor_size + 1]

        return np.sum(sensor_area * agent.species.sense_weight)

    def diffuse_and_evaporate_using_gaussian(self):
        blurred = gaussian_filter(self.pheromones_grid, sigma=self.settings.diffusion_rate, axes=(0, 1))
        self.pheromones_grid = blurred * self.settings.evaporation_rate

    def update_pheromone_visuals1(self):
        # Mask and intensities calculation
        mask = self.pheromones_grid > 0
        intensities = np.minimum(self.pheromones_grid[mask] * 255, 255).astype(np.uint8)
        coords = np.argwhere(mask)

        # Only update when changes occur
        if coords.size > 0:
            colors = np.column_stack((intensities, np.zeros_like(intensities), np.zeros_like(intensities)))
            pixel_array = pygame.surfarray.pixels3d(self.pheromones_surface)
            pixel_array[coords[:, 0], coords[:, 1]] = colors
            del pixel_array

    def update_pheromone_visuals(self):
        # Mask for the first dimension (where dim=0 is to be red and dim=1 is to be blue)
        red_mask = self.pheromones_grid[:, :, 0] > 0
        #blue_mask = self.pheromones_grid[:, :, 1] > 0

        # Red intensities for the first dimension
        red_intensities = np.minimum(self.pheromones_grid[red_mask, 0] * 255, 255).astype(np.uint8)
        red_coords = np.argwhere(red_mask)

        # Blue intensities for the second dimension
        #blue_intensities = np.minimum(self.pheromones_grid[blue_mask, 1] * 255, 255).astype(np.uint8)
        #blue_coords = np.argwhere(blue_mask)

        # Only update when changes occur
        if red_coords.size > 0: #or blue_coords.size > 0:
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
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill((0, 0, 0))
            self.update_pheromone_visuals()
            screen.blit(self.pheromones_surface, dest=(0, 0))
            #for id, agent in enumerate(self.agents):
            #   self.update(agent, 1, id)
            self.update_agents(1)
            self.diffuse_and_evaporate_using_uniform_filter()
            pygame.display.flip()

        pygame.quit()
        print(self.forward)
        print(self.right)
        print(self.left)
        sys.exit()



    def update_agent_batch(self, agent_batch, deltaTime, id_offset):
        self.update_agents(deltaTime, id_offset, agent_batch)

    def update_agents_multiprocessing(self, deltaTime):
        # Teile die Agenten in Batches auf
        batch_size = 100
        agent_batches = [self.agents[i:i + batch_size] for i in range(0, len(self.agents), batch_size)]

        # Nutze ProcessPoolExecutor für Multiprocessing
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.update_agent_batch, batch, deltaTime, offset) for offset, batch in
                enumerate(agent_batches)]
            for future in futures:
                future.result()

    def update_agents_multithreaded(self, deltaTime):
        # Teile die Agenten in Batches auf, z.B. in Gruppen von 100
        batch_size = 100
        agent_batches = [self.agents[i:i + batch_size] for i in range(0, len(self.agents), batch_size)]

        # Nutze ThreadPoolExecutor für Multithreading
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.update_agent_batch, batch, deltaTime, offset) for offset, batch in
                enumerate(agent_batches)]

            # Warte auf die Fertigstellung aller Batches
            for future in futures:
                future.result()

    def run_multithread(self):
        running = True
        screen = self._init_pygame()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill((0, 0, 0))
            self.update_pheromone_visuals()
            screen.blit(self.pheromones_surface, dest=(0, 0))

            # Nutze Multithreading für die Aktualisierung der Agenten
            # self.update_agents_multithreaded(1)
            self.update_agents_multiprocessing(1)
            self.diffuse_and_evaporate_using_gaussian()
            pygame.display.flip()

        pygame.quit()

        sys.exit()

    def diffuse_and_evaporate_using_uniform_filter(self):
        blurred = uniform_filter(self.pheromones_grid, size=3, mode='constant')
        self.pheromone_grid = blurred * self.settings.evaporation_rate


if __name__ == "__main__":
    sim = Simulation()
    sim.run()  # sim.run_multithread()
