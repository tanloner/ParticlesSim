from deprecated.constants import *
from agent import Agent, Species
import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter, convolve
from settings import Settings, SpawnMode
import random
import math
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor


def hash_state(state: int) -> int:
    state ^= 2747636419
    state *= 2654435769
    state ^= state >> 16
    state *= 2654435769
    state ^= state >> 16
    state *= 2654435769
    return state & 0xFFFFFFFF


def scale_to_range_01(state: int) -> float:
    return state / 4294967295.0

class Map:
    def __init__(self, settings: Settings):
        self.pheromone_grid = np.zeros((settings.width, settings.height, len(settings.species)))
        self.agents = []
        self.settings = settings

    def initialize_agents(self):
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
                    start_pos, angle = self._random_point_in_circle(center, 50)
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

    def spawn_agent(self, pos: tuple, angle: float, species: Species):
        self.agents.append(Agent(pos, angle, species))

    def deposit_pheromone(self, pos: tuple):
        pheromone_grid[*pos] += 1

    def diffuse_and_evaporate_using_convolve(self):
        kernel = np.array(
            [[0, diffusion_rate, 0], [diffusion_rate, 1 - 4 * diffusion_rate, diffusion_rate], [0, diffusion_rate, 0]])
        self.pheromone_grid = convolve(self.pheromone_grid, kernel, mode='constant')
        self._evaporate()

    def diffuse_and_evaporate_using_uniform_filter(self):
        blurred = uniform_filter(self.pheromone_grid, size=3, mode='constant')
        self.pheromone_grid = blurred * evaporation_rate

    def diffuse_and_evaporate_using_gaussian(self):
        blurred = gaussian_filter(self.pheromone_grid, sigma=self.settings.diffusion_rate)
        self.pheromone_grid = blurred * evaporation_rate

    def _evaporate(self):
        self.pheromone_grid *= evaporation_rate

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
            self.pheromone_grid[x, y] = np.minimum(1, self.pheromone_grid[x, y] + species[i].sense_weight * deltaTime)

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
        sensor_area = self.pheromone_grid[
                      sensor_x - agent.species.sensor_size: sensor_x + agent.species.sensor_size + 1,
                      sensor_y - agent.species.sensor_size: sensor_y + agent.species.sensor_size + 1]

        return np.sum(sensor_area * agent.species.sense_weight)

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

