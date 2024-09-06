import numpy as np

class SensorSettings:
    def __init__(self):
        self.sensor_angle_spacing = 1.1
        self.sensor_offset_distance = 2
        self.sensor_size = 3

class Species(SensorSettings):
    def __init__(self, move_speed: float = 0.0, turn_speed: float = 0.0, color: tuple[int, int, int] = (255, 255, 255),
                 sense_weight=(1), deposit_weight=(1)):
        super().__init__()
        self.move_speed = move_speed
        self.turn_speed = turn_speed
        self.color = color
        self.sense_weight = sense_weight
        self.deposit_weight = deposit_weight

    def calculate_colors(self, intensities: np.ndarray):
        red = intensities + self.color[0]
        red = (red/40)**3
        red = red.clip(0, 255)

        green = intensities + self.color[1]
        green = green.clip(0, 255)
        blue = intensities + self.color[2]
        blue = blue.clip(0, 255)
        c = np.column_stack(
            (red, green, blue))
        return c



    def __repr__(self) -> str:
        return f"Species(move_speed={self.move_speed}, turn_speed={self.turn_speed}, color={self.color})"

    @property
    def move_speed(self) -> float:
        return self._move_speed

    @move_speed.setter
    def move_speed(self, speed: float):
        self._move_speed = speed

    @property
    def turn_speed(self) -> float:
        return self._turn_speed

    @turn_speed.setter
    def turn_speed(self, speed: float):
        self._turn_speed = speed

    @property
    def color(self) -> tuple[int, int, int]:
        return self._color

    @color.setter
    def color(self, color: tuple[int, int, int]):
        #if any(c < 0 or c > 255 for c in color):
        #    raise ValueError("Each color component must be between 0 and 255")
        self._color = color

import numpy as np

class Agent:
    def __init__(self, x = 0, y = 0, angle = 0, species = None):
        self.pos = np.array([x, y], dtype=float)
        self.angle = angle
        self.species = species



