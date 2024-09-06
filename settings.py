import json
from enum import Enum
from agent import Species

class SpawnMode(Enum):
    RANDOM = 0
    CENTER = 1
    INWARD_CIRCLE = 2
    OUTWARD_CIRCLE = 3
    LINE = 4
    RANDOM_CIRCLE = 5

class Settings:
    # Default Settings
    width: int = 700
    height: int = 700
    num_agents: int = 1000
    spawn_mode: int = SpawnMode.OUTWARD_CIRCLE
    species: list = [Species(1, 0.1, color=(0, -255, -255), sense_weight=(1,-1), deposit_weight=(1,0)), Species(1, 0.1, color=(-255, 0, 0), sense_weight=(-1, 1), deposit_weight=(0, 1))]
    species_probabilities: list = [100,1]
    evaporation_rate = 0.98
    diffusion_rate = 0.4

    def __init__(self):
        pass

    @classmethod
    def from_dict(cls, settings_dict: dict):
        instance = cls()
        for key, value in settings_dict.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        return instance

    @classmethod
    def from_file(cls, file_path: str):
        with open(file_path, 'r') as f:
            settings_dict = json.load(f)
        return cls.from_dict(settings_dict)
