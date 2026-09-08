from dataclasses import dataclass

@dataclass
class SimConfig:
    dt: float = 0.05
    seed: int = 7
    road_width: float = 10.0
    world_length: float = 500.0
    traffic_count: int = 70
    pedestrian_count: int = 20
    max_speed: float = 16.0
    spawn_radius: float = 120.0
