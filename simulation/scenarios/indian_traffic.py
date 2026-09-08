from __future__ import annotations
from simulation.world import SimWorld

def make_world(seed=7, traffic=70, pedestrians=20, road_width=10.0):
    return SimWorld(seed=seed, traffic_count=traffic, pedestrian_count=pedestrians, road_width=road_width)

def trigger(world: SimWorld, name: str):
    if name == "motorcycle_cut_in":
        a = next((x for x in world.actors if x.kind == "motorcycle"), None)
        if a: a.x = world.ego.x + 28; a.y = -6; a.vx = max(a.vx,8); a.vy = 3.5
    elif name == "pedestrian_emerge":
        a = next((x for x in world.actors if x.kind == "person"), None)
        if a: a.x = world.ego.x + 22; a.y = -6; a.vy = 2.0
    elif name == "rolling_object":
        a = next((x for x in world.actors if x.kind == "rolling_object"), None)
        if a: a.x = world.ego.x + 25; a.y = 5; a.vx = 3; a.vy = -5
    elif name == "sudden_stop":
        a = min((x for x in world.actors if x.kind in {"car","truck","bus"}), key=lambda x: abs(x.x-world.ego.x), default=None)
        if a: a.x = world.ego.x + 25; a.y = 0; a.vx = 0
