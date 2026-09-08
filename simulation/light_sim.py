
import math, random, time
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class Actor:
    track_id: int
    cls: str
    x: float          # lateral, +left
    y: float          # forward
    vx: float = 0.0
    vy: float = 0.0
    width: float = 1.0
    length: float = 2.0

@dataclass
class Ego:
    x: float = 0.0
    y: float = 0.0
    speed: float = 0.0
    steering: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0

class IIC2DSimulator:
    def __init__(self, width=1100, height=700, seed=7):
        self.width, self.height = width, height
        self.rng=random.Random(seed)
        self.road_half_width=6.0
        self.ego=Ego()
        self.actors=[]
        self.time=0.0
        self.scenario="pedestrian_crossing"
        self.reset(self.scenario)

    def reset(self, scenario=None):
        if scenario: self.scenario=scenario
        self.ego=Ego(speed=3.0)
        self.time=0.0
        if self.scenario=="pedestrian_crossing":
            self.actors=[
                Actor(1,"car", -2.5, 18, 0, 0, 1.8, 4.5),
                Actor(2,"person", 2.0, 11, -0.4, -0.5, .7, .7),
                Actor(3,"motorcycle",-3.2,27,.1,-.3,.8,2.0),
            ]
        elif self.scenario=="blocked_road":
            self.actors=[
                Actor(10,"car",0.0,18,0,0,1.8,4.5),
                Actor(11,"car",2.5,25,0,-.4,1.8,4.5),
            ]
        elif self.scenario=="overtake":
            self.actors=[
                Actor(20,"truck",1.8,23,0,-.5,2.5,7.0),
                Actor(21,"car",-2.8,35,.2,-.5,1.8,4.5),
            ]
        elif self.scenario=="unknown":
            self.actors=[
                Actor(30,"car",0.0,25,0,-.3,1.8,4.5),
            ]
        else:
            self.actors=[]
        return self.state()

    def update(self, steering, throttle_brake, dt=0.08):
        self.ego.steering=max(-1,min(1,float(steering)))
        tb=max(-1,min(1,float(throttle_brake)))
        if tb >= 0:
            self.ego.throttle=tb; self.ego.brake=0
        else:
            self.ego.throttle=0; self.ego.brake=-tb
        accel=2.2*self.ego.throttle - 5.5*self.ego.brake - 0.35
        self.ego.speed=max(0.0,min(13.0,self.ego.speed+accel*dt))
        # simple bicycle-ish visual motion
        self.ego.x += self.ego.steering * (0.7 + self.ego.speed*0.08)*dt
        self.time += dt
        for a in self.actors:
            a.x += a.vx*dt
            a.y += a.vy*dt
        return self.state()

    def state(self):
        return {"ego":self.ego, "actors":self.actors, "time":self.time}
