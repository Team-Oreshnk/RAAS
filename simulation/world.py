
import math, random
from dataclasses import dataclass

@dataclass
class SimActor:
    track_id:int
    cls:str
    s:float
    lateral:float
    vs:float=0.0
    vl:float=0.0
    width:float=1.0
    length:float=2.0
    visible:bool=True
    reveal_s:float=0.0
    confidence:float=0.95

@dataclass
class Ego:
    s:float=0.0
    lateral:float=0.0
    speed:float=4.8
    steering:float=0.0
    throttle:float=0.0
    brake:float=0.0

CLASS_DIMS={
 "car":(1.8,4.5),"motorcycle":(.8,2.0),"bus":(2.5,10.0),
 "truck":(2.5,7.0),"autorickshaw":(1.4,2.8),"person":(.7,.7),
 "bicycle":(.7,1.8),"pothole":(1.6,1.2)
}

class IndianRoadSim:
    """Continuous presentation scenario: left bend -> right bend with staged sudden hazards."""
    def __init__(self,scenario="showcase",seed=11):
        self.rng=random.Random(seed)
        self.reset(scenario)

    def reset(self,scenario="showcase"):
        self.scenario=scenario
        self.time=0.0
        self.event=False
        self.event_text=""
        self.last_stage=""
        self.ego=Ego(speed=4.8)
        # The actor coordinates are road-relative: s = forward along road, lateral = left.
        self.actors=[
            # Traffic already visible, moving with the road.
            SimActor(101,"car",34,-2.7,-1.2,0,1.8,4.5,True,0),
            SimActor(102,"autorickshaw",48,2.7,-1.0,0,1.4,2.8,True,0),
            SimActor(103,"motorcycle",63,-2.5,-0.7,0,.8,2.0,True,0),

            # Sudden pedestrian after the first (left) bend.
            SimActor(201,"person",24,2.9,0,-1.05,.7,.7,False,17),

            # Sudden oncoming auto deeper in the right bend.
            SimActor(202,"autorickshaw",67,-2.8,-2.0,0,1.4,2.8,False,55),

            # A second moving vehicle creates a constrained free-space decision.
            SimActor(203,"car",78,2.2,-1.1,0,1.8,4.5,False,60),

            # Pothole appears later, offset from the current path.
            SimActor(204,"pothole",88,1.1,0,0,1.6,1.2,False,74),

            # Pedestrian cut-in later.
            SimActor(205,"person",105,-4.2,0,1.6,.7,.7,False,91),
        ]
        return self.snapshot()

    def snapshot(self):
        return [a for a in self.actors if a.visible or self.ego.s >= a.reveal_s]

    def update(self,steering,throttle_brake,dt,emergency_stop=False):
        self.time += dt
        self.ego.steering=max(-1,min(1,float(steering)))
        tb=max(-1,min(1,float(throttle_brake)))
        self.ego.throttle=max(tb,0); self.ego.brake=max(-tb,0)
        # Smooth longitudinal dynamics.
        accel=2.25*self.ego.throttle-6.2*self.ego.brake-0.20
        if emergency_stop:
            self.ego.speed=0.0
        else:
            self.ego.speed=max(0,min(9.5,self.ego.speed+accel*dt))
        self.ego.s += self.ego.speed*dt
        # Steering is an input to a simple bicycle-like lateral response.
        steer_gain=0.72+0.045*self.ego.speed
        self.ego.lateral += self.ego.steering*steer_gain*dt
        self.ego.lateral=max(-4.7,min(4.7,self.ego.lateral))

        for a in self.actors:
            a.s += a.vs*dt
            a.lateral += a.vl*dt

        for a in self.actors:
            if (not a.visible) and self.ego.s >= a.reveal_s:
                a.visible=True
                self.event=True
                self.event_text=f"SUDDEN {a.cls.upper()} DETECTED"
        return self.snapshot()
