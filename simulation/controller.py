
import numpy as np
class IICController:
    def __init__(self,k_steer=0.32,k_speed=0.28):
        self.k_steer=k_steer; self.k_speed=k_speed
    def action(self,trajectory,target_speed,current_speed):
        if trajectory is None or len(trajectory)<2:
            return 0.0,-1.0
        idx=min(len(trajectory)-1,max(2,int(len(trajectory)*0.18)))
        x,y=trajectory[idx]
        steer=float(np.clip(self.k_steer*y/max(x,2.0),-1,1))
        speed_error=float(target_speed-current_speed)
        tb=float(np.clip(self.k_speed*speed_error,-1,1))
        return steer,tb
