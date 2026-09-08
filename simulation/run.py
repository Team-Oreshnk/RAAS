"""Run the CARLA-free local simulator.
The simulator is intentionally separate from the perception/planner API so it
can be developed and tested without CARLA. The API bridge is added later.
"""
import argparse, time
from simulation.scenarios.indian_traffic import make_world, trigger

def main():
    p=argparse.ArgumentParser(); p.add_argument('--traffic',type=int,default=70); p.add_argument('--pedestrians',type=int,default=20); p.add_argument('--scenario',default='chaos'); p.add_argument('--seconds',type=float,default=60)
    a=p.parse_args(); w=make_world(traffic=a.traffic,pedestrians=a.pedestrians)
    print('IIC 3.0 LOCAL 3D SIMULATOR'); print(f'traffic={a.traffic} pedestrians={a.pedestrians} duration={a.seconds}s')
    while w.time<a.seconds:
        if a.scenario!='chaos' and int(w.time)==5: trigger(w,a.scenario)
        w.step(.05, {'throttle':.4})
        if int(w.time*10)%10==0: print(f't={w.time:5.1f}s ego_speed={w.ego.speed*3.6:4.1f}km/h actors={len(w.actors)}')
        time.sleep(.01)
if __name__=='__main__': main()
