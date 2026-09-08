import numpy as np
from occupancy.occupancy_grid import OccupancyGridConfig,build_occupancy,dhruv_to_planner
from prediction.future_occupancy import build_future_from_dt
from integration.planner_adapter import IntegratedPlanner
from interfaces.schemas import EgoState,Goal,OCCUPIED

def test_shapes():
    c=OccupancyGridConfig()
    g=build_occupancy([],c)
    p,*_=dhruv_to_planner(g,c)
    assert g.shape==(80,40) and p.shape==(40,80)
def test_future():
    c=OccupancyGridConfig()
    o=[{"id":1,"x":2,"y":12,"width":2,"length":4,"vx":0,"vy":3}]
    f,t=build_future_from_dt(o,.5,2,c)
    assert f.shape==(4,80,40) and t==[.5,1,1.5,2]
def test_e2e():
    c=OccupancyGridConfig()
    o=[{"id":1,"x":2,"y":12,"width":2,"length":4,"vx":0,"vy":3}]
    s=IntegratedPlanner(occupancy_config=c)
    out=s.plan_from_objects(o,EgoState(0,0,0,4),Goal(30,0),future_dt=.5,horizon_s=2)
    assert out.diagnostics["candidate_count"]==31
