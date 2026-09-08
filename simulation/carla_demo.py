"""Full closed-loop CARLA demo: camera -> Shreyas -> BEV -> occupancy -> planner -> control.

Run CARLA separately, then:
    python -m simulation.carla_demo --map Town10HD --vehicles 100 --walkers 50 --display
"""
from __future__ import annotations
import argparse, queue, math, time
import cv2, numpy as np
from bev import BEVGrid, BEVProjector, GroundCalibration, Intrinsics
from interfaces.schemas import EgoState, Goal, SceneContext
from integration.end_to_end import IICPipeline, PipelineConfig
from simulation.config import SimConfig
from simulation.controller import PlannerVehicleController
from simulation.scenarios.indian_traffic import IndianTrafficScenario


def main():
    import carla
    ap=argparse.ArgumentParser()
    ap.add_argument("--host",default="127.0.0.1"); ap.add_argument("--port",type=int,default=2000)
    ap.add_argument("--tm-port",type=int,default=8000); ap.add_argument("--map",default="Town10HD")
    ap.add_argument("--vehicles",type=int,default=100); ap.add_argument("--walkers",type=int,default=50)
    ap.add_argument("--duration",type=float,default=90); ap.add_argument("--weights",default="best.pt")
    ap.add_argument("--display",action="store_true"); ap.add_argument("--no-yolo",action="store_true",
                    help="use CARLA ground truth detections instead of YOLO for pipeline debugging")
    args=ap.parse_args()
    cfg=SimConfig(host=args.host,port=args.port,tm_port=args.tm_port,map_name=args.map,
                  traffic_vehicles=args.vehicles,walkers=args.walkers,duration_s=args.duration)
    client=carla.Client(cfg.host,cfg.port); client.set_timeout(20)
    world=client.load_world(cfg.map_name)
    original=world.get_settings(); settings=world.get_settings(); settings.synchronous_mode=True; settings.fixed_delta_seconds=cfg.fixed_dt; world.apply_settings(settings)
    tm=client.get_trafficmanager(cfg.tm_port); tm.set_synchronous_mode(True); tm.set_random_device_seed(cfg.seed)
    tm.set_global_distance_to_leading_vehicle(2.0)
    scenario=IndianTrafficScenario(world,tm,cfg.seed,cfg.traffic_vehicles,cfg.walkers)
    ego=scenario.spawn_ego(); scenario.spawn_dense_traffic(); scenario.spawn_walkers(); scenario.schedule_events()
    image_q=queue.Queue(maxsize=2); actors=[]
    try:
        cam_bp=world.get_blueprint_library().find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x",str(cfg.camera_width)); cam_bp.set_attribute("image_size_y",str(cfg.camera_height)); cam_bp.set_attribute("fov",str(cfg.camera_fov))
        cam_tf=carla.Transform(carla.Location(x=cfg.camera_x,y=cfg.camera_y,z=cfg.camera_z),carla.Rotation(pitch=cfg.camera_pitch))
        cam=world.spawn_actor(cam_bp,cam_tf,attach_to=ego); actors.append(cam)
        def cb(img):
            try: image_q.put_nowait(img)
            except queue.Full:
                try: image_q.get_nowait(); image_q.put_nowait(img)
                except queue.Empty: pass
        cam.listen(cb)
        intr=Intrinsics.from_fov(cfg.camera_width,cfg.camera_height,cfg.camera_fov)
        calib=GroundCalibration(intr,cfg.camera_z,pitch_deg=-cfg.camera_pitch,mount_x_m=cfg.camera_x)
        grid=BEVGrid(x_min=-6,x_max=40,y_min=-15,y_max=15,resolution=0.2)
        projector=BEVProjector(calib,grid)
        pipeline=IICPipeline(projector,config=PipelineConfig(weights=args.weights))
        controller=PlannerVehicleController()
        print("IIC 3.0 CARLA CLOSED-LOOP DEMO")
        print(f"map={cfg.map_name} traffic={cfg.traffic_vehicles} walkers={cfg.walkers}")
        print("WARNING: this is an Indian-road-style stress scenario on a CARLA map, not a literal Indian city map.")
        start=0.0; frame_no=0
        while start < cfg.duration_s:
            world.tick(); start += cfg.fixed_dt; scenario.step(start)
            try: img=image_q.get(timeout=2)
            except queue.Empty: continue
            raw=np.frombuffer(img.raw_data,dtype=np.uint8).reshape((img.height,img.width,4))[:,:,:3].copy()
            vel=ego.get_velocity(); speed=math.sqrt(vel.x**2+vel.y**2+vel.z**2)
            ego_state=EgoState(0.0,0.0,0.0,speed=speed)
            goal=Goal(30.0,0.0)
            nearby=len([a for a in world.get_actors().filter("vehicle.*") if a.id!=ego.id and a.get_location().distance(ego.get_location())<cfg.density_radius_m])
            ctx=SceneContext(visibility=0.9,traffic_density=min(1.0, nearby / max(cfg.traffic_vehicles, 1)),pedestrian_density=min(1.0,len(world.get_actors().filter("walker.pedestrian.*"))/80))
            result=pipeline.process_frame(raw,ego_state,goal,ctx,start)
            po=result["planner_output"]
            control=controller.control(ego,po.trajectory.points,po.target_speed,po.emergency_brake)
            ego.apply_control(control)
            if frame_no%10==0:
                print(f"t={start:5.1f}s det={len(result['detections']):2d} safe={po.diagnostics['safe_candidate_count']:2d}/31 speed={speed*3.6:5.1f} target={po.target_speed*3.6:5.1f} risk={po.trajectory.predicted_collision_risk:.2f} brake={po.emergency_brake} {po.reason}")
            if args.display:
                annotated=pipeline.detector.annotate(raw,result["detections"])
                cv2.putText(annotated,f"target {po.target_speed*3.6:.1f} km/h | brake {po.emergency_brake}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2)
                cv2.imshow("IIC CARLA - Perception",annotated)
                bev=projector.render(raw,result["projected_detections"]); cv2.imshow("IIC CARLA - BEV",cv2.resize(bev,(600,920)))
                if cv2.waitKey(1)&255==ord('q'): break
            frame_no+=1
    finally:
        try: cam.stop()
        except Exception: pass
        for a in reversed(actors):
            try:a.destroy()
            except:pass
        scenario.destroy(); world.apply_settings(original); tm.set_synchronous_mode(False)
        if args.display: cv2.destroyAllWindows()

if __name__=="__main__": main()
