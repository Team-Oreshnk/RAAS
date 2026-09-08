"""Fast IIC closed loop: MetaDrive -> YOLO -> BEV -> occupancy -> prediction
-> adaptive planner -> speed/steering controller -> MetaDrive.

The expensive perception/planning stage is intentionally throttled. The
vehicle controller still runs every simulator step using the latest safe
planner result.
"""
from __future__ import annotations

import argparse
import time
import numpy as np

from bev import BEVProjector, BEVGrid, GroundCalibration, Intrinsics
from interfaces.schemas import EgoState, Goal, SceneContext
from integration.end_to_end import IICPipeline, PipelineConfig
from simulation.metadrive.config import MetaDriveIICConfig
from simulation.metadrive.environment import IICMetaDriveEnv
from simulation.metadrive.controller import TrajectoryController


def make_projector(cfg: MetaDriveIICConfig):
    intr = Intrinsics.from_fov(
        cfg.camera_width,
        cfg.camera_height,
        cfg.camera_fov_deg,
    )
    calib = GroundCalibration(
        intr=intr,
        height_m=cfg.camera_height_m,
        pitch_deg=cfg.camera_pitch_deg,
        yaw_deg=cfg.camera_yaw_deg,
    )
    # x = forward, y = left. 0.25 m/cell is sufficient for the demo and is
    # substantially cheaper than the original 0.2 m/cell grid.
    grid = BEVGrid(
        x_min=-6.0,
        x_max=40.0,
        y_min=-15.0,
        y_max=15.0,
        resolution=0.25,
    )
    return BEVProjector(calib, grid)


def step_compat(env, action):
    result = env.step(action)
    if not isinstance(result, tuple):
        result = tuple(result)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, bool(terminated), bool(truncated), info
    if len(result) == 4:
        obs, reward, done, info = result
        return obs, reward, bool(done), False, info
    raise RuntimeError(f"Unexpected env.step() output: {len(result)} values")


def run(args):
    sim_cfg = MetaDriveIICConfig(
        use_render=not args.no_render,
        camera_width=args.width,
        camera_height=args.height,
        camera_fov_deg=args.fov,
        traffic_density=args.traffic,
        seed=args.seed,
        weights=args.weights,
        imgsz=args.imgsz,
        conf=args.conf,
        process_every_n_steps=max(1, args.perception_every),
    )

    env = IICMetaDriveEnv(sim_cfg)
    env.build()
    obs, info = env.reset()

    print("IIC 3.0 + MetaDrive started")
    print(f"camera       = {args.width}x{args.height}")
    print(f"YOLO imgsz   = {args.imgsz}")
    print(f"traffic      = {args.traffic}")
    print(f"perception   = every {args.perception_every} simulator steps")
    print("Stack: YOLO -> BEV -> Occupancy -> Future Occupancy -> Planner -> Speed/Brake")

    # --------------------------------------------------------------
    # NO-PERCEPTION MODE
    # --------------------------------------------------------------
    if args.no_perception:
        try:
            for step in range(args.steps):
                # Zero action keeps the vehicle from driving off a curved road.
                action = np.array([0.0, 0.0], dtype=np.float32)
                _, _, term, trunc, _ = step_compat(env, action)
                if step % args.print_every == 0:
                    s = env.ego_state()
                    print(
                        f"step={step:05d} speed={s['speed_mps']:.2f} m/s "
                        f"pos=({s['position_world'][0]:.1f},{s['position_world'][1]:.1f})"
                    )
                if term or trunc:
                    obs, info = env.reset()
        finally:
            env.close()
        return

    projector = make_projector(sim_cfg)
    pipeline = IICPipeline(
        projector=projector,
        config=PipelineConfig(
            weights=args.weights,
            imgsz=args.imgsz,
            tracker=sim_cfg.tracker,
            conf=args.conf,
            future_dt=sim_cfg.planner_dt,
            horizon_s=sim_cfg.prediction_horizon_s,
        ),
    )

    controller = TrajectoryController(
        max_target_speed_mps=sim_cfg.max_target_speed_mps,
        steering_gain=sim_cfg.steering_gain,
        lookahead_m=sim_cfg.lookahead_m,
        speed_kp=sim_cfg.speed_kp,
        max_throttle=sim_cfg.max_throttle,
        max_brake=sim_cfg.max_brake,
    )

    # Cached planner result: perception is expensive, control is cheap.
    last_plan = None
    last_result = None
    last_process_ms = 0.0
    frame_count = 0
    perception_count = 0
    start_wall = time.perf_counter()

    try:
        for step in range(args.steps):
            ego_sim = env.ego_state()

            # ----------------------------------------------------------
            # EXPENSIVE STAGE: only run every N simulator steps.
            # ----------------------------------------------------------
            if step % sim_cfg.process_every_n_steps == 0 or last_plan is None:
                t0 = time.perf_counter()
                cam = env.get_camera_frame()

                ego = EgoState(
                    x=0.0,
                    y=0.0,
                    heading=0.0,
                    speed=ego_sim["speed_mps"],
                )
                goal = Goal(
                    x=sim_cfg.goal_forward_m,
                    y=sim_cfg.goal_lateral_m,
                )

                # Density is updated from detected objects after perception.
                scene = SceneContext(
                    visibility=1.0,
                    pedestrian_density=0.0,
                    traffic_density=float(sim_cfg.traffic_density),
                    intersection_proximity=0.0,
                    road_width=10.0,
                )

                result = pipeline.process_frame(
                    cam.image,
                    ego,
                    goal,
                    scene,
                    timestamp=cam.timestamp,
                )

                last_result = result
                last_plan = result["planner_output"]
                perception_count += 1
                last_process_ms = (time.perf_counter() - t0) * 1000.0

            # ----------------------------------------------------------
            # CONTROLLER RUNS EVERY SIM STEP.
            # It uses the latest trajectory, target speed and brake state.
            # ----------------------------------------------------------
            action = controller.control(
                last_plan,
                ego_sim["speed_mps"],
            )

            _, _, term, trunc, _ = step_compat(env, action)
            frame_count += 1

            if step % args.print_every == 0:
                p = last_plan
                elapsed = max(time.perf_counter() - start_wall, 1e-6)
                sim_fps = frame_count / elapsed
                detections = len(last_result["detections"]) if last_result else 0
                print(
                    f"step={step:05d} "
                    f"det={detections:02d} "
                    f"speed={ego_sim['speed_mps']*3.6:4.1f}km/h "
                    f"target={p.target_speed*3.6:4.1f}km/h "
                    f"risk={p.trajectory.collision_risk:.2f} "
                    f"future={p.trajectory.predicted_collision_risk:.2f} "
                    f"unknown={p.trajectory.unknown_risk:.2f} "
                    f"brake={p.emergency_brake} "
                    f"proc={last_process_ms:4.0f}ms "
                    f"FPS={sim_fps:4.1f}"
                )

            if term or trunc:
                print("Episode ended; resetting.")
                obs, info = env.reset()
                pipeline._last_ground.clear()
                last_plan = None
                last_result = None

    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description="IIC 3.0 fast MetaDrive closed loop")
    p.add_argument("--weights", default="best.pt")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--traffic", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--height", type=int, default=216)
    p.add_argument("--fov", type=float, default=66.0)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--conf", type=float, default=0.30)
    p.add_argument("--perception-every", type=int, default=3,
                   help="Run YOLO+BEV+occupancy+planner every N simulator steps")
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--no-perception", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
