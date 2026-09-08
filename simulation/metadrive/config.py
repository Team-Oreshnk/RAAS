from dataclasses import dataclass


@dataclass
class MetaDriveIICConfig:
    # Rendering / camera. Keep the camera small because YOLO is the main
    # compute bottleneck. Increase only after the closed loop is stable.
    use_render: bool = True
    camera_width: int = 384
    camera_height: int = 216
    camera_fov_deg: float = 66.0
    camera_height_m: float = 1.5
    camera_pitch_deg: float = 8.0
    camera_yaw_deg: float = 0.0

    # Simulation
    map_name: str = "C"
    seed: int = 7
    horizon: int = 1500
    traffic_density: float = 0.10
    num_scenarios: int = 1
    random_lane_width: bool = False
    random_lane_num: bool = False

    # IIC planner / prediction
    goal_forward_m: float = 35.0
    goal_lateral_m: float = 0.0
    planner_dt: float = 0.5
    prediction_horizon_s: float = 2.0
    process_every_n_steps: int = 3

    # Controller
    max_target_speed_mps: float = 12.0
    steering_gain: float = 1.25
    lookahead_m: float = 6.0
    speed_kp: float = 0.22
    max_throttle: float = 0.65
    max_brake: float = 1.0

    # YOLO
    weights: str = "best.pt"
    imgsz: int = 512
    conf: float = 0.30
    tracker: str = "bytetrack.yaml"
