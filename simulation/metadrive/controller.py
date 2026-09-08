"""Convert an IIC local trajectory into MetaDrive [steering, throttle_brake]."""
from __future__ import annotations
import math
import numpy as np

class TrajectoryController:
    def __init__(self, max_target_speed_mps=12.0, steering_gain=1.4,
                 lookahead_m=6.0, speed_kp=0.18, max_throttle=0.8,
                 max_brake=1.0):
        self.max_target_speed_mps = float(max_target_speed_mps)
        self.steering_gain = float(steering_gain)
        self.lookahead_m = float(lookahead_m)
        self.speed_kp = float(speed_kp)
        self.max_throttle = float(max_throttle)
        self.max_brake = float(max_brake)

    def _steering(self, points: np.ndarray) -> float:
        if points is None or len(points) == 0:
            return 0.0
        pts = np.asarray(points, dtype=float)
        d = np.linalg.norm(pts, axis=1)
        idx = int(np.argmin(np.abs(d - self.lookahead_m)))
        x, y = pts[idx, :2]  # planner frame: x forward, y left
        if x <= 0.1:
            return 0.0
        curvature_like = math.atan2(float(y), float(x))
        return float(np.clip(self.steering_gain * curvature_like, -1.0, 1.0))

    def control(self, planner_output, current_speed_mps: float):
        if planner_output.emergency_brake:
            return np.array([0.0, -self.max_brake], dtype=np.float32)

        traj = planner_output.trajectory.points
        steer = self._steering(traj)
        target = float(np.clip(planner_output.target_speed, 0.0, self.max_target_speed_mps))
        error = target - float(current_speed_mps)

        # MetaDrive's second action is throttle/brake: positive = throttle,
        # negative = brake. This is the native continuous action convention.
        throttle_brake = float(np.clip(self.speed_kp * error, -self.max_brake, self.max_throttle))
        return np.array([steer, throttle_brake], dtype=np.float32)
