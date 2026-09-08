"""Thin adapter around MetaDrive 0.4.x.

This module intentionally owns only simulator concerns: creating the world,
getting RGB frames and ego state, and advancing the vehicle with a normalized
MetaDrive action [steering, throttle_brake].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from .config import MetaDriveIICConfig

@dataclass
class CameraFrame:
    image: np.ndarray
    timestamp: float
    step: int

class IICMetaDriveEnv:
    def __init__(self, config: MetaDriveIICConfig | None = None):
        self.config = config or MetaDriveIICConfig()
        self.env = None
        self.step_index = 0
        self.sim_time = 0.0

    def _import(self):
        try:
            from metadrive.envs.metadrive_env import MetaDriveEnv
            from metadrive.component.sensors.rgb_camera import RGBCamera
        except ImportError as exc:
            raise RuntimeError(
                "MetaDrive is not installed in this Python 3.10 environment. "
                "Run: pip install metadrive-simulator"
            ) from exc
        return MetaDriveEnv, RGBCamera

    def build(self):
        MetaDriveEnv, RGBCamera = self._import()
        c = self.config
        self.env = MetaDriveEnv({
            "use_render": c.use_render,
            "image_observation": True,
            "norm_pixel": False,
            "stack_size": 1,
            "map": c.map_name,
            "start_seed": c.seed,
            "num_scenarios": c.num_scenarios,
            "horizon": c.horizon,
            "traffic_density": c.traffic_density,
            "random_lane_width": c.random_lane_width,
            "random_lane_num": c.random_lane_num,
            "show_interface": True,
            "show_fps": True,
            "vehicle_config": {
                "image_source": "rgb_camera",
                "show_navi_mark": False,
                "show_line_to_dest": False,
            },
            "sensors": {
                "rgb_camera": (RGBCamera, c.camera_width, c.camera_height),
            },
            "window_size": (c.camera_width, c.camera_height),
            "log_level": 50,
        })
        return self

    def reset(self, seed: int | None = None):
        if self.env is None:
            self.build()
        obs, info = self.env.reset(seed=self.config.seed if seed is None else seed)
        self.step_index = 0
        self.sim_time = 0.0
        return obs, info

    @staticmethod
    def _extract_image(obs: Any) -> np.ndarray:
        if isinstance(obs, dict):
            image = obs.get("image")
            if image is None:
                # Some versions/configurations expose the sensor directly.
                for value in obs.values():
                    if isinstance(value, np.ndarray) and value.ndim >= 3:
                        image = value
                        break
            if image is None:
                raise RuntimeError(f"MetaDrive observation has no image: {list(obs.keys())}")
        else:
            image = obs

        image = np.asarray(image)
        # ImageStateObservation with stack_size=1 is H,W,C,1.
        if image.ndim == 4:
            image = image[..., -1]
        if image.dtype != np.uint8:
            image = np.clip(image * (255.0 if image.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError(f"Unexpected RGB frame shape: {image.shape}")
        return image

    def frame_from_obs(self, obs) -> CameraFrame:
        return CameraFrame(self._extract_image(obs), self.sim_time, self.step_index)

    def get_camera_frame(self) -> CameraFrame:
        if self.env is None:
            raise RuntimeError("Environment has not been reset")
        sensor = self.env.engine.get_sensor("rgb_camera")
        # MetaDrive's sensor returns BGR uint8 by default. The YOLO wrapper
        # accepts numpy images directly, so we preserve that convention.
        image = sensor.perceive(to_float=False)
        image = np.asarray(image)
        if image.ndim == 4:
            image = image[..., -1]
        return CameraFrame(image.astype(np.uint8), self.sim_time, self.step_index)

    def ego(self):
        agent = getattr(self.env, "agent", None)
        if agent is None:
            agent = getattr(self.env, "vehicle", None)
        if agent is None:
            raise RuntimeError("MetaDrive ego vehicle is unavailable")
        return agent

    def ego_state(self):
        vehicle = self.ego()
        pos = np.asarray(vehicle.position, dtype=float)
        # MetaDrive heading is a 2-D vector in world coordinates. For the IIC
        # planner the ego frame is reset every frame, so heading=0 locally.
        speed = float(getattr(vehicle, "speed", 0.0))
        return {
            "position_world": pos,
            "speed_mps": speed,
            "heading_world": np.asarray(getattr(vehicle, "heading", [1.0, 0.0]), dtype=float),
        }

    def step(self, action):
        if self.env is None:
            raise RuntimeError("Environment has not been reset")
        action = np.asarray(action, dtype=np.float32).reshape(2)
        action = np.clip(action, -1.0, 1.0)
        obs, reward, terminated, truncated, info = self.env.step(action)
        # MetaDrive uses a fixed simulator step; obtain it from engine if the
        # installed version exposes it, otherwise use the common 0.05 s step.
        dt = float(getattr(self.env.engine, "step_time", 0.05))
        self.sim_time += dt
        self.step_index += 1
        return obs, reward, terminated, truncated, info

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
