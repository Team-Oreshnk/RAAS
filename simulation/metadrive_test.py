
import numpy as np
from simulation.metadrive.config import MetaDriveIICConfig
from simulation.metadrive.environment import IICMetaDriveEnv


def step_compat(env, action):
    result = tuple(env.step(action))
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, bool(terminated), bool(truncated), info
    if len(result) == 4:
        obs, reward, done, info = result
        return obs, reward, bool(done), False, info
    raise RuntimeError(f"Unexpected env.step() output: {len(result)} values")


def main():
    cfg = MetaDriveIICConfig(
        use_render=True,
        camera_width=512,
        camera_height=256,
        traffic_density=0.0,
        num_scenarios=1,
    )
    env = IICMetaDriveEnv(cfg)
    env.build()
    obs, info = env.reset()

    print("MetaDrive sanity test started.")
    print("The car is intentionally stationary.")
    print("Press Ctrl+C to stop.")

    try:
        for step in range(100000):
            obs, reward, term, trunc, info = step_compat(
                env, np.array([0.0, 0.0], dtype=np.float32)
            )
            if step % 100 == 0:
                frame = env.get_camera_frame()
                ego = env.ego_state()
                print(
                    f"step={step:06d} "
                    f"speed={ego['speed_mps']:.2f} m/s "
                    f"frame={frame.image.shape}"
                )
            if term or trunc:
                obs, info = env.reset()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
