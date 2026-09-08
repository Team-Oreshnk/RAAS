import math
import numpy as np
import matplotlib.pyplot as plt

from interfaces.schemas import (
    EgoState,
    Goal,
    OccupancyInput,
    SceneContext,
    FREE,
    OCCUPIED,
    UNKNOWN,
)
from planning.adaptive_planner import AdaptivePlanner


def main():
    resolution = 0.5
    H, W = 100, 100
    origin = (-W*resolution/2, -H*resolution/2)

    occupancy = np.full((H, W), FREE, dtype=np.uint8)

    planner = AdaptivePlanner()

    def cell(x, y):
        return (
            int(np.floor((x-origin[0])/resolution)),
            int(np.floor((y-origin[1])/resolution)),
        )

    obstacles = [
        (0.0, 5.0, 2.2, 2.5),
        (-5.0, 12.0, 1.8, 2.0),
        (5.0, 16.0, 1.8, 2.0),
    ]

    for x, y, width, length in obstacles:
        gx, gy = cell(x, y)
        hw = max(1, int(width/resolution/2))
        hl = max(1, int(length/resolution/2))
        occupancy[
            max(0, gy-hl):min(H, gy+hl+1),
            max(0, gx-hw):min(W, gx+hw+1),
        ] = OCCUPIED

    gx, gy = cell(-5.0, 7.0)
    occupancy[
        max(0, gy-5):min(H, gy+6),
        max(0, gx-5):min(W, gx+6),
    ] = UNKNOWN

    ego = EgoState(
        x=0.0,
        y=-20.0,
        heading=math.pi/2,
        speed=5.0,
    )

    goal = Goal(x=0.0, y=20.0)

    context = SceneContext(
        visibility=0.95,
        pedestrian_density=0.1,
        traffic_density=0.2,
        intersection_proximity=0.0,
    )

    occ_input = OccupancyInput(
        grid=occupancy,
        resolution=resolution,
        origin_x=origin[0],
        origin_y=origin[1],
    )

    result = planner.plan(
        occupancy=occ_input,
        ego=ego,
        goal=goal,
        scene_context=context,
    )

    candidates = planner.generator.generate(ego, goal)

    fig, ax = plt.subplots(figsize=(10, 10))

    yy, xx = np.where(occupancy == UNKNOWN)
    if len(xx):
        ax.scatter(
            origin[0]+(xx+0.5)*resolution,
            origin[1]+(yy+0.5)*resolution,
            marker="s",
            s=20,
            alpha=0.35,
            label="Unknown",
        )

    yy, xx = np.where(occupancy == OCCUPIED)
    if len(xx):
        ax.scatter(
            origin[0]+(xx+0.5)*resolution,
            origin[1]+(yy+0.5)*resolution,
            marker="s",
            s=20,
            alpha=0.65,
            label="Occupied",
        )

    for path in candidates:
        ax.plot(
            path[:, 0],
            path[:, 1],
            linewidth=1,
            alpha=0.18,
        )

    selected = result.trajectory.points

    ax.plot(
        selected[:, 0],
        selected[:, 1],
        linewidth=4,
        label="Selected trajectory",
    )

    ax.scatter(
        ego.x, ego.y,
        s=140,
        marker="o",
        label="Ego",
        zorder=10,
    )

    ax.scatter(
        goal.x, goal.y,
        s=180,
        marker="*",
        label="Goal",
        zorder=10,
    )

    ax.arrow(
        ego.x,
        ego.y,
        math.cos(ego.heading)*3,
        math.sin(ego.heading)*3,
        width=0.08,
        head_width=0.5,
        length_includes_head=True,
    )

    text = (
        f"Target speed: {result.target_speed:.2f} m/s\n"
        f"Collision: {result.trajectory.collision}\n"
        f"Collision risk: {result.trajectory.collision_risk:.2f}\n"
        f"Predicted risk: {result.trajectory.predicted_collision_risk:.2f}\n"
        f"Unknown risk: {result.trajectory.unknown_risk:.2f}\n"
        f"Emergency brake: {result.emergency_brake}\n"
        f"Reason: {result.reason}\n"
        f"Safe candidates: "
        f"{result.diagnostics['safe_candidate_count']}/"
        f"{result.diagnostics['candidate_count']}"
    )

    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.85),
    )

    ax.set_title("Adaptive Path Planner")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()
