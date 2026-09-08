import math
import numpy as np

from interfaces.schemas import (
    EgoState,
    Goal,
    OccupancyInput,
    SceneContext,
    OCCUPIED,
)
from planning.adaptive_planner import AdaptivePlanner


def make_input():
    resolution = 0.5
    H, W = 100, 100
    origin = (-W*resolution/2, -H*resolution/2)
    grid = np.zeros((H, W), dtype=np.uint8)
    return grid, resolution, origin


def cell(x, y, origin, resolution):
    return (
        int(np.floor((x-origin[0])/resolution)),
        int(np.floor((y-origin[1])/resolution)),
    )


def test_empty_road():
    grid, resolution, origin = make_input()
    planner = AdaptivePlanner()

    result = planner.plan(
        OccupancyInput(
            grid, resolution, origin[0], origin[1]
        ),
        EgoState(0, -20, math.pi/2, 5),
        Goal(0, 20),
        SceneContext(),
    )

    assert result.emergency_brake is False
    assert result.trajectory.collision is False
    np.testing.assert_allclose(
        result.trajectory.points[0],
        [0, -20],
        atol=1e-8,
    )
    np.testing.assert_allclose(
        result.trajectory.points[-1],
        [0, 20],
        atol=1e-8,
    )


def test_center_obstacle_is_avoided():
    grid, resolution, origin = make_input()
    gx, gy = cell(0, 5, origin, resolution)
    grid[gy-3:gy+4, gx-4:gx+5] = OCCUPIED

    planner = AdaptivePlanner()

    result = planner.plan(
        OccupancyInput(
            grid, resolution, origin[0], origin[1]
        ),
        EgoState(0, -20, math.pi/2, 5),
        Goal(0, 20),
        SceneContext(),
    )

    assert result.emergency_brake is False
    assert result.trajectory.collision is False
    assert np.max(np.abs(result.trajectory.points[:, 0])) > 0.5


def test_completely_blocked_road_brakes():
    grid, resolution, origin = make_input()
    gx, gy = cell(0, 5, origin, resolution)
    grid[gy-3:gy+4, gx-30:gx+31] = OCCUPIED

    planner = AdaptivePlanner()

    result = planner.plan(
        OccupancyInput(
            grid, resolution, origin[0], origin[1]
        ),
        EgoState(0, -20, math.pi/2, 5),
        Goal(0, 20),
        SceneContext(),
    )

    assert result.emergency_brake is True
    assert result.target_speed == 0.0


def test_trajectory_reaches_goal():
    planner = AdaptivePlanner()

    ego = EgoState(0, -10, math.pi/2, 3)
    goal = Goal(2, 25)

    candidates = planner.generator.generate(ego, goal)

    for path in candidates:
        np.testing.assert_allclose(
            path[0], [ego.x, ego.y], atol=1e-8
        )
        np.testing.assert_allclose(
            path[-1], [goal.x, goal.y], atol=1e-8
        )
