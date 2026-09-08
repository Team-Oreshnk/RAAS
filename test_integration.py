"""Checks the BEV <-> planner boundary."""
import numpy as np
from bev import Intrinsics, GroundCalibration, BEVGrid, BEVProjector
from integration.planner_adapter import BEVPlannerAdapter
from interfaces.schemas import FREE, OCCUPIED, UNKNOWN

intr = Intrinsics.from_fov(800, 600, 90)
calib = GroundCalibration(intr, 2.4, pitch_deg=8)
grid = BEVGrid(x_min=-6, x_max=40, y_min=-15, y_max=15, resolution=0.2)
proj = BEVProjector(calib, grid)
adapter = BEVPlannerAdapter(proj)

# Every BEV cell must map to exactly one planner cell and back.
a = np.arange(np.prod(grid.shape), dtype=np.uint32).reshape(grid.shape)
b = adapter.bev_raster_to_planner(a)
c = adapter.planner_raster_to_bev(b)
assert np.array_equal(a, c)
assert b.shape == (grid.cols, grid.rows)

# Verify a known metric point lands in the correct planner index.
pc, pr = 80, 85
X = grid.x_min + (pc + 0.5) * grid.resolution
Y = grid.y_min + (pr + 0.5) * grid.resolution
bev_c, bev_r = grid.ground_to_grid(X, Y)
assert int(np.rint(grid.cols-1-bev_c)) == pr
assert int(np.rint(grid.rows-1-bev_r)) == pc

# UNKNOWN outside camera view, FREE inside.
occ = adapter.observable_as_occupancy()
assert np.all(np.isin(occ.grid, [FREE, UNKNOWN]))
assert np.any(occ.grid == UNKNOWN)

print("BEV/planner raster conversion OK")
print("planner shape:", occ.grid.shape)
print("origin:", (occ.origin_x, occ.origin_y))
print("resolution:", occ.resolution)
