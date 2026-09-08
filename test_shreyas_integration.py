"""Tests that do not require downloading/loading the YOLO weights."""
import numpy as np
from bev import BEVGrid, BEVProjector, GroundCalibration, Intrinsics
from integration.end_to_end import IICPipeline


def test_metric_velocity_and_regrid_without_model():
    intr = Intrinsics.from_fov(800, 600, 90)
    calib = GroundCalibration(intr, 2.4, pitch_deg=8.0, mount_x_m=2.9)
    projector = BEVProjector(calib, BEVGrid())
    pipe = IICPipeline(projector)
    d = [{"id": 1, "ground": (10.0, 1.0), "cls": "car", "reliable": True}]
    d = pipe._add_metric_velocity(d, 1.0)
    assert d[0]["width"] == 1.8
    assert d[0]["length"] == 4.5
    g = np.zeros(pipe.stack.occupancy.config.shape, dtype=np.uint8)
    out = pipe._dhruv_grid_to_bev(g)
    assert out.shape == projector.grid.shape
    assert out.dtype == np.uint8
