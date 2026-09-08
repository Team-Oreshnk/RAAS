# Shreyas integration

Shreyas's `best.pt` is now wrapped by `perception/shreyas.py`.

The original script ran inference at import time, opened OpenCV GUI windows,
and wrote a fixed JSON file. The integrated wrapper is inference-only and
returns one frame of detections to the common pipeline.

## Expected pipeline

```text
frame
  -> ShreyasDetector.track()
  -> BEVProjector.project_detections()
  -> metric velocity estimation
  -> Dhruv build_occupancy()
  -> Dhruv future prediction
  -> regrid to BEV raster
  -> AdaptivePlanner
```

## Important velocity fix

Shreyas's original script calculated a scalar pixel displacement per second.
That value must **not** be passed to the planner as a metric velocity.

The integrated pipeline instead projects the tracked ground-contact point into
metres and computes `vx` (forward) and `vy` (left) from consecutive frames.

## Model

`best.pt` is the trained Indian-vehicle model supplied by Shreyas.
It is loaded only when the detector is first used.

Install dependencies:

```bash
pip install -r requirements.txt
```

For the first end-to-end run, use a real frame/video and a real calibration.
Do not assume the default synthetic calibration is correct for Mumbai footage.
