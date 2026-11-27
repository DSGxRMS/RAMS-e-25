"""Projected EUFS vehicle and cone geometry for the Small Track experiment."""

import numpy as np

# Conservative XY boxes from the chassis collision mesh and four tyre cylinders.
# The front boxes enclose each tyre over the full EUFS steering range (±0.52 rad).
FOOTPRINT = (
    (.3595, -.0065, 1.3635, .6925),
    (.79, .70, .273, .213),
    (.79, -.70, .273, .213),
    (-.79, .70, .253, .100),
    (-.79, -.70, .253, .100),
)
SMALL_CONE_RADIUS_M = .15
LARGE_CONE_RADIUS_M = .16
MARGIN_M = .10
SWEEP_STRIDE = 5


def nearby_cones(cones, x, y, reach_m):
    """Keep all observed cones within a conservative rollout reach."""
    data = np.asarray(cones, dtype=np.float32).reshape(-1, 3)
    radial = np.hypot(data[:, 0] - x, data[:, 1] - y)
    return data[np.isfinite(data).all(axis=1) & (radial <= reach_m)]


def swept_clearance(poses, cones):
    """Return minimum signed clearance per trajectory from (time, sample, xy/yaw).

    Check sampled poses with a pad derived from the translation and rotation
    summed across the intermediate 5 ms model steps. Every interpolated pose
    in a segment is within half that total motion of one of its ends.
    """
    import torch

    if not len(cones):
        return poses.new_full((poses.shape[1],), float("inf"))
    last = len(poses) - 1
    indices = list(range(0, last, SWEEP_STRIDE)) + [last]
    positions = poses[indices]
    pad = torch.zeros_like(positions[..., 0])
    if last:
        steps = poses[1:] - poses[:-1]
        rotation = torch.atan2(steps[..., 2].sin(), steps[..., 2].cos()).abs()
        # 1.9 m bounds the distance of each footprint corner from the origin.
        travel = steps[..., :2].norm(dim=-1) + 1.9 * rotation
        extra = (-last) % SWEEP_STRIDE
        if extra:
            travel = torch.nn.functional.pad(travel, (0, 0, 0, extra))
        half = .5 * travel.reshape(-1, SWEEP_STRIDE, poses.shape[1]).sum(dim=1)
        pad[:-1] = half
        pad[1:] = torch.maximum(pad[1:], half)

    yaw = positions[..., 2, None]
    cos_yaw, sin_yaw = yaw.cos(), yaw.sin()
    minimum = poses.new_full((poses.shape[1],), float("inf"))
    for chunk in cones.split(16):
        dx = chunk[:, 0] - positions[..., 0, None]
        dy = chunk[:, 1] - positions[..., 1, None]
        forward = cos_yaw * dx + sin_yaw * dy
        lateral = -sin_yaw * dx + cos_yaw * dy
        for center_x, center_y, half_length, half_width in FOOTPRINT:
            ex = (forward - center_x).abs() - half_length
            ey = (lateral - center_y).abs() - half_width
            distance = torch.hypot(ex.clamp_min(0), ey.clamp_min(0)) + torch.maximum(ex, ey).clamp_max(0)
            clearance = distance - chunk[:, 2] - MARGIN_M - pad[..., None]
            minimum = torch.minimum(minimum, clearance.amin(dim=(0, 2)))
    return minimum
