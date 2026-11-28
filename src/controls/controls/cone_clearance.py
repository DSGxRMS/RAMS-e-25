"""Planar clearance for the EUFS chassis, tyres and cones."""

import numpy as np

from controls.steering_actuator import steering_option

# Outward unit normals and supporting offsets in base_footprint, in metres.
# Derived from the projected EUFS sisu-20d.dae collision mesh in its scene
# transform. Offsets are rounded outward so the eight sides enclose the mesh.
# The diagonals trim the empty corners alongside the nose and rear bodywork.
CHASSIS_PLANES = (
    (1.0, 0.0, 1.723),
    (.549179, .835705, 1.030),
    (0.0, 1.0, .686),
    (-.960676, .277671, .998),
    (-1.0, 0.0, 1.004),
    (-.942596, -.333936, 1.032),
    (0.0, -1.0, .699),
    (.547916, -.836533, 1.029),
)

# The front boxes enclose each tyre over the full EUFS steering range (±0.52 rad).
TYRE_BOXES = (
    (.79, .70, .273, .213),
    (.79, -.70, .273, .213),
    (-.79, .70, .253, .100),
    (-.79, -.70, .253, .100),
)
SMALL_CONE_RADIUS_M = .15
LARGE_CONE_RADIUS_M = .16
MARGIN_M = .10
SWEEP_STRIDE = 5
PREFERRED_MARGIN_M = .25
MAX_CLEARANCE_LOG_PENALTY = 6.0
CLEARANCE_SPEED_FULL_PCT = 65.0
CLEARANCE_SPEED_HYSTERESIS_PCT = 10.0
CLEARANCE_SPEED_FLOOR_MPS = 1.5
CLEARANCE_CRAWL_MPS = 0.5
CLEARANCE_SPEED_RELEASE_MPS2 = 1.5
ESCAPE_PROGRESS_M = .02
ESCAPE_SWEEP_TOLERANCE_M = .002


def cone_clearance_enabled(model):
    return steering_option("MPPI_CONE_CLEARANCE", model == "exp13")


class ClearanceSpeedGovernor:
    """Tighten on low feasibility and release slowly with hysteresis."""

    def __init__(self, max_speed_mps):
        self.max_speed_mps = max_speed_mps
        self.cap_mps = max_speed_mps

    def _requested_cap(self, feasible_pct):
        fraction = min(max(feasible_pct / CLEARANCE_SPEED_FULL_PCT, 0.0), 1.0)
        floor = min(CLEARANCE_SPEED_FLOOR_MPS, self.max_speed_mps)
        return floor + (self.max_speed_mps - floor) * fraction**2

    def observe(self, feasible_pct, dt_sim_s):
        if feasible_pct is None:
            return
        requested = self._requested_cap(feasible_pct)
        if requested < self.cap_mps:
            self.cap_mps = requested
        else:
            # Require more clearance to raise the cap than to hold it.
            release = self._requested_cap(feasible_pct - CLEARANCE_SPEED_HYSTERESIS_PCT)
            step = CLEARANCE_SPEED_RELEASE_MPS2 * min(max(dt_sim_s, 0.0), 0.1)
            self.cap_mps = max(self.cap_mps, min(release, self.cap_mps + step))


def _footprint_gap(forward, lateral, radius, xp):
    # The largest signed plane distance is a conservative distance outside
    # a polygon corner. The tyre boxes use their exact signed distances.
    chassis = forward - CHASSIS_PLANES[0][2]
    for nx, ny, offset in CHASSIS_PLANES[1:]:
        chassis = xp.maximum(chassis, nx * forward + ny * lateral - offset)
    distance = chassis
    for center_x, center_y, half_length, half_width in TYRE_BOXES:
        ex = abs(forward - center_x) - half_length
        ey = abs(lateral - center_y) - half_width
        tyre = xp.hypot(ex.clip(min=0), ey.clip(min=0)) + xp.maximum(ex, ey).clip(max=0)
        distance = xp.minimum(distance, tyre)
    return distance - radius


def measured_footprint_gap(cones):
    """Signed gap in metres from the footprint to observed cones in base_footprint."""
    data = np.asarray(cones, dtype=float).reshape(-1, 3)
    if not len(data):
        return None
    return float(_footprint_gap(data[:, 0], data[:, 1], data[:, 2], np).min())


def clearance_log_penalty(margins):
    """Prefer larger margins, including when choosing a crawl trajectory."""
    shortfall = (1.0 - margins / PREFERRED_MARGIN_M).clamp_min(0)
    return MAX_CLEARANCE_LOG_PENALTY * shortfall.square()


def nearby_cones(cones, x, y, reach_m):
    """Keep all observed cones within a conservative rollout reach."""
    data = np.asarray(cones, dtype=np.float32).reshape(-1, 3)
    radial = np.hypot(data[:, 0] - x, data[:, 1] - y)
    return data[np.isfinite(data).all(axis=1) & (radial <= reach_m)]


def _pose_gaps(poses, cos_yaw, sin_yaw, cones, torch):
    dx = cones[:, 0] - poses[..., 0, None]
    dy = cones[:, 1] - poses[..., 1, None]
    forward = cos_yaw * dx + sin_yaw * dy
    lateral = -sin_yaw * dx + cos_yaw * dy
    return _footprint_gap(forward, lateral, cones[:, 2], torch)


def _moves_away(gaps, travel, torch):
    """Check one trajectory and cone per column, using every integration step."""
    initial = gaps[0]
    # At any point of a segment, distance is bounded from below by the
    # start gap minus motion so far and the end gap minus remaining motion.
    segment_gap = .5 * (gaps[:-1] + gaps[1:] - travel[..., None])
    lower_limit = initial - ESCAPE_SWEEP_TOLERANCE_M
    lower_limit = torch.where(initial >= 0, lower_limit.clamp_min(0), lower_limit)
    no_deeper = (gaps >= initial - 1e-6).all(dim=0) & (segment_gap >= lower_limit).all(dim=0)
    outside = gaps > MARGIN_M
    no_reentry = (~outside[:-1] | (segment_gap > MARGIN_M)).all(dim=0)
    progressed = (gaps[-1] > MARGIN_M) | (gaps[-1] - initial >= ESCAPE_PROGRESS_M)
    return no_deeper & no_reentry & progressed


def swept_clearance(poses, cones):
    """Return minimum signed clearance per trajectory from (time, sample, xy/yaw).

    Check sampled poses with a pad derived from the translation and rotation
    summed across the intermediate 5 ms model steps. Every interpolated pose
    in a segment is within half that total motion of one of its ends.
    """
    return _swept_clearance(poses, cones, check_escape=False)[0]


def swept_clearance_with_escape(poses, cones):
    """Return strict margins and separately flag crawl paths that move away."""
    return _swept_clearance(poses, cones, check_escape=True)


def _swept_clearance(poses, cones, *, check_escape):
    import torch

    minimum = poses.new_full((poses.shape[1],), float("inf"))
    escaping = torch.zeros_like(minimum, dtype=torch.bool)
    if not len(cones):
        return minimum, escaping
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
        padded_travel = torch.nn.functional.pad(travel, (0, 0, 0, extra)) if extra else travel
        half = .5 * padded_travel.reshape(-1, SWEEP_STRIDE, poses.shape[1]).sum(dim=1)
        pad[:-1] = half
        pad[1:] = torch.maximum(pad[1:], half)

    yaw = positions[..., 2, None]
    cos_yaw, sin_yaw = yaw.cos(), yaw.sin()
    if check_escape and last:
        full_yaw = poses[..., 2, None]
        full_cos, full_sin = full_yaw.cos(), full_yaw.sin()
        all_cones_ok = torch.ones_like(escaping)
    for chunk in cones.split(16):
        gaps = _pose_gaps(positions, cos_yaw, sin_yaw, chunk, torch)
        margins = (gaps - pad[..., None]).amin(dim=0) - MARGIN_M
        minimum = torch.minimum(minimum, margins.amin(dim=-1))
        if check_escape and last:
            cone_ok = margins > 0
            starts_inside = gaps[0] <= MARGIN_M
            escaping |= starts_inside.any(dim=-1)
            near = torch.nonzero(starts_inside.any(dim=0)).flatten()
            if len(near):
                fine_gaps = _pose_gaps(poses, full_cos, full_sin, chunk[near], torch)
                cone_ok[:, near] |= starts_inside[:, near] & _moves_away(fine_gaps, travel, torch)
            all_cones_ok &= cone_ok.all(dim=-1)
    if check_escape and last:
        escaping &= all_cones_ok
    return minimum, escaping
