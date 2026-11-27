"""Cone-map geometry and remembered corridors for the cone-memory planner."""

from dataclasses import dataclass, replace
from functools import cached_property
import math

import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial import Delaunay, QhullError, cKDTree


MAX_CONE_EDGE = 3.0
MAX_PROVISIONAL_TAIL = 3.0
ORANGE_CONE = 2
EPS = 1e-8


@dataclass(frozen=True)
class Config:
    """Distances are in metres, timestamps in seconds, and angles in degrees."""

    lookahead: float = 2.5
    fov_angle: float = 60.0
    search_yaw: float = 20.0
    horizon: float = 20.0
    sample_step: float = 0.25
    width_min: float = 2.2
    width_max: float = 5.0
    nominal_width: float = 3.0
    clearance: float = 0.8
    max_curvature: float = 0.65
    max_age: float = 8.0
    width_age: float = 3.0
    provisional_length: float = 2.0
    recovery_updates: int = 2
    colour_mode: str = 'auto'
    boundary_max_edge: float = MAX_CONE_EDGE
    anchor_ahead: float = 4.0
    entry_gap: float = 0.0
    connector_max_lead: float = 2.75

    def __post_init__(self):
        if not (0 < self.sample_step <= 0.5 and 0 < self.lookahead <= self.horizon and self.horizon > 5
                and 0 < self.clearance < self.width_min / 2 <= self.width_max / 2
                and self.width_min <= self.nominal_width <= self.width_max):
            raise ValueError('Invalid cone-memory geometry parameters')
        if self.colour_mode not in ('auto', 'blue_left', 'blue_right'):
            raise ValueError('colour_mode must be auto, blue_left or blue_right')
        if (not 0 < self.fov_angle < 180 or not 0 <= self.search_yaw < 90 or not 0 < self.max_curvature
                or not 0 < self.width_age <= self.max_age or self.recovery_updates < 1 or self.provisional_length < 0):
            raise ValueError('Invalid cone-memory search parameters')
        if not (0.25 < self.boundary_max_edge <= self.horizon and 0 <= self.entry_gap <= self.anchor_ahead <= self.horizon):
            raise ValueError('Invalid boundary edge or entry limits')
        if not 2.75 <= self.connector_max_lead <= self.horizon:
            raise ValueError('Invalid connector lead limit')


@dataclass
class Cone:
    """A locally associated, stationary cone."""

    point: np.ndarray
    colour: int
    seen: float


@dataclass(frozen=True)
class Corridor:
    """Ordered center and boundary samples from supported cross-sections."""

    center: np.ndarray
    left: np.ndarray
    right: np.ndarray
    seen: np.ndarray
    confirmed: np.ndarray
    point_sources: np.ndarray = None

    @cached_property
    def sources(self):
        if self.point_sources is None:
            return np.repeat('nominal', len(self.center))
        return self.point_sources

    @cached_property
    def persistent(self):
        """Keep measured samples and accepted shadow joins for the run."""
        return self.confirmed | (self.sources == 'shadow_spline')

    @cached_property
    def arc(self):
        return arc_length(self.center)

    def part(self, start=None, end=None):
        selection = slice(start, end)
        return Corridor(self.center[selection], self.left[selection], self.right[selection],
                        self.seen[selection], self.confirmed[selection], self.sources[selection])


@dataclass
class Result:
    """Planner output with no vehicle control commands."""

    state: str
    points: np.ndarray
    remaining: float = 0.0
    provisional: bool = False
    probe: tuple = None
    anchors: tuple = (None, None)
    cone_counts: tuple = (0, 0)
    planning_mode: str = None
    point_sources: np.ndarray = None


@dataclass
class Scan:
    """Geometry of the most recent planner search for the live view."""

    car: np.ndarray
    probe: tuple
    stamp: float
    startup: bool
    tested_yaws: tuple = ()
    selected_yaw: float = None
    shadow_bands_deg: tuple = ()
    shadow_selected: bool = False
    walls: tuple = ()
    candidate: np.ndarray = None
    shadow_candidate: np.ndarray = None
    nominal_cones: tuple = ()
    shadow_cones: tuple = ()
    shadow_outcome: str = 'inactive'
    shadow_added: np.ndarray = None
    tls_candidate: np.ndarray = None
    tls_added: np.ndarray = None


def arc_length(points):
    """Return an array of cumulative distances along a polyline."""
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def unit(vector):
    return vector / max(np.linalg.norm(vector), EPS)


def left_normal(direction):
    return np.array([-direction[1], direction[0]])


def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def cone_links(points, max_edge=MAX_CONE_EDGE):
    """Return length-gated Delaunay edges and whether triangulation succeeded."""
    if len(points) >= 3:
        try:
            triangles = Delaunay(points).simplices
            edges = np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
            edges = np.unique(np.sort(edges, axis=1), axis=0)
            distances = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
            return set(map(tuple, edges[(distances > EPS) & (distances <= max_edge + EPS)])), True
        except QhullError:
            pass
    return set(), False


def segments(corridor_left, corridor_right):
    """Return start and end arrays for both finite walls."""
    walls = np.concatenate((corridor_left[:-1], corridor_right[:-1]))
    ends = np.concatenate((corridor_left[1:], corridor_right[1:]))
    return walls, ends


def wall_clearance(points, starts, ends):
    """Distance from each point to the nearest finite wall segment."""
    legs = ends - starts
    offsets = points[:, None, :] - starts
    ratio = np.clip(np.sum(offsets * legs, axis=2) / np.maximum(np.sum(legs**2, axis=1), EPS), 0, 1)
    return np.min(np.linalg.norm(offsets - ratio[..., None] * legs, axis=2), axis=1)


def validate_prefix(points, starts, ends, cfg):
    """Return the number of path points supported before the first violation."""
    if len(points) < 2 or not len(starts):
        return 0
    legs = np.diff(points, axis=0)
    lengths = np.linalg.norm(legs, axis=1)
    wall_legs = ends - starts
    offset = starts - points[:-1, None, :]
    determinant = cross(legs[:, None, :], wall_legs)
    denominator = np.where(np.abs(determinant) > EPS, determinant, 1.0)
    along_path = cross(offset, wall_legs) / denominator
    along_wall = cross(offset, legs[:, None, :]) / denominator
    crosses = (np.abs(determinant) > EPS) & (along_path >= 0) & (along_path <= 1)
    crosses &= (along_wall >= 0) & (along_wall <= 1)
    # Clearance is 1-Lipschitz. The extra half-sample margin covers gaps between samples.
    sample_count = max(1, math.ceil(float(max(lengths)) / 0.1))
    fractions = np.linspace(0.0, 1.0, sample_count + 1)
    samples = points[:-1, None, :] + fractions[None, :, None] * legs[:, None, :]
    distance = wall_clearance(samples.reshape(-1, 2), starts, ends).reshape(len(legs), -1)
    valid = (distance.min(axis=1) >= cfg.clearance + lengths / (2 * sample_count))
    valid &= (lengths > EPS) & ~crosses.any(axis=1)
    if len(legs) > 1:
        turn = np.abs(np.arctan2(cross(legs[:-1], legs[1:]), np.sum(legs[:-1] * legs[1:], axis=1)))
        valid[1:] &= turn <= cfg.max_curvature * (lengths[:-1] + lengths[1:]) / 2 + 0.03
    failures = np.flatnonzero(~valid)
    return int(failures[0] + 1) if len(failures) else len(points)


def sample_at(corridor, station):
    station = np.clip(station, 0, corridor.arc[-1])
    i = min(np.searchsorted(corridor.arc, station, side='right') - 1, len(corridor.center) - 2)
    leg = corridor.center[i + 1] - corridor.center[i]
    ratio = (station - corridor.arc[i]) / max(corridor.arc[i + 1] - corridor.arc[i], EPS)
    return corridor.center[i] + leg * ratio, unit(leg)


def project(corridor, car, heading, prior=0.0, window=5.0):
    """Project on a bounded, forward-facing part of the remembered branch."""
    legs = np.diff(corridor.center, axis=0)
    lengths = np.diff(corridor.arc)
    relative = car - corridor.center[:-1]
    u = np.clip(np.sum(relative * legs, axis=1) / np.maximum(lengths**2, EPS), 0, 1)
    station = corridor.arc[:-1] + u * lengths
    error = np.linalg.norm(relative - u[:, None] * legs, axis=1)
    alignment = legs @ heading / np.maximum(lengths, EPS)
    allowed = (station >= prior - 1.0) & (station <= prior + window) & (alignment > 0.25)
    score = np.where(allowed, error + 0.45 * (1 - alignment) + 0.02 * abs(station - prior), np.inf)
    best = np.argmin(score)
    if not allowed[best] or error[best] > 1.5:
        return None
    return float(station[best])


def entry_available(corridor, car, heading, cfg):
    """Check a provisional approach to a forward, two-sided corridor entrance."""
    if not cfg.entry_gap or len(corridor.confirmed) < 2 or not corridor.confirmed[:2].all():
        return False
    start, tangent = sample_at(corridor, 0.0)
    offset = start - car
    half_width = np.linalg.norm(corridor.left[0] - corridor.right[0]) / 2
    return (0 < offset @ tangent <= cfg.entry_gap and heading @ tangent >= math.cos(math.radians(25))
            and abs(cross(tangent, offset)) <= half_width - cfg.clearance)


def trace_wall(points, indices, links, triangulated, car, direction, side, cfg, nearest_fallback=False,
               *, delaunay_only=False):
    """Trace one ordered wall from an anchor on the active branch."""
    if not len(indices):
        return []
    relative = points[indices] - car
    forward = relative @ direction
    lateral = side * (relative @ left_normal(direction))
    anchors = (forward >= -2.5) & (forward <= cfg.anchor_ahead) & (lateral >= 0.4)
    anchors &= lateral <= cfg.width_max / 2 + 0.7
    if not anchors.any():
        return []
    if delaunay_only:
        score = np.linalg.norm(relative, axis=1)
    else:
        score = abs(forward + 1.15) + 1.8 * abs(lateral - cfg.nominal_width / 2)
    chain = [int(indices[np.argmin(np.where(anchors, score, np.inf))])]
    walked = 0.0
    while walked < cfg.horizon + 3.0:
        current = chain[-1]
        legs = points[indices] - points[current]
        lengths = np.linalg.norm(legs, axis=1)
        directions = legs / np.maximum(lengths[:, None], EPS)
        turns = np.abs(np.arctan2(cross(direction, directions), directions @ direction))
        allowed = (lengths > 0.25) & (lengths <= cfg.boundary_max_edge + EPS)
        allowed &= (turns <= math.radians(67)) & ~np.isin(indices, chain)
        on_mesh = np.array([tuple(sorted((current, int(i)))) in links for i in indices])
        if delaunay_only or (triangulated and (allowed & on_mesh).any()):
            allowed &= on_mesh
        else:
            allowed &= turns <= math.radians(42)
        score = lengths + 2.3 * turns
        long_edge = not allowed.any()
        if long_edge:
            if delaunay_only or not nearest_fallback:
                break
            allowed = (lengths > cfg.boundary_max_edge + EPS) & ~np.isin(indices, chain)
            allowed &= turns <= math.radians(67)
            if not allowed.any():
                break
            score = lengths
        ranks = np.flatnonzero(allowed)
        ranks = ranks[np.argsort(score[ranks], kind='stable')]
        first = ranks[0]
        if not long_edge and len(ranks) > 1:
            second = ranks[1]
            different = directions[first] @ directions[second] < math.cos(math.radians(38))
            if different and score[second] - score[first] < 0.35:
                break
        chain.append(int(indices[first]))
        direction = directions[first]
        walked += lengths[first]
    return chain


def orange_gate_pairs(points, left, right, links, orange, heading, cfg):
    """Pair orange entry cones across the lane using near-transverse Delaunay edges."""
    pairs = []
    normal = left_normal(heading)
    for i in left:
        if not orange[i]:
            continue
        for j in right:
            if not orange[j] or tuple(sorted((i, j))) not in links:
                continue
            span = points[i] - points[j]
            stagger = abs(span @ heading)
            if stagger <= 0.45 and cfg.width_min <= span @ normal <= cfg.width_max:
                pairs.append((stagger, i, j))
    gates, used_right = {}, set()
    for _, i, j in sorted(pairs):
        if i not in gates and j not in used_right:
            gates[i] = j
            used_right.add(j)
    return gates


def cross_sections(points, seen, left, right, links, cfg, entry_pairs=None):
    """Pair finite walls in order and use compatible Delaunay midpoints."""
    if len(left) < 2 or len(right) < 2:
        return None
    left, right = np.array(left), np.array(right)
    lwall, rwall = points[left], points[right]
    larc, rarc = arc_length(lwall), arc_length(rwall)
    left_legs = np.diff(lwall, axis=0)
    left_directions = left_legs / np.diff(larc)[:, None]
    tangents = np.vstack((left_directions[0], left_directions[:-1] + left_directions[1:], left_directions[-1]))
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    left_stations = np.unique(np.r_[np.arange(0, larc[-1], cfg.sample_step), larc])
    entry_gates = {}
    if entry_pairs:
        right_indices = {key: j for j, key in enumerate(right)}
        entry_gates = {larc[i]: (i, right_indices[entry_pairs[key]])
                       for i, key in enumerate(left) if key in entry_pairs}
        # Leave room for each midpoint, whose forward station can differ from the left cone.
        entry_stations = np.array(list(entry_gates))
        near_entry = np.any(abs(left_stations[:, None] - entry_stations) < cfg.sample_step - EPS, axis=1)
        left_stations = np.sort(np.r_[left_stations[~near_entry], entry_stations])
    last_entry = max(entry_gates, default=-math.inf)
    right_legs = np.diff(rwall, axis=0)
    pairs = [(i, j) for i in left for j in right if tuple(sorted((i, j))) in links
             and np.linalg.norm(points[i] - points[j]) <= MAX_CONE_EDGE + EPS]
    pairs = np.array(pairs, dtype=int).reshape(-1, 2)
    first, second = points[pairs[:, 0]], points[pairs[:, 1]]
    edge_stamps = np.min(seen[pairs], axis=1)
    unused = np.ones(len(pairs), dtype=bool)
    gates = []
    fixed_midpoints = []
    right_progress = -EPS
    left_progress = 0.0
    for station in left_stations:
        if station in entry_gates:
            i, j = entry_gates[station]
            if rarc[j] >= right_progress - EPS:
                a, b = lwall[i], rwall[j]
                fixed_midpoints.append(len(gates))
                gates.append(((a + b) / 2, a, b, min(seen[left[i]], seen[right[j]])))
                left_progress, right_progress = station, rarc[j]
            continue
        index = min(np.searchsorted(larc, station, side='right') - 1, len(left) - 2)
        fraction = (station - larc[index]) / (larc[index + 1] - larc[index])
        a = lwall[index] + left_legs[index] * fraction
        direction = unit((1 - fraction) * tangents[index] + fraction * tangents[index + 1])
        across = right_legs @ direction
        ratio = ((a - rwall[:-1]) @ direction) / np.where(abs(across) > EPS, across, 1.0)
        b = rwall[:-1] + np.clip(ratio, 0, 1)[:, None] * right_legs
        rprogress = rarc[:-1] + np.clip(ratio, 0, 1) * np.diff(rarc)
        width = -(b - a) @ left_normal(direction)
        valid = (abs(across) > EPS) & (ratio >= -EPS) & (ratio <= 1 + EPS)
        valid &= (width >= cfg.width_min) & (width <= cfg.width_max)
        valid &= rprogress >= right_progress - 0.10
        if gates:
            valid &= rprogress <= right_progress + max(0.8, 3 * (station - left_progress))
        if not valid.any():
            if gates and station >= last_entry:
                break
            continue
        j = np.argmin(np.where(valid, abs(width - cfg.nominal_width) + 0.06 * rprogress, np.inf))
        wall_b = b[j]
        stamp = min(seen[left[index]], seen[left[index + 1]], seen[right[j]], seen[right[j + 1]])
        center = (a + wall_b) / 2
        # A nearby real cross-colour gate can supply the midpoint at this station.
        matching = unused & (np.linalg.norm(a - first, axis=1) <= 0.07)
        matching &= (np.linalg.norm(wall_b - second, axis=1) <= 0.07) & (abs((first - second) @ direction) <= 0.10)
        if matching.any():
            k = np.flatnonzero(matching)[0]
            center = (first[k] + second[k]) / 2
            stamp = min(stamp, edge_stamps[k])
            unused[k] = False
        if not gates or np.linalg.norm(center - gates[-1][0]) > 0.08:
            gates.append((center, a, wall_b, stamp))
            left_progress = station
            right_progress = max(right_progress, rprogress[j])
    if len(gates) < 2:
        return None
    center, a, b, stamps = (np.array(values) for values in zip(*gates))
    # Cross-section midpoints inherit kinks at the cones. Smooth on a fixed
    # arc-length scale and check the result against the original finite walls.
    arc = arc_length(center)
    radius = np.minimum(np.minimum(arc, arc[-1] - arc), 0.65)
    for dim in range(2):
        values = center[:, dim].copy()
        center[:, dim] = (2 * values + np.interp(arc - radius, arc, values)
                          + np.interp(arc + radius, arc, values)) / 4
    center[fixed_midpoints] = (a[fixed_midpoints] + b[fixed_midpoints]) / 2
    starts, ends = segments(lwall, rwall)
    count = validate_prefix(center, starts, ends, cfg)
    if entry_pairs and (not fixed_midpoints or fixed_midpoints[0] >= count):
        return None
    return Corridor(center[:count], a[:count], b[:count], stamps[:count], np.ones(count, dtype=bool))


def join_corridors(previous, candidate, progress, cfg, *, match_walls=True):
    """Keep the near-car prefix and replace its tail only at an aligned overlap."""
    if candidate is None or len(candidate.center) < 2:
        return previous
    if previous is None:
        return candidate
    if not np.any(previous.confirmed):
        return candidate
    previous_tangent = np.diff(previous.center, axis=0)
    candidate_tangent = np.diff(candidate.center, axis=0)
    distance, match = cKDTree(candidate.center).query(previous.center)
    pair = np.minimum(match[:-1], len(candidate_tangent) - 1)
    alignment = np.sum(previous_tangent * candidate_tangent[pair], axis=1)
    alignment /= np.maximum(np.linalg.norm(previous_tangent, axis=1)
                            * np.linalg.norm(candidate_tangent[pair], axis=1), EPS)
    aligned = (distance < 0.18) & np.r_[alignment > 0.97, False]
    if match_walls:
        aligned &= np.linalg.norm(previous.left - candidate.left[match], axis=1) < 0.4
        aligned &= np.linalg.norm(previous.right - candidate.right[match], axis=1) < 0.4
    support = aligned & (previous.confirmed == candidate.confirmed[match])
    seen = np.where(support, np.maximum(previous.seen, candidate.seen[match]), previous.seen)
    previous = Corridor(previous.center, previous.left, previous.right, seen, previous.confirmed, previous.sources)
    tail_remaining = candidate.arc[-1] - candidate.arc[match]
    confirmed = np.flatnonzero(previous.confirmed)
    confirmed_end = previous.arc[confirmed[-1]] if len(confirmed) else 0.0
    extends = previous.arc + tail_remaining > confirmed_end + 0.08
    if not previous.confirmed[-1]:
        extends |= previous.arc + tail_remaining >= confirmed_end - 0.08
    overlaps = np.flatnonzero(aligned & (previous.arc >= progress + 0.35)
                              & (previous.arc <= progress + 2.0)
                              & candidate.confirmed[match] & extends)
    for i in overlaps:
        j = match[i] + 1
        while j < len(candidate.center) and np.linalg.norm(candidate.center[j] - previous.center[i]) < 0.14:
            j += 1
        if j == len(candidate.center):
            continue
        merged = Corridor(*(np.concatenate((old[:i + 1], new[j:])) for old, new in (
            (previous.center, candidate.center), (previous.left, candidate.left),
            (previous.right, candidate.right), (previous.seen, candidate.seen),
            (previous.confirmed, candidate.confirmed), (previous.sources, candidate.sources))))
        walls = segments(merged.left, merged.right)
        start = max(0, i - 1)
        if (validate_prefix(merged.center[start:], *walls, cfg) == len(merged.center) - start
                and not np.any(merged.confirmed[1:] & ~merged.confirmed[:-1])):
            return merged
    return previous


def connector_curve(start, heading, target, tangent, cfg):
    """Build a cubic connector with the requested endpoint tangents."""
    span = np.linalg.norm(target - start)
    if span < 0.18 or (target - start) @ heading <= 0:
        return None
    handle = span / 3.0
    points = np.array([start, start + handle * heading, target - handle * tangent, target])
    spline = BSpline([0, 0, 0, 0, 1, 1, 1, 1], points, 3)
    parameters = np.linspace(0, 1, max(5, math.ceil(span / 0.1)))
    velocity = spline.derivative(1)(parameters)
    speed = np.linalg.norm(velocity, axis=1)
    curvature = abs(cross(velocity, spline.derivative(2)(parameters))) / np.maximum(speed**3, EPS)
    if speed.min() < 0.05 or curvature.max() > cfg.max_curvature:
        return None
    return spline


def provisional_tail(corridor, wall, stamps, side, width, cfg):
    """Infer a bounded extension from a real wall joined to a confirmed corridor."""
    limit = min(cfg.provisional_length, MAX_PROVISIONAL_TAIL)
    confirmed = np.flatnonzero(corridor.confirmed)
    if len(wall) < 2 or len(confirmed) < 2:
        return corridor
    end = int(confirmed[-1])
    base = corridor.part(end=end + 1)
    supported = base.left[-1] if side == 1 else base.right[-1]
    arc = arc_length(wall)
    legs = np.diff(wall, axis=0)
    span = np.diff(arc)
    ratios = np.clip(np.sum((supported - wall[:-1]) * legs, axis=1) / span**2, 0, 1)
    projections = wall[:-1] + ratios[:, None] * legs
    direction = unit(base.center[-1] - base.center[-2])
    distance = np.linalg.norm(projections - supported, axis=1)
    aligned = legs @ direction > 0.9 * span
    index = int(np.argmin(np.where(aligned, distance, np.inf)))
    if not aligned[index] or distance[index] > 0.35 or abs(width - np.linalg.norm(base.left[-1] - base.right[-1])) > 0.5:
        return corridor
    start = arc[index] + ratios[index] * span[index]
    samples = np.arange(start + cfg.sample_step, min(arc[-1] + EPS, start + limit + EPS / 2),
                        cfg.sample_step)
    if not len(samples):
        return corridor
    i = np.minimum(np.searchsorted(arc, samples, side='right') - 1, len(legs) - 1)
    actual = wall[i] + legs[i] * ((samples - arc[i]) / span[i])[:, None]
    directions = wall[np.minimum(i + 2, len(wall) - 1)] - wall[np.maximum(i - 1, 0)]
    normals = np.column_stack((-directions[:, 1], directions[:, 0])) / np.linalg.norm(directions, axis=1)[:, None]
    inferred = actual - side * width * normals
    left, right = (actual, inferred) if side == 1 else (inferred, actual)
    tail = (left + right) / 2
    under_cap = arc_length(np.vstack((base.center[-1], tail)))[1:] <= limit + EPS
    count = int(np.count_nonzero(under_cap))
    if not count:
        return corridor
    merged = Corridor(np.vstack((base.center, tail[:count])), np.vstack((base.left, left[:count])),
                      np.vstack((base.right, right[:count])),
                      np.r_[base.seen, np.minimum(stamps[i], stamps[i + 1])[:count]],
                      np.r_[base.confirmed, np.zeros(count, dtype=bool)],
                      np.r_[base.sources, np.repeat('one_side', count)])
    start = max(0, end - 1)
    size = validate_prefix(merged.center[start:], *segments(merged.left, merged.right), cfg)
    accepted = merged.part(end=start + size)
    if len(accepted.center) > len(corridor.center):
        return accepted
    return corridor


class ConeMemoryPlanner:
    """Plan a current corridor or use its remembered, still-supported prefix."""

    nearest_fallback = False

    def __init__(self, config=None):
        self.cfg = config or Config()
        self.reset()

    def reset(self, *, keep_path_memory=False):
        """Reset active tracking, optionally retaining coordinates in the same map."""
        if not keep_path_memory:
            self.remembered_path_points = set()
        self.cones = {}
        self.visible = set()
        self.next_id = 0
        self.last_map_stamp = None
        self.observation_snapshots = False
        self.revision = 0
        self.last_revision = -1
        self.cache = None
        self.scan = None
        self.startup_scan_complete = False
        self.orange_entry_end = None
        self.progress = 0.0
        self.last_car = None
        self.last_build = None
        self.last_time = None
        self.state = 'no_valid_path'
        self.held_probe = None
        self.lead = self.cfg.lookahead
        self.recovery_revision = -1
        self.good_updates = 0
        self.counts = (0, 0)
        self.anchors = (None, None)
        self.blue_on_left = {'blue_left': True, 'blue_right': False}.get(self.cfg.colour_mode)
        self.width = self.cfg.nominal_width
        self.width_stamp = -math.inf

    def observe(self, blue, yellow, stamp, observation_snapshot=False, *, orange=()):
        """Associate a map message once. Only direct observations refresh unchanged cones."""
        if self.last_map_stamp is not None and stamp <= self.last_map_stamp:
            return
        self.last_map_stamp = stamp
        self.observation_snapshots = observation_snapshot
        visible = set()
        changed = False
        for colour, items in enumerate((blue, yellow, orange)):
            points = np.array(items, dtype=float).reshape(-1, 2)
            points = points[np.isfinite(points).all(axis=1)]
            points = np.unique(np.round(points, 4), axis=0)
            keys = [key for key, cone in self.cones.items() if cone.colour == colour]
            if keys and len(points):
                tree = cKDTree([self.cones[key].point for key in keys])
                distances, candidates = tree.query(points, distance_upper_bound=0.35)
            else:
                distances = np.full(len(points), np.inf)
                candidates = np.zeros(len(points), dtype=int)
            for i in np.argsort(distances, kind='stable'):
                point = points[i]
                if np.isfinite(distances[i]):
                    key = keys[candidates[i]]
                    if key in visible:
                        continue
                    cone = self.cones[key]
                    if observation_snapshot or distances[i] > 0.06:
                        cone.seen = stamp
                        if distances[i] > 0.06:
                            cone.point = point
                        changed = True
                else:
                    key = self.next_id
                    self.next_id += 1
                    self.cones[key] = Cone(point, colour, stamp)
                    changed = True
                visible.add(key)
        if changed or visible != self.visible:
            self.revision += 1
        self.visible = visible

    def _probe(self, car, heading):
        if self.held_probe is not None and self.state == 'search':
            return self.held_probe
        return self._path_probe(car, heading, self.lead)

    def _path_probe(self, car, heading, lead):
        if self.cache is None:
            return car, heading
        confirmed = np.flatnonzero(self.cache.confirmed)
        if not len(confirmed):
            return car, heading
        ceiling = self.cache.arc[confirmed[-1]]
        if ceiling < self.progress:
            return sample_at(self.cache, self.progress)
        return sample_at(self.cache, min(self.progress + lead, ceiling))

    def _search_offsets(self, recovering, startup):
        if recovering and not startup and self.cfg.search_yaw:
            return (-self.cfg.search_yaw, self.cfg.search_yaw)
        return ()

    def _shadow_bands(self, startup):
        """Return the active bands beside the nominal sector."""
        return ()

    def _defer_shadow(self, corridor):
        """Allow a planner to merge a shadow candidate after other extensions."""
        return False

    def _join_candidate(self, candidate):
        return join_corridors(self.cache, candidate, self.progress, self.cfg)

    def _trace_boundary(self, points, indices, links, triangulated, car, facing, side, *, orange_entry=False):
        return trace_wall(points, indices, links, triangulated, car, facing, side, self.cfg,
                          self.nearest_fallback and not orange_entry)

    def _build_corridor(self, points, ages, colours, left, right, links, *, entry_pairs=None):
        return cross_sections(points, ages, left, right, links, self.cfg, entry_pairs=entry_pairs)

    def _assemble(self, car, heading, probe, now):
        startup = not self.startup_scan_complete
        self.scan = Scan(car, probe, now, startup)
        if self.orange_entry_end is not None:
            endpoint, entry_heading = self.orange_entry_end
            if (endpoint - car) @ entry_heading <= 0:
                self.orange_entry_end = None
        use_orange_entry = startup or self.orange_entry_end is not None
        if self.observation_snapshots:
            self.cones = {key: cone for key, cone in self.cones.items() if now - cone.seen <= self.cfg.max_age}
        keys = [key for key, cone in self.cones.items() if now - cone.seen <= self.cfg.max_age
                and (use_orange_entry or cone.colour != ORANGE_CONE)
                and np.linalg.norm(cone.point - car) <= self.cfg.horizon + 8]
        keys.sort(key=lambda key: self.cones[key].colour == ORANGE_CONE)
        if not keys:
            self.counts, self.anchors = (0, 0), (None, None)
            self.scan.shadow_bands_deg = self._shadow_bands(startup)
            if self.scan.shadow_bands_deg:
                self.scan.shadow_outcome = 'no_shadow_cones'
            return None, None, None
        points = np.array([self.cones[key].point for key in keys])
        ages = np.array([self.cones[key].seen for key in keys])
        colours = np.array([self.cones[key].colour for key in keys])
        orange = colours == ORANGE_CONE
        links, triangulated = cone_links(points[~orange], max(MAX_CONE_EDGE, self.cfg.boundary_max_edge))
        nearby = np.linalg.norm(points - car, axis=1) < max(5.0, self.cfg.anchor_ahead + 0.5)
        available = np.array([key in self.visible for key in keys])
        if self.last_map_stamp is None or now - self.last_map_stamp > 0.4:
            available[:] = False
        if self.cache is not None:
            close, _ = cKDTree(self.cache.center).query(points)
            nearby |= close <= self.cfg.width_max / 2 + 0.25
        facing = heading
        if self.cache is not None:
            _, facing = sample_at(self.cache, self.progress)
        relative = points - probe[0]
        yaw = math.atan2(probe[1][1], probe[1][0])
        in_range = np.linalg.norm(relative, axis=1) <= self.cfg.horizon

        def window(offset):
            axis = np.array([math.cos(yaw + offset), math.sin(yaw + offset)])
            angle = np.abs(np.arctan2(relative @ left_normal(axis), relative @ axis))
            return (angle <= math.radians(self.cfg.fov_angle / 2)) & in_range

        nominal = window(0.0)
        startup_window = np.linalg.norm(points - car, axis=1) <= self.cfg.horizon
        cone_window = startup_window if startup else nominal
        self.scan.nominal_cones = tuple((self.cones[keys[i]], bool(available[i]))
                                        for i in np.flatnonzero(cone_window))
        if not startup:
            self.scan.tested_yaws = (0.0,)
        lateral = (points - car) @ left_normal(facing)
        if self.blue_on_left is None:
            blue = lateral[(colours == 0) & nearby]
            yellow = lateral[(colours == 1) & nearby]
            if len(blue) and len(yellow) and np.median(blue) * np.median(yellow) < 0:
                self.blue_on_left = bool(np.median(blue) > 0)
        left_colour = 0 if self.blue_on_left else 1

        def trace(mask):
            if self.blue_on_left is None:
                return [], []
            left = np.flatnonzero(mask & (colours == left_colour))
            right = np.flatnonzero(mask & (colours == 1 - left_colour))
            return (self._trace_boundary(points, left, links, triangulated, car, facing, 1),
                    self._trace_boundary(points, right, links, triangulated, car, facing, -1))

        chains = trace(cone_window | nearby)
        counted = cone_window & available
        self.counts = tuple(int(np.count_nonzero(counted[chain])) for chain in chains)
        recovering = min(self.counts) <= 1 or self.state == 'search'
        proposals = [(0.0, chains, False)]
        offsets = self._search_offsets(recovering, startup)
        self.scan.tested_yaws += offsets
        for offset in offsets:
            alternate = trace(window(math.radians(offset)) | nearby)
            if all(alternate != chain_pair for _, chain_pair, _ in proposals):
                proposals.append((offset, alternate, False))
        bands = self._shadow_bands(startup)
        self.scan.shadow_bands_deg = bands
        if bands:
            axis = probe[1]
            angle = np.degrees(np.arctan2(relative @ left_normal(axis), relative @ axis))
            shadow = np.zeros(len(points), dtype=bool)
            for low, high in bands:
                shadow |= (angle >= low) & (angle <= high)
            shadow &= in_range & ~nominal
            self.scan.shadow_cones = tuple((self.cones[keys[i]], bool(available[i]))
                                           for i in np.flatnonzero(shadow))
            self.scan.shadow_outcome = 'no_shadow_cones' if not shadow.any() else 'unchanged_boundaries'
            alternate = trace(nominal | shadow | nearby)
            if alternate != chains:
                self.scan.shadow_outcome = 'no_centerline'
                proposals.append((0.0, alternate, True))
        best, best_walls = None, chains
        best_score = -math.inf
        for yaw_offset, (left, right), uses_shadow in proposals:
            corridor = self._build_corridor(points, ages, colours, left, right, links)
            if corridor is None or len(corridor.center) < 2:
                continue
            if uses_shadow:
                corridor = replace(corridor, point_sources=np.repeat('shadow', len(corridor.center)))
                self.scan.shadow_candidate = corridor.center
                self.scan.shadow_outcome = 'candidate_not_joined'
                if self._defer_shadow(corridor):
                    continue
            start = project(corridor, car, heading)
            if start is None:
                if not entry_available(corridor, car, heading, self.cfg):
                    continue
                start = 0.0
            if self.cache is not None:
                near, tangent = sample_at(self.cache, self.progress)
                overlap = project(corridor, near, tangent)
                if overlap is None or np.linalg.norm(sample_at(corridor, overlap)[0] - near) > 0.5:
                    continue
            tail = corridor.arc[-1] - start
            score = tail - 2 * np.linalg.norm(sample_at(corridor, start)[0] - car)
            if score > best_score:
                best, best_walls, best_score = corridor, (left, right), score
                if not startup:
                    self.scan.selected_yaw = yaw_offset
                    self.scan.shadow_selected = uses_shadow
        if (best is None or self.nearest_fallback) and use_orange_entry and np.any(orange & available & startup_window):
            entry = (orange & available & startup_window & (abs(lateral) >= 0.4)
                     & (abs(lateral) <= self.cfg.width_max / 2 + 0.7))
            mesh, triangulated = cone_links(points, max(self.cfg.width_max, self.cfg.boundary_max_edge))
            entry_walls = []
            for colour, side in ((left_colour, 1), (1 - left_colour, -1)):
                coloured = (colours == colour) & startup_window if self.blue_on_left is not None else False
                indices = np.flatnonzero(coloured | (entry & (side * lateral > 0)))
                entry_walls.append(self._trace_boundary(points, indices, mesh, triangulated, car, facing, side,
                                                        orange_entry=True))
            left, right = entry_walls
            gates = orange_gate_pairs(points, left, right, mesh, entry, facing, self.cfg)
            if gates:
                corridor = self._build_corridor(points, ages, colours, left, right, mesh, entry_pairs=gates)
                if corridor is not None and len(corridor.center) >= 2 and (
                        project(corridor, car, heading) is not None or entry_available(corridor, car, heading, self.cfg)):
                    best = replace(corridor, point_sources=np.repeat('orange_entry', len(corridor.center)))
                    best_walls = entry_walls
                    if startup:
                        midpoints = [(points[i] + points[j]) / 2 for i, j in gates.items()]
                        endpoint = max(midpoints, key=lambda point: point @ facing)
                        self.orange_entry_end = (endpoint, facing)
                        self.counts = tuple(int(np.count_nonzero(counted[chain])) for chain in best_walls)
        self.anchors = tuple(tuple(points[chain[2]]) if len(chain) >= 3 else None for chain in best_walls)
        wall_data = tuple((points[chain], ages[chain]) for chain in best_walls)
        self.scan.walls = tuple(wall for wall, _ in wall_data)
        if best is not None:
            self.scan.candidate = best.center
        return best, wall_data, points[available]

    def _validate_cache(self, visible, walls, now):
        if self.cache is None:
            return
        cache = self.cache
        keep_from = max(0, np.searchsorted(cache.arc, self.progress - 1.0) - 1)
        self.progress = max(0.0, self.progress - cache.arc[keep_from])
        cache = cache.part(start=keep_from)
        ahead = cache.arc >= self.progress
        rejected = ~cache.persistent & (now - cache.seen > self.cfg.max_age) & ahead
        if len(visible):
            distance, _ = cKDTree(visible).query(cache.center)
            rejected |= (distance < self.cfg.clearance + self.cfg.sample_step / 2) & ahead
        if walls is not None and any(len(wall) >= 2 for wall, _ in walls):
            starts, ends = segments(walls[0][0], walls[1][0])
            index = max(0, np.searchsorted(cache.arc, self.progress) - 1)
            count = validate_prefix(cache.center[index:], starts, ends, self.cfg)
            if index + count < len(cache.center):
                rejected[index + count:] = True
        failures = np.flatnonzero(rejected)
        if len(failures):
            cache = cache.part(end=int(failures[0]))
        self.cache = cache if len(cache.center) >= 2 else None

    def plan(self, car, yaw, now):
        """Advance the planning state using current pose and accumulated observations."""
        cfg = self.cfg
        car = np.asarray(car, dtype=float)
        heading = np.array([math.cos(yaw), math.sin(yaw)])
        dt = 0.0 if self.last_time is None else max(0.0, now - self.last_time)
        if self.last_time is not None and (now < self.last_time or np.linalg.norm(car - self.last_car) > 2.0):
            self.reset(keep_path_memory=now >= self.last_time)
            return Result('no_valid_path', np.empty((0, 2)))
        travel = 0.0 if self.last_car is None else float(np.linalg.norm(car - self.last_car))
        self.last_time, self.last_car = now, car
        if self.cache is not None:
            station = project(self.cache, car, heading, self.progress, max(1.0, 3 * travel))
            if station is None and self.progress < EPS and entry_available(self.cache, car, heading, cfg):
                station = 0.0
            if station is None:
                self.cache = self.held_probe = None
                self.progress = 0.0
                self.state = 'no_valid_path'
                return Result(self.state, np.empty((0, 2)))
            self.progress = station
        probe = self._probe(car, heading)
        build = self.last_build is None or now - self.last_build >= 0.12
        visible, walls = np.empty((0, 2)), None
        if build:
            self.last_revision = self.revision
            self.last_build = now
            candidate, walls, observed = self._assemble(car, heading, probe, now)
            if observed is not None:
                visible = observed
            if candidate is not None:
                station = project(candidate, car, heading)
                if station is None and entry_available(candidate, car, heading, cfg):
                    station = 0.0
                if station is not None:
                    if self.cache is None:
                        self.progress = station
                        self.cache = candidate
                    else:
                        self.cache = self._join_candidate(candidate)
                        if self.cache is candidate:
                            self.progress = station
                    latest = float(np.max(candidate.seen))
                    # Refresh width from genuinely newer supporting evidence.
                    if latest > self.width_stamp:
                        recent = candidate.seen >= latest - 0.4
                        self.width = float(np.median(np.linalg.norm(candidate.left[recent] - candidate.right[recent], axis=1)))
                        self.width_stamp = latest
            if min(self.counts) <= 1 and self.state != 'search':
                self.held_probe = probe if self.cache is not None else None
                self.state = 'search'
                self.good_updates = 0
                self.recovery_revision = self.last_revision
            if self.state == 'search' and self.recovery_revision != self.last_revision:
                self.recovery_revision = self.last_revision
                supported = candidate is not None and self.cache is not None
                if supported and min(self.counts) >= 2 and self.cache.confirmed[-1]:
                    self.good_updates += 1
                    if self.good_updates >= cfg.recovery_updates:
                        self.state, self.held_probe = 'track', None
                        self.lead = 0.0
                else:
                    self.good_updates = 0
            if self.cache is not None and self.state == 'no_valid_path' and min(self.counts) >= 2:
                self.state = 'track'
            if self.cache is not None and walls is not None and self.state == 'search':
                if (min(self.counts) <= 1 < max(self.counts) and now - self.width_stamp <= cfg.width_age
                        and np.any(self.cache.confirmed)):
                    side = 0 if self.counts[0] > self.counts[1] else 1
                    self.cache = provisional_tail(self.cache, *walls[side], 1 if side == 0 else -1, self.width, cfg)
        self._validate_cache(visible, walls, now)
        if self.cache is not None and self.cache.arc[-1] > self.progress + 0.20:
            # Include the segment containing the car's projection for controller tracking.
            first = max(0, np.searchsorted(self.cache.arc, self.progress, side='right') - 1)
            points = self.cache.center[first:]
            entry = self.progress < EPS and entry_available(self.cache, car, heading, cfg)
            if self.state == 'no_valid_path':
                self.state = 'track' if min(self.counts) >= 2 else 'search'
                if self.state == 'search':
                    self.held_probe = probe
            self.lead = min(cfg.lookahead, self.lead + 1.5 * dt)
            self.startup_scan_complete = True
            self.remembered_path_points.update(map(tuple, self.cache.center[self.cache.persistent]))
            provisional = bool(entry or not self.cache.confirmed[first:].all())
            return Result(self.state, points, float(self.cache.arc[-1] - self.progress),
                          provisional, self._probe(car, heading), self.anchors, self.counts)
        if self.cache is None:
            self.progress = 0.0
        self.state, self.held_probe = 'no_valid_path', None
        return Result(self.state, np.empty((0, 2)), probe=probe, anchors=self.anchors, cone_counts=self.counts)
