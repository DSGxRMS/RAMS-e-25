#!/usr/bin/env python3
# skidpad_debug.py
#
# Live debug plotter for /path_points (nav_msgs/Path) + /slam/map_cones,
# now with ellipse-based cone matching and virtual ("ghost") cone generation.
#
# Original behaviour (unchanged):
#   - Plots cones and car pose.
#   - Plots received path points as scatter (NOT joined).
#   - Fits and plots a B-spline through the received poses.
#
# Added:
#   - CONE MATCHING / GHOST CONES (ported from fsd_path_planning/cone_matching,
#     with no fsd_path_planning dependency):
#       * Blue/yellow cones are sorted into traces (greedy NN chain from the car).
#       * Each cone gets a match search direction: the local trace tangent
#         rotated across the track.
#       * Around each cone an ELLIPSE is built in the frame where that search
#         direction is +x: x^2/major^2 + y^2/minor^2 < 1. Cones of the other
#         side inside the ellipse, inside the search angle, and whose own
#         search direction points back at us are match candidates; the two
#         nearest survive and the nearest is selected (optionally monotonic).
#       * Cones with NO match spawn a VIRTUAL cone at the far end of the
#         ellipse axis along the search direction, at min_track_width (or at
#         the ellipse minor_radius if virtual.place_at_axis_end).
#       * Ghosts are inserted into the other side's trace at the correct index
#         and hairpins (<85 deg) created by insertion are pruned.
#       * Matching is re-run on the augmented traces for the final matches.
#   - Ghost cones are drawn as hollow markers, matched pairs as thin link
#     lines, and pair midpoints as "matched centers".
#   - The B-spline can be fitted through the planner's /path_points (default,
#     original behaviour) or through the matched centers, via spline.source.
#   - Optional /slam/odom fallback so the node still works (and still plans)
#     before any /path_points message arrives.
#
# Notes:
# - Requires: matplotlib, numpy. SciPy is optional (spline only).
# - If SciPy is missing, this node will log an error and plot only the scatter points.

import math
import threading
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
from eufs_msgs.msg import ConeArrayWithCovariance

import numpy as np

# Matplotlib
import matplotlib.pyplot as plt

# SciPy spline fitting (preferred)
try:
    from scipy.interpolate import splprep, splev
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def yaw_from_quat(qx, qy, qz, qw) -> float:
    """Return yaw (rad) from quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


# ===========================================================================
#  Small self-contained math utils (replacements for fsd_path_planning.utils)
# ===========================================================================

LEFT = 0   # blue cones
RIGHT = 1  # yellow cones

_EPS = 1e-9


def unit_2d(vecs: np.ndarray) -> np.ndarray:
    """Row-wise normalisation of an (..., 2) array. Zero rows stay zero."""
    vecs = np.asarray(vecs, dtype=float)
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    safe = np.where(norms < _EPS, 1.0, norms)
    out = vecs / safe
    out[np.broadcast_to(norms < _EPS, out.shape)] = 0.0
    return out


def rotate(points: np.ndarray, angle: float) -> np.ndarray:
    """Rotate an (..., 2) array of points by `angle` radians (CCW)."""
    points = np.asarray(points, dtype=float)
    c, s = math.cos(angle), math.sin(angle)
    rot = np.array([[c, -s], [s, c]], dtype=float)
    return points @ rot.T


def angle_from_2d_vector(vecs: np.ndarray) -> np.ndarray:
    """Heading angle of each (..., 2) vector, in (-pi, pi]."""
    vecs = np.asarray(vecs, dtype=float)
    return np.arctan2(vecs[..., 1], vecs[..., 0])


def vec_angle_between(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Unsigned angle in [0, pi] between two (broadcastable) (..., 2) arrays.
    Uses atan2(|cross|, dot) which is numerically stable near 0 and pi.
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    dot = np.sum(v1 * v2, axis=-1)
    cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
    return np.abs(np.arctan2(cross, dot))


def cdist_sq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared euclidean distance matrix, shape (len(a), len(b))."""
    a = np.asarray(a, dtype=float).reshape(-1, 2)
    b = np.asarray(b, dtype=float).reshape(-1, 2)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=float)
    diff = a[:, None, :] - b[None, :, :]
    return np.sum(diff * diff, axis=-1)


def trace_angles_between(trace: np.ndarray) -> np.ndarray:
    """
    Interior angle at every non-endpoint vertex of a polyline.
    Returns an array of length len(trace) - 2 (empty if len(trace) < 3).
    A small angle means a hairpin / doubling back.
    """
    trace = np.asarray(trace, dtype=float)
    if len(trace) < 3:
        return np.zeros(0, dtype=float)
    to_prev = trace[:-2] - trace[1:-1]
    to_next = trace[2:] - trace[1:-1]
    return vec_angle_between(to_prev, to_next)


# ===========================================================================
#  Trace sorting (cones arrive unordered from the SLAM map)
# ===========================================================================

def sort_cones_greedy(
    cones: np.ndarray,
    car_pos: np.ndarray,
    car_dir: np.ndarray,
    max_gap: float,
    turn_penalty: float = 2.0,
    relaxed_gap_factor: float = 2.0,
) -> np.ndarray:
    """
    Order a set of same-colour cones into a trace, starting from the cone
    closest to the car (preferring cones in front) and chaining nearest
    neighbours. Sharp direction changes are penalised so the chain follows the
    track instead of cutting across it.

    If no cone is within `max_gap`, a single relaxed hop up to
    `max_gap * relaxed_gap_factor` is allowed, but only almost straight ahead
    (< 45 deg). That keeps a sparsely mapped trace in one piece without
    letting the chain jump to the other side of the track.
    """
    cones = np.asarray(cones, dtype=float).reshape(-1, 2)
    n = len(cones)
    if n <= 1:
        return cones.copy()

    rel = cones - car_pos
    dist_to_car = np.linalg.norm(rel, axis=1)
    forward = rel @ car_dir
    start_cost = dist_to_car + np.where(forward < 0.0, 1e3, 0.0)
    start = int(np.argmin(start_cost))

    used = np.zeros(n, dtype=bool)
    used[start] = True
    order = [start]

    current = cones[start]
    current_dir = np.asarray(car_dir, dtype=float)

    for _ in range(n - 1):
        vecs = cones - current
        dists = np.linalg.norm(vecs, axis=1)
        angles = vec_angle_between(unit_2d(vecs), current_dir)

        reachable = (~used) & (dists > _EPS)
        valid = reachable & (dists <= max_gap) & (angles <= math.radians(110.0))

        if not valid.any():
            # second chance: one longer hop, but it must be nearly straight on
            valid = (
                reachable
                & (dists <= max_gap * relaxed_gap_factor)
                & (angles <= math.radians(45.0))
            )
            if not valid.any():
                break

        cost = dists + turn_penalty * angles
        cost[~valid] = np.inf

        idx = int(np.argmin(cost))
        new_dir = cones[idx] - current
        norm = np.linalg.norm(new_dir)
        if norm > _EPS:
            current_dir = new_dir / norm
        current = cones[idx]
        used[idx] = True
        order.append(idx)

    return cones[np.asarray(order, dtype=int)]


def calculate_match_search_direction(cones: np.ndarray, cone_type: int) -> np.ndarray:
    """
    For every cone of a *sorted* trace, the unit direction in which to look for
    a match on the other side: the local trace tangent rotated across the
    track. Left (blue) cones look to their right, right (yellow) cones look to
    their left.
    """
    cones = np.asarray(cones, dtype=float).reshape(-1, 2)
    n = len(cones)
    if n < 2:
        return np.zeros((n, 2), dtype=float)

    tangents = np.zeros((n, 2), dtype=float)
    tangents[0] = cones[1] - cones[0]
    tangents[-1] = cones[-1] - cones[-2]
    if n > 2:
        tangents[1:-1] = cones[2:] - cones[:-2]

    tangents = unit_2d(tangents)
    rot = -math.pi / 2.0 if cone_type == LEFT else math.pi / 2.0
    return unit_2d(rotate(tangents, rot))


# ===========================================================================
#  Ellipse-based matching  (ported from functional_cone_matching.py)
# ===========================================================================

def find_boolean_mask_of_all_potential_matches(
    start_points: np.ndarray,
    directions: np.ndarray,
    other_side_cones: np.ndarray,
    other_side_directions: np.ndarray,
    major_radius: float,
    minor_radius: float,
    max_search_angle: float,
) -> np.ndarray:
    """
    (M, N) boolean mask: for each cone on our side, which cones of the other
    side are acceptable match candidates.

    Geometry: the frame of each start point is rotated so that its search
    direction becomes +x. In that frame the acceptance region is the ellipse

        x^2 / major_radius^2 + y^2 / minor_radius^2 < 1

    so `major_radius` extends *across the track* (along the search direction)
    and `minor_radius` limits the lateral spread. On top of the ellipse test,
    candidates outside the search angle are rejected, and so are candidates
    whose own search direction does not point back towards us. Finally only
    the two closest survivors are kept.
    """
    start_points = np.asarray(start_points, dtype=float).reshape(-1, 2)
    other_side_cones = np.asarray(other_side_cones, dtype=float).reshape(-1, 2)

    return_value = np.zeros((len(start_points), len(other_side_cones)), dtype=bool)
    if len(start_points) == 0 or len(other_side_cones) == 0:
        return return_value

    radii_square = np.array([major_radius, minor_radius], dtype=float) ** 2

    angles = angle_from_2d_vector(directions)

    # (M, N, 2)
    from_start_points_to_other_side = other_side_cones - start_points[:, None]

    have_other_dirs = (
        other_side_directions is not None
        and len(other_side_directions) == len(other_side_cones)
        and np.any(np.linalg.norm(other_side_directions, axis=1) > _EPS)
    )
    if have_other_dirs:
        direction_diff = vec_angle_between(directions[:, None], other_side_directions)
    else:
        direction_diff = None

    for i in range(len(start_points)):
        rotated = rotate(from_start_points_to_other_side[i], -angles[i])

        inside_ellipse = (rotated ** 2 / radii_square).sum(axis=1) < 1.0
        return_value[i] = inside_ellipse

        angle_of_rotated = angle_from_2d_vector(rotated)
        over_angle = np.abs(angle_of_rotated / 2.0) > max_search_angle
        return_value[i, over_angle] = False

        if direction_diff is not None:
            # the other cone must "face" us: its search direction should be
            # roughly opposite to ours (angle between them > 90 deg)
            not_facing = direction_diff[i] < (math.pi / 2.0)
            return_value[i, not_facing] = False

    # keep at most the 2 nearest candidates per cone
    distances = np.linalg.norm(from_start_points_to_other_side, axis=-1)
    for i in range(len(start_points)):
        d = distances[i].copy()
        d[~return_value[i]] = np.inf
        idxs = np.argsort(d)[:2]
        idxs = idxs[np.isfinite(d[idxs])]
        new_mask = np.zeros_like(return_value[i])
        new_mask[idxs] = True
        return_value[i] = new_mask

    return return_value


def select_best_match_candidate(
    matchable_cones: np.ndarray,
    match_boolean_mask: np.ndarray,
    other_side_cones: np.ndarray,
    matches_should_be_monotonic: bool,
) -> np.ndarray:
    """
    Pick one match per cone (index into `other_side_cones`), -1 if none.
    Unlike the original, the distance argmin is restricted to cones allowed by
    the mask, so a rejected-but-closer cone can no longer win.
    """
    matchable_cones = np.asarray(matchable_cones, dtype=float).reshape(-1, 2)
    if len(other_side_cones) == 0 or len(matchable_cones) == 0:
        return np.full(len(matchable_cones), -1, dtype=int)

    dists = cdist_sq(matchable_cones, other_side_cones)
    dists = np.where(match_boolean_mask, dists, np.inf)

    matched = np.argmin(dists, axis=1).astype(int)
    matched[~np.isfinite(np.min(dists, axis=1))] = -1

    if matches_should_be_monotonic:
        current_max = matched[0]
        for i in range(1, len(matched)):
            if matched[i] == -1:
                continue
            current_max = max(current_max, matched[i])
            if matched[i] != current_max:
                matched[i] = -1
            else:
                matched[i] = current_max

    matched[~match_boolean_mask.any(axis=1)] = -1
    return matched


def calculate_positions_of_virtual_cones(
    cones: np.ndarray,
    indices_of_unmatched_cones: np.ndarray,
    search_directions: np.ndarray,
    placement_distance: float,
) -> np.ndarray:
    """
    Ghost cones sit at the far end of the ellipse axis that runs along the
    search direction, i.e. cone + search_direction * placement_distance.
    """
    cones = np.asarray(cones, dtype=float).reshape(-1, 2)
    if len(indices_of_unmatched_cones) == 0 or len(search_directions) != len(cones):
        return np.zeros((0, 2), dtype=float)

    idx = np.asarray(indices_of_unmatched_cones, dtype=int)
    dirs = search_directions[idx]
    # a cone with no usable direction cannot spawn a ghost
    keep = np.linalg.norm(dirs, axis=1) > _EPS
    return cones[idx][keep] + dirs[keep] * placement_distance


def calculate_insert_index_for_one_cone(
    car_position: np.ndarray, final_cones: np.ndarray, virtual_cone: np.ndarray
) -> int:
    d_virtual = float(np.linalg.norm(virtual_cone - car_position))
    d_existing = float(np.linalg.norm(final_cones[0] - car_position))
    return 0 if d_virtual < d_existing else 1


def calculate_insert_index_of_new_cone(
    closest_index: int, second_closest_index: int, cone_is_between_closest_two: bool
) -> int:
    if cone_is_between_closest_two:
        return min(closest_index, second_closest_index) + 1
    if closest_index < second_closest_index:
        return closest_index
    return closest_index + 1


def insert_virtual_cones_to_existing(
    other_side_cones: np.ndarray,
    other_side_virtual_cones: np.ndarray,
    car_position: np.ndarray,
    min_corner_angle_deg: float = 85.0,
) -> np.ndarray:
    """
    Merge ghost cones into the real trace at the correct position, then drop
    vertices that create a hairpin (interior angle below the threshold).
    """
    existing_cones, cones_to_insert = (
        (other_side_cones, other_side_virtual_cones)
        if len(other_side_cones) > len(other_side_virtual_cones)
        else (other_side_virtual_cones, other_side_cones)
    )
    existing_cones = np.asarray(existing_cones, dtype=float).reshape(-1, 2).copy()
    cones_to_insert = np.asarray(cones_to_insert, dtype=float).reshape(-1, 2).copy()

    if len(existing_cones) == 0:
        return cones_to_insert
    if len(cones_to_insert) == 0:
        return existing_cones

    order_to_insert = cdist_sq(cones_to_insert, existing_cones).min(axis=1).argsort()
    cones_to_insert = cones_to_insert[order_to_insert]

    for cone_to_insert in cones_to_insert:
        distance_to_existing = np.linalg.norm(existing_cones - cone_to_insert, axis=1)
        indices_sorted = distance_to_existing.argsort()

        if len(indices_sorted) == 1:
            index_to_insert = calculate_insert_index_for_one_cone(
                car_position, existing_cones, cone_to_insert
            )
        else:
            closest_index, second_closest_index = (
                int(indices_sorted[0]),
                int(indices_sorted[1]),
            )
            if abs(closest_index - second_closest_index) != 1:
                continue

            to_closest = existing_cones[closest_index] - cone_to_insert
            to_second = existing_cones[second_closest_index] - cone_to_insert
            is_between = bool(vec_angle_between(to_closest, to_second) > math.pi / 2.0)

            index_to_insert = calculate_insert_index_of_new_cone(
                closest_index, second_closest_index, is_between
            )

        existing_cones = np.insert(existing_cones, index_to_insert, cone_to_insert, axis=0)

    if len(existing_cones) >= 3:
        angles = trace_angles_between(existing_cones)
        mask_low = angles < math.radians(min_corner_angle_deg)
        mask_low = np.concatenate([[False], mask_low, [False]])
        if mask_low.any():
            existing_cones = existing_cones[~mask_low]

    return existing_cones


def combine_and_sort_virtual_with_real(
    other_side_cones: np.ndarray,
    other_side_virtual_cones: np.ndarray,
    car_pos: np.ndarray,
    min_corner_angle_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (combined_cones, mask_is_virtual)."""
    if len(other_side_cones) == 0:
        return (
            other_side_virtual_cones,
            np.ones(len(other_side_virtual_cones), dtype=bool),
        )
    if len(other_side_virtual_cones) == 0:
        return other_side_cones, np.zeros(len(other_side_cones), dtype=bool)

    combined = insert_virtual_cones_to_existing(
        other_side_cones, other_side_virtual_cones, car_pos, min_corner_angle_deg
    )

    epsilon = 1e-2
    dist_to_real = cdist_sq(combined, other_side_cones)
    mask_is_virtual = np.all(dist_to_real > epsilon ** 2, axis=1)
    return combined, mask_is_virtual


def calculate_matches_for_side(
    cones: np.ndarray,
    cone_type: int,
    other_side_cones: np.ndarray,
    major_radius: float,
    minor_radius: float,
    max_search_angle: float,
    matches_should_be_monotonic: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (matchable_cones, match_index_per_cone, search_directions)."""
    matchable_cones = np.asarray(cones, dtype=float).reshape(-1, 2)

    if len(matchable_cones) <= 1:
        return (
            matchable_cones,
            np.full(len(matchable_cones), -1, dtype=int),
            np.zeros((len(matchable_cones), 2), dtype=float),
        )

    search_directions = calculate_match_search_direction(matchable_cones, cone_type)

    other_side_cones = np.asarray(other_side_cones, dtype=float).reshape(-1, 2)
    if len(other_side_cones) > 1:
        other_side_search_directions = calculate_match_search_direction(
            other_side_cones, RIGHT if cone_type == LEFT else LEFT
        )
    else:
        other_side_search_directions = np.zeros((len(other_side_cones), 2), dtype=float)

    mask = find_boolean_mask_of_all_potential_matches(
        matchable_cones,
        search_directions,
        other_side_cones,
        other_side_search_directions,
        major_radius,
        minor_radius,
        max_search_angle,
    )

    matches = select_best_match_candidate(
        matchable_cones, mask, other_side_cones, matches_should_be_monotonic
    )
    return matchable_cones, matches, search_directions


def calculate_cones_for_other_side(
    cones: np.ndarray,
    cone_type: int,
    other_side_cones: np.ndarray,
    major_radius: float,
    minor_radius: float,
    max_search_angle: float,
    min_track_width: float,
    car_pos: np.ndarray,
    matches_should_be_monotonic: bool,
    min_corner_angle_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the other side's cones, filling holes with ghost cones."""
    matchable_cones, matches, search_directions = calculate_matches_for_side(
        cones,
        cone_type,
        other_side_cones,
        major_radius,
        minor_radius,
        max_search_angle,
        matches_should_be_monotonic,
    )

    indices_no_match = np.where(matches == -1)[0]
    virtual_positions = calculate_positions_of_virtual_cones(
        matchable_cones, indices_no_match, search_directions, min_track_width
    )

    combined, mask_is_virtual = combine_and_sort_virtual_with_real(
        other_side_cones, virtual_positions, car_pos, min_corner_angle_deg
    )

    if len(combined) < 2:
        combined = np.asarray(other_side_cones, dtype=float).reshape(-1, 2)
        mask_is_virtual = np.zeros(len(combined), dtype=bool)

    return combined, mask_is_virtual


def match_both_sides_with_virtual_cones(
    left_cones_with_virtual: np.ndarray,
    right_cones_with_virtual: np.ndarray,
    major_radius: float,
    minor_radius: float,
    max_search_angle: float,
    matches_should_be_monotonic: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    _, left_to_right, _ = calculate_matches_for_side(
        left_cones_with_virtual,
        LEFT,
        right_cones_with_virtual,
        major_radius,
        minor_radius,
        max_search_angle,
        matches_should_be_monotonic,
    )
    _, right_to_left, _ = calculate_matches_for_side(
        right_cones_with_virtual,
        RIGHT,
        left_cones_with_virtual,
        major_radius,
        minor_radius,
        max_search_angle,
        matches_should_be_monotonic,
    )
    return left_to_right, right_to_left


def calculate_virtual_cones_for_both_sides(
    left_cones: np.ndarray,
    right_cones: np.ndarray,
    car_position: np.ndarray,
    min_track_width: float,
    major_radius: float,
    minor_radius: float,
    max_search_angle: float,
    matches_should_be_monotonic: bool = True,
    min_corner_angle_deg: float = 85.0,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Main entry point. Returns, for each side, a tuple of
    (cones_including_virtual, mask_is_virtual, match_index_to_other_side).
    """
    empty_cones = np.zeros((0, 2), dtype=float)
    empty_bool = np.zeros(0, dtype=bool)
    empty_int = np.zeros(0, dtype=int)
    dummy = (empty_cones, empty_bool, empty_int)

    left_cones = np.asarray(left_cones, dtype=float).reshape(-1, 2)
    right_cones = np.asarray(right_cones, dtype=float).reshape(-1, 2)

    if len(left_cones) < 2 and len(right_cones) < 2:
        return dummy, dummy

    # if one side is far better observed than the other, trust only that side
    min_len = min(len(left_cones), len(right_cones))
    max_len = max(len(left_cones), len(right_cones))
    if min_len == 0 or (max_len / min_len) > 2:
        if len(left_cones) < len(right_cones):
            left_cones = empty_cones
        else:
            right_cones = empty_cones

    if len(left_cones) >= 2:
        right_with_virtual, right_mask_virtual = calculate_cones_for_other_side(
            left_cones,
            LEFT,
            right_cones,
            major_radius,
            minor_radius,
            max_search_angle,
            min_track_width,
            car_position,
            matches_should_be_monotonic,
            min_corner_angle_deg,
        )
    else:
        right_with_virtual = right_cones
        right_mask_virtual = np.zeros(len(right_cones), dtype=bool)

    if len(right_cones) >= 2:
        left_with_virtual, left_mask_virtual = calculate_cones_for_other_side(
            right_cones,
            RIGHT,
            left_cones,
            major_radius,
            minor_radius,
            max_search_angle,
            min_track_width,
            car_position,
            matches_should_be_monotonic,
            min_corner_angle_deg,
        )
    else:
        left_with_virtual = left_cones
        left_mask_virtual = np.zeros(len(left_cones), dtype=bool)

    left_to_right, right_to_left = match_both_sides_with_virtual_cones(
        left_with_virtual,
        right_with_virtual,
        major_radius,
        minor_radius,
        max_search_angle,
        matches_should_be_monotonic,
    )

    return (
        (left_with_virtual, left_mask_virtual, left_to_right),
        (right_with_virtual, right_mask_virtual, right_to_left),
    )


# ===========================================================================
#  Debug plotter node
# ===========================================================================

class PPDebug(Node):
    def __init__(self):
        super().__init__("pp_debug")

        # ---------------- Parameters ----------------
        self.declare_parameter("topics.path_in", "/path_points")
        self.declare_parameter("topics.map_in", "/slam/map_cones")
        self.declare_parameter("topics.odom_in", "/slam/odom")

        self.declare_parameter("plot.hz", 20.0)

        # Spline params
        self.declare_parameter("spline.include_car_pose", True)   # include pose[0] in spline fit
        self.declare_parameter("spline.smoothing", 0.5)           # s in splprep (0 = interpolate)
        self.declare_parameter("spline.num_samples", 200)
        # "path"           -> fit through /path_points (original behaviour)
        # "matched_centers"-> fit through the midpoints of the matched cone pairs
        self.declare_parameter("spline.source", "path")

        # Plotting params
        self.declare_parameter("plot.window_lock", False)         # if True, do not auto-rescale
        self.declare_parameter("plot.padding_m", 5.0)
        self.declare_parameter("plot.show_virtual_cones", True)
        self.declare_parameter("plot.show_matches", True)
        self.declare_parameter("plot.show_matched_centers", True)

        # ---- Cone matching / virtual cones ----
        self.declare_parameter("matching.enabled", True)
        self.declare_parameter("matching.major_radius_m", 6.5)
        self.declare_parameter("matching.minor_radius_m", 3.5)
        self.declare_parameter("matching.max_search_angle_deg", 50.0)
        self.declare_parameter("matching.monotonic", True)
        self.declare_parameter("matching.min_corner_angle_deg", 85.0)
        self.declare_parameter("matching.max_pair_width_m", 8.0)
        # only match cones within this radius of the car (0 = no limit)
        self.declare_parameter("matching.roi_radius_m", 30.0)

        self.declare_parameter("virtual.min_track_width_m", 3.0)
        # place the ghost at the ellipse axis end instead of at min_track_width
        self.declare_parameter("virtual.place_at_axis_end", False)

        # ---- Trace sorting ----
        self.declare_parameter("trace.max_gap_m", 6.0)
        self.declare_parameter("trace.turn_penalty", 2.0)

        # QoS
        self.declare_parameter("qos.best_effort", True)
        self.declare_parameter("qos.depth", 50)

        gp = self.get_parameter
        self.path_topic = str(gp("topics.path_in").value)
        self.map_topic = str(gp("topics.map_in").value)
        self.odom_topic = str(gp("topics.odom_in").value)

        self.plot_hz = float(gp("plot.hz").value)
        self.plot_period = 1.0 / max(1e-6, self.plot_hz)

        self.include_car_pose = bool(gp("spline.include_car_pose").value)
        self.spline_s = float(gp("spline.smoothing").value)
        self.spline_num = int(gp("spline.num_samples").value)
        self.spline_source = str(gp("spline.source").value).strip().lower()

        self.window_lock = bool(gp("plot.window_lock").value)
        self.padding_m = float(gp("plot.padding_m").value)
        self.show_virtual_cones = bool(gp("plot.show_virtual_cones").value)
        self.show_matches = bool(gp("plot.show_matches").value)
        self.show_matched_centers = bool(gp("plot.show_matched_centers").value)

        self.matching_enabled = bool(gp("matching.enabled").value)
        self.major_radius = float(gp("matching.major_radius_m").value)
        self.minor_radius = float(gp("matching.minor_radius_m").value)
        self.max_search_angle = math.radians(float(gp("matching.max_search_angle_deg").value))
        self.matches_monotonic = bool(gp("matching.monotonic").value)
        self.min_corner_angle_deg = float(gp("matching.min_corner_angle_deg").value)
        self.max_pair_width = float(gp("matching.max_pair_width_m").value)
        self.roi_radius = float(gp("matching.roi_radius_m").value)

        self.min_track_width = float(gp("virtual.min_track_width_m").value)
        self.place_at_axis_end = bool(gp("virtual.place_at_axis_end").value)
        # distance at which a ghost cone is placed along the search direction
        self.ghost_distance = (
            self.minor_radius if self.place_at_axis_end else self.min_track_width
        )

        self.trace_max_gap = float(gp("trace.max_gap_m").value)
        self.trace_turn_penalty = float(gp("trace.turn_penalty").value)

        best_effort = bool(gp("qos.best_effort").value)
        depth = int(gp("qos.depth").value)

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=depth,
            reliability=QoSReliabilityPolicy.BEST_EFFORT if best_effort else QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # ---------------- State ----------------
        self.lock = threading.Lock()

        # Latest path message info
        self.frame_id: str = "map"
        self.car_pose: Optional[Tuple[float, float, float]] = None  # (x,y,yaw)
        self.path_points: List[Tuple[float, float]] = []            # remaining points (excluding car pose)

        # Cones
        self.blue: List[Tuple[float, float]] = []
        self.yellow: List[Tuple[float, float]] = []
        self.orange: List[Tuple[float, float]] = []
        self.big: List[Tuple[float, float]] = []

        # Matching results (computed in timer_cb, plot-only state)
        self.left_cones_with_virtual = np.zeros((0, 2))
        self.right_cones_with_virtual = np.zeros((0, 2))
        self.left_is_virtual = np.zeros(0, dtype=bool)
        self.right_is_virtual = np.zeros(0, dtype=bool)
        self.matched_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        self.matched_centers: List[Tuple[float, float]] = []

        # ---------------- Subscriptions ----------------
        self.create_subscription(Path, self.path_topic, self.cb_path, qos)
        self.create_subscription(ConeArrayWithCovariance, self.map_topic, self.cb_map, qos)
        # fallback pose source, so matching works before any path is published
        self.create_subscription(Odometry, self.odom_topic, self.cb_odom, qos)

        # ---------------- Plot setup ----------------
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("pp_debug: /path_points B-spline + cones + ghost cones + car pose")
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True)

        # Persistent artists (initialized empty)
        self.scat_blue = self.ax.scatter([], [], c="tab:blue", s=28, label="blue cones")
        self.scat_yellow = self.ax.scatter([], [], c="gold", s=28, label="yellow cones")
        self.scat_orange = self.ax.scatter([], [], c="tab:orange", s=28, label="orange cones")
        self.scat_big = self.ax.scatter(
            [], [], c="darkorange", s=90, marker="s", label="big orange cones"
        )

        self.scat_pts = self.ax.scatter(
            [], [], c="red", s=22, marker="x", label="path points (raw)"
        )

        # Ghost / virtual cones: hollow markers in the colour of their side
        self.scat_ghost_blue = self.ax.scatter(
            [], [], facecolors="none", edgecolors="tab:blue", s=70, marker="o",
            linewidths=1.5, label="virtual blue",
        )
        self.scat_ghost_yellow = self.ax.scatter(
            [], [], facecolors="none", edgecolors="goldenrod", s=70, marker="o",
            linewidths=1.5, label="virtual yellow",
        )
        self.scat_centers = self.ax.scatter(
            [], [], facecolors="none", edgecolors="purple", s=45, marker="s",
            linewidths=1.2, label="matched centers",
        )

        # One Line2D holding every match link, separated by NaN gaps
        self.match_lines, = self.ax.plot(
            [], [], linewidth=0.7, color="gray", alpha=0.7, label="matches"
        )

        self.car_dot, = self.ax.plot(
            [], [], marker="o", color="black", markersize=8, linestyle="", label="car"
        )
        self.car_arrow = None

        self.spline_line, = self.ax.plot(
            [], [], linewidth=2.0, color="red", label="B-spline"
        )

        # legend outside the axes: with ghosts + matches it covers the track
        self.ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
        self.fig.subplots_adjust(right=0.78)

        if not _HAS_SCIPY:
            self.get_logger().error(
                "[pp_debug] SciPy not available. Install python3-scipy / pip scipy. "
                "Spline will NOT be drawn."
            )

        if self.spline_source not in ("path", "matched_centers"):
            self.get_logger().warn(
                f"[pp_debug] unknown spline.source '{self.spline_source}', using 'path'."
            )
            self.spline_source = "path"

        self.get_logger().info(
            f"[pp_debug] Subscribing path={self.path_topic}, cones={self.map_topic}, "
            f"odom={self.odom_topic}, plot_hz={self.plot_hz} | "
            f"matching={'on' if self.matching_enabled else 'off'} "
            f"ellipse=({self.major_radius}, {self.minor_radius}) "
            f"ghost_dist={self.ghost_distance}m | spline_source={self.spline_source}"
        )

        # Timer to refresh plot
        self.create_timer(self.plot_period, self.timer_cb)

    # ---------------- Callbacks ----------------
    def cb_path(self, msg: Path):
        poses = msg.poses
        if not poses:
            return

        with self.lock:
            self.frame_id = msg.header.frame_id.strip() or "map"

            # Pose[0] is current car pose (as published by your upstream node)
            p0 = poses[0].pose
            x0 = float(p0.position.x)
            y0 = float(p0.position.y)
            q = p0.orientation
            yaw0 = yaw_from_quat(q.x, q.y, q.z, q.w)
            self.car_pose = (x0, y0, yaw0)

            # Remaining poses are future points
            pts = []
            for ps in poses[1:]:
                pts.append((float(ps.pose.position.x), float(ps.pose.position.y)))
            self.path_points = pts

    def cb_odom(self, msg: Odometry):
        """
        Fallback pose source. The path message carries the car pose in pose[0],
        but before the planner publishes anything (or if it stops), odom keeps
        the car pose - and therefore the cone matching - alive.
        """
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

        with self.lock:
            if self.car_pose is None:
                self.frame_id = msg.header.frame_id.strip() or self.frame_id
            self.car_pose = (x, y, yaw)

    def cb_map(self, msg: ConeArrayWithCovariance):
        blue, yellow, orange, big = [], [], [], []

        for c in msg.blue_cones:
            blue.append((float(c.point.x), float(c.point.y)))
        for c in msg.yellow_cones:
            yellow.append((float(c.point.x), float(c.point.y)))
        for c in msg.orange_cones:
            orange.append((float(c.point.x), float(c.point.y)))
        for c in msg.big_orange_cones:
            big.append((float(c.point.x), float(c.point.y)))

        with self.lock:
            self.blue = blue
            self.yellow = yellow
            self.orange = orange
            self.big = big

    # ---------------- Cone matching / ghost cones ----------------
    def _cones_in_roi(
        self, cones: List[Tuple[float, float]], car_xy: np.ndarray
    ) -> np.ndarray:
        """Restrict cones to a radius around the car (0 disables the limit)."""
        arr = np.asarray(cones, dtype=float).reshape(-1, 2)
        if self.roi_radius <= 0.0 or len(arr) == 0:
            return arr
        d = np.linalg.norm(arr - car_xy, axis=1)
        return arr[d <= self.roi_radius]

    def _run_cone_matching(
        self,
        blue: List[Tuple[float, float]],
        yellow: List[Tuple[float, float]],
        car: Tuple[float, float, float],
    ):
        """
        Sort both traces, run the ellipse matcher with ghost-cone generation,
        and collect the matched pairs and their midpoints.

        Returns (left_cones, left_is_virtual, right_cones, right_is_virtual,
                 pairs, centers).
        """
        empty = np.zeros((0, 2), dtype=float)
        cx, cy, cyaw = car
        car_xy = np.array([cx, cy], dtype=float)
        car_dir = np.array([math.cos(cyaw), math.sin(cyaw)], dtype=float)

        left_raw = self._cones_in_roi(blue, car_xy)
        right_raw = self._cones_in_roi(yellow, car_xy)

        left = sort_cones_greedy(
            left_raw, car_xy, car_dir, self.trace_max_gap, self.trace_turn_penalty
        )
        right = sort_cones_greedy(
            right_raw, car_xy, car_dir, self.trace_max_gap, self.trace_turn_penalty
        )

        if len(left) < 2 and len(right) < 2:
            return empty, np.zeros(0, dtype=bool), empty, np.zeros(0, dtype=bool), [], []

        (left_cones, left_is_virtual, left_to_right), (
            right_cones,
            right_is_virtual,
            right_to_left,
        ) = calculate_virtual_cones_for_both_sides(
            left,
            right,
            car_xy,
            self.ghost_distance,
            self.major_radius,
            self.minor_radius,
            self.max_search_angle,
            self.matches_monotonic,
            self.min_corner_angle_deg,
        )

        pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        centers: List[Tuple[float, float]] = []
        seen = set()
        max_w2 = self.max_pair_width ** 2

        def add_pair(p: np.ndarray, q: np.ndarray):
            d2 = float(np.sum((p - q) ** 2))
            if d2 > max_w2 or d2 < 1e-6:
                return
            mx = 0.5 * float(p[0] + q[0])
            my = 0.5 * float(p[1] + q[1])
            key = (round(mx, 3), round(my, 3))
            if key in seen:
                return
            seen.add(key)
            pairs.append((np.asarray(p, dtype=float), np.asarray(q, dtype=float)))
            centers.append((mx, my))

        for i, j in enumerate(left_to_right):
            if 0 <= j < len(right_cones):
                add_pair(left_cones[i], right_cones[int(j)])
        for i, j in enumerate(right_to_left):
            if 0 <= j < len(left_cones):
                add_pair(right_cones[i], left_cones[int(j)])

        # order the centers along the track so the spline follows the driving
        # direction instead of zig-zagging between duplicated matches
        if centers:
            centers_arr = sort_cones_greedy(
                np.asarray(centers, dtype=float),
                car_xy,
                car_dir,
                self.trace_max_gap,
                self.trace_turn_penalty,
            )
            centers = [(float(p[0]), float(p[1])) for p in centers_arr]

        return left_cones, left_is_virtual, right_cones, right_is_virtual, pairs, centers

    # ---------------- Spline helpers ----------------
    def _fit_bspline(self, pts: np.ndarray) -> Optional[np.ndarray]:
        """
        Fit a B-spline through pts (N,2). Returns sampled curve (M,2) or None.
        Uses SciPy splprep/splev.

        - For small N, automatically reduces spline degree.
        """
        if not _HAS_SCIPY:
            return None

        n = pts.shape[0]
        if n < 2:
            return None

        # Reduce degree if too few points (splprep requires m > k)
        k = min(3, n - 1)
        if k < 1:
            return None

        # Parameterize and fit
        x = pts[:, 0]
        y = pts[:, 1]

        # s controls smoothing; s=0 -> interpolating spline
        try:
            tck, _u = splprep([x, y], s=self.spline_s, k=k)
            uu = np.linspace(0.0, 1.0, max(20, self.spline_num))
            out = splev(uu, tck)
            curve = np.vstack(out).T  # (M,2)
            return curve
        except Exception as e:
            self.get_logger().warn(f"[pp_debug] Spline fit failed: {e}")
            return None

    # ---------------- Plot refresh ----------------
    def timer_cb(self):
        with self.lock:
            car = self.car_pose
            pts = list(self.path_points)

            blue = list(self.blue)
            yellow = list(self.yellow)
            orange = list(self.orange)
            big = list(self.big)

        # ---- Cone matching + ghost cone generation ----
        left_cones = right_cones = np.zeros((0, 2))
        left_virtual_mask = right_virtual_mask = np.zeros(0, dtype=bool)
        pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        centers: List[Tuple[float, float]] = []

        if self.matching_enabled and car is not None and (blue or yellow):
            try:
                (
                    left_cones,
                    left_virtual_mask,
                    right_cones,
                    right_virtual_mask,
                    pairs,
                    centers,
                ) = self._run_cone_matching(blue, yellow, car)
            except Exception as e:  # never let a bad frame kill the plotter
                self.get_logger().warn(f"[pp_debug] cone matching failed: {e}")

        with self.lock:
            self.left_cones_with_virtual = left_cones
            self.right_cones_with_virtual = right_cones
            self.left_is_virtual = left_virtual_mask
            self.right_is_virtual = right_virtual_mask
            self.matched_pairs = pairs
            self.matched_centers = centers

        # Update scatter data for cones
        def _set_scatter(sc, data):
            arr = np.asarray(data, dtype=float).reshape(-1, 2) if len(data) else np.empty((0, 2))
            sc.set_offsets(arr if len(arr) else np.empty((0, 2)))

        _set_scatter(self.scat_blue, blue)
        _set_scatter(self.scat_yellow, yellow)
        _set_scatter(self.scat_orange, orange)
        _set_scatter(self.scat_big, big)

        # Ghost / virtual cones
        if self.show_virtual_cones:
            gb = (
                left_cones[left_virtual_mask]
                if len(left_virtual_mask) == len(left_cones) and len(left_cones)
                else np.empty((0, 2))
            )
            gy = (
                right_cones[right_virtual_mask]
                if len(right_virtual_mask) == len(right_cones) and len(right_cones)
                else np.empty((0, 2))
            )
            _set_scatter(self.scat_ghost_blue, gb)
            _set_scatter(self.scat_ghost_yellow, gy)
        else:
            _set_scatter(self.scat_ghost_blue, np.empty((0, 2)))
            _set_scatter(self.scat_ghost_yellow, np.empty((0, 2)))

        # Match links: one polyline with NaN separators between segments
        if self.show_matches and pairs:
            mx: List[float] = []
            my: List[float] = []
            for p, q in pairs:
                mx.extend([p[0], q[0], math.nan])
                my.extend([p[1], q[1], math.nan])
            self.match_lines.set_data(mx, my)
        else:
            self.match_lines.set_data([], [])

        # Matched pair midpoints
        if self.show_matched_centers and centers:
            _set_scatter(self.scat_centers, centers)
        else:
            _set_scatter(self.scat_centers, np.empty((0, 2)))

        # Car pose
        if car is not None:
            cx, cy, cyaw = car
            self.car_dot.set_data([cx], [cy])

            # Update heading arrow
            if self.car_arrow is not None:
                try:
                    self.car_arrow.remove()
                except Exception:
                    pass

            arrow_len = 2.0
            hx = cx + arrow_len * math.cos(cyaw)
            hy = cy + arrow_len * math.sin(cyaw)
            self.car_arrow = self.ax.annotate(
                "",
                xy=(hx, hy),
                xytext=(cx, cy),
                arrowprops=dict(arrowstyle="->", linewidth=2.0),
            )
        else:
            self.car_dot.set_data([], [])

        # Raw path points scatter (NOT joined)
        if pts:
            self.scat_pts.set_offsets(np.array(pts, dtype=float))
        else:
            self.scat_pts.set_offsets(np.empty((0, 2)))

        # Build spline input points (car + source points)
        source_pts = centers if self.spline_source == "matched_centers" else pts

        spline_pts_list: List[Tuple[float, float]] = []
        if self.include_car_pose and car is not None:
            spline_pts_list.append((car[0], car[1]))
        spline_pts_list.extend(source_pts)

        spline_curve = None
        if len(spline_pts_list) >= 2:
            spline_curve = self._fit_bspline(np.array(spline_pts_list, dtype=float))

        if spline_curve is not None and spline_curve.shape[0] >= 2:
            self.spline_line.set_data(spline_curve[:, 0], spline_curve[:, 1])
        else:
            self.spline_line.set_data([], [])

        # Auto-scale view (unless locked)
        if not self.window_lock:
            all_xy = []
            if car is not None:
                all_xy.append((car[0], car[1]))
            all_xy.extend(pts)
            all_xy.extend(blue)
            all_xy.extend(yellow)
            all_xy.extend(orange)
            all_xy.extend(big)
            all_xy.extend(centers)
            if len(left_cones):
                all_xy.extend([tuple(p) for p in left_cones])
            if len(right_cones):
                all_xy.extend([tuple(p) for p in right_cones])

            if all_xy:
                arr = np.array(all_xy, dtype=float)
                xmin, ymin = arr.min(axis=0)
                xmax, ymax = arr.max(axis=0)
                pad = self.padding_m
                self.ax.set_xlim(xmin - pad, xmax + pad)
                self.ax.set_ylim(ymin - pad, ymax + pad)

        # Refresh
        self.fig.canvas.draw_idle()
        plt.pause(0.001)


def main():
    rclpy.init()
    node = PPDebug()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            plt.close("all")
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()