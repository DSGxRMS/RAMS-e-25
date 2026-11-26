#!/usr/bin/env python3
"""
pp_tels_new.py  -  team planner + PCD (Predictive Corridor Delaunay), version 2
(route memory, hairpin tips, curvature-limited line, orange-gate fix; pp_tls.py is v1)

Separate node next to the team's pp_node_skidpad (which stays untouched):
    ros2 run pathplanning pp_tels_new
    ros2 launch bringup stack_tels_new.launch.py (whole stack with pp_tels_new)
Same topics, QoS, parameters and /path_points format as pp_node_skidpad,
so the controller cannot tell the difference. Run ONE of the two planners.

What changed
------------
Every tick PCD is tried FIRST:
    memory of cones -> walls traced from same-colour Delaunay edges ->
    clothoid prediction into the blind part of a bend -> virtual partner
    cones where one wall is missing -> width-checked blue-yellow Delaunay
    midpoints -> heading-limited ordering -> smoothing spline ->
    wall-clearance check.
If PCD returns a long-enough path it is published (MODE: PCD in the plot).
Otherwise the ORIGINAL logic runs unchanged (NORMAL / FALLBACK modes,
big-orange candidate), e.g.
    - many orange / big-orange cones in the FOV (start line, skidpad lanes)
    - colour side not decided yet (first frames)
    - PCD path shorter than pcd.min_path_points / pcd.min_path_length_m

Colours: the controls_rl SLAM map publishes BLUE on the car's RIGHT.
pcd.color_mode = "auto" works the side out from the cones (latched);
set "blue_right" or "blue_left" to force it.

New parameters (all others unchanged)
    pcd.enable               True
    pcd.color_mode           "auto" | "blue_left" | "blue_right"
    pcd.track_width_m        3.0    (start value / fixed value)
    pcd.measure_width        True   (measure the width from the cones)
    pcd.use_memory           True
    pcd.use_prediction       True
    pcd.use_completion       True
    pcd.smoothing            0.4
    pcd.wall_margin_m        0.75
    pcd.min_path_points      5
    pcd.min_path_length_m    3.0
    pcd.cone_clearance_m     0.5    (cut path closer than this to a real cone
                                     or wall segment; 0 = off)
    pcd.defer_orange_count   4      (0 = never hand over for orange cones)

Debug topics (extra outputs, read by birdview.py; also work in RViz)
    /pp/mode             std_msgs/String  JSON {mode, reason, width, left}
    /pp/debug_markers    visualization_msgs/MarkerArray
    debug.publish        True   (False = don't publish them)

Set pcd.enable:=False to get exactly the old behaviour.
"""
from __future__ import annotations

# og code where only one virtal infront of car.
# challenges: overlap the boundary when too close.

# ============================================================
# pp_node_skidpad.py
#
# SLAM-based skidpad path planner with live visualization.
#
# NORMAL MODE:
#   If BLUE > 1 AND YELLOW > 1:
#       - Normal Delaunay on all FOV cones
#       - Original cone-type constraints
#       - Edge midpoints
#       - Big-orange special candidate
#       - Candidate filtering
#       - Greedy path
#
# FALLBACK MODE:
#   If BLUE <= 1 OR YELLOW <= 1:
#       - Ignore the side having <= 1 cone
#       - Select the side having more cones
#       - Require selected side to have >= 2 cones
#       - Add virtual colourless cone in front of car
#       - Delaunay(selected side + virtual cone)
#       - Use edge midpoints as candidates
#       - Candidate filtering
#       - Greedy path
#
# LIVE VISUALIZATION:
#   - Blue cones
#   - Yellow cones
#   - Orange cones
#   - Big-orange cones
#   - Virtual colourless cone
#   - FOV sector
#   - Delaunay edges
#   - Candidate points
#   - Final greedy path
#   - Car position + heading
#   - NORMAL / FALLBACK status
#
# ============================================================


import math
import threading
from typing import List, Tuple

import numpy as np

import rclpy
from rclpy.node import Node

from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
)

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from eufs_msgs.msg import ConeArrayWithCovariance


# ============================================================
# Matplotlib
# ============================================================

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge
except ImportError:
    plt = None


# ============================================================
# Optional SciPy Delaunay
# ============================================================

try:

    from scipy.spatial import Delaunay, QhullError

    _HAS_SCIPY = True

except ImportError:

    Delaunay = None
    _HAS_SCIPY = False



import json

from std_msgs.msg import String

try:
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point
    _HAS_VIZ_MSGS = True
except ImportError:                      # debug topics are then simply off
    _HAS_VIZ_MSGS = False

# ============================================================
# PCD PLANNER (embedded copy of corridor_delaunay.py v2,
# pure numpy/scipy, no ROS)
# ============================================================

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.spatial import Delaunay as _PCDDelaunay, cKDTree


@dataclass
class PCDConfig:
    # FOV (same sector model as the team's Delaunay node)
    fov_vertex: float = -5.0
    fov_radius: float = 30.0
    fov_angle_deg: float = 60.0
    # memory
    use_memory: bool = True
    merge_radius: float = 0.6
    window_ahead: float = 25.0
    window_behind: float = 4.0
    recency: float = 40.0           # [m] forget cones not seen during the last 40 m of driving
    # walls
    use_walls: bool = True          # False (ablation): no wall tracing; all cones go straight to Delaunay
    max_wall_edge: float = 6.5
    max_wall_turn_deg: float = 70.0
    fallback_max_edge: float = 5.5  # [m]   the non-Delaunay fallback must be shorter ...
    fallback_max_turn_deg: float = 40.0   # ... and straighter ...
    max_turn_change_deg: float = 45.0     # ... and turn about as much as the step before
    # corridor
    use_completion: bool = True     # False (ablation): no virtual cones
    tight_bend_factor: float = 0.9  # skip a virtual cone if |kappa| * w >= this (offset towards the centre)
    measure_width: bool = False     # False: use default_width as THE track width (fixed-width tracks)
    default_width: float = 3.0
    width_clip: Tuple[float, float] = (2.5, 6.0)
    # prediction
    use_prediction: bool = True
    predict_cones: int = 1
    predict_min_radius: float = 2.8     # v2: was 3.5; FS rules (9 m turning diameter) allow ~3 m centre radius
    predict_max_radius: float = 40.0
    predict_decay: float = 0.6
    predict_min_chain: int = 4
    check_predictions: bool = True
    predict_max_turn_deg: float = 60.0  # v2: was 45; hairpins turn 180 deg
    # edges / midpoints
    edge_min_frac: float = 0.55
    edge_max_frac: float = 1.75
    min_spacing: float = 1.0
    min_car_dist: float = 1.0
    max_hop: float = 7.0
    max_chain_turn_deg: float = 60.0
    order: str = "greedy"           # "greedy" (heading-limited best next point) or "graph"
                                    # (shortest-path search). The graph search was tested and was
                                    # NOT better on our tracks (more off-track frames), so greedy
                                    # stays the default; see pcd_v2_benchmark_output.txt
    turn_penalty: float = 1.0       # graph: edge cost = length * (1 + turn_penalty * turn) + trust penalty
    trust_penalty: float = 0.3      # graph: per metre, times (1 - weight of the midpoint reached)
    detour_weight: float = 1.0      # graph: end choice maximises length - detour_weight * (cost - length)
    # v2: unknown-colour cones (AMZ 2020: distant / uncertain cones as boundary candidates)
    use_unknown: bool = True
    unknown_fov_radius: float = 26.0
    unknown_fov_angle_deg: float = 170.0
    unknown_max_edge: float = 5.5       # same limits as the non-Delaunay wall step
    unknown_max_turn_deg: float = 40.0
    unknown_max_cones: int = 6          # per wall per frame
    unknown_dedup_radius: float = 1.0   # an unknown cone this close to a coloured one IS that cone
    # v2: path shaping
    tip_follow: bool = True             # v2: walls may follow a hairpin's inner tip (short, sharp steps)
    tip_max_edge: float = 2.0           # [m]
    tip_max_turn_deg: float = 110.0
    fit_through_car: bool = False       # False: spline fitted through the midpoints only (no hook to an off-line car)
    min_curvature: bool = True          # KA-RaceIng 2022: curvature-minimised line inside the corridor
    mc_trigger_kappa: float = 0.12      # only reshape when the path bends tighter than this [1/m]
    mc_margin: float = 1.0              # keep this far from both walls [m] (car half-width 0.8 + 0.2)
    mc_ridge: float = 0.02              # small pull towards the centre line
    # output
    smoothing: float = 0.4
    sample_step: float = 0.5
    safety_check: bool = True
    wall_margin: float = 0.75       # [m] path must stay this far from both walls (half a car)
    check_horizon: float = 15.0     # [m] how far along the path the check runs


REAL, VIRTUAL, PREDICTED, UNKNOWN = 0, 1, 2, 3
W_OF = {(REAL, REAL): 1.0, (REAL, VIRTUAL): 0.6, (VIRTUAL, REAL): 0.6,
        (REAL, UNKNOWN): 0.8, (UNKNOWN, REAL): 0.8, (UNKNOWN, UNKNOWN): 0.7,
        (UNKNOWN, VIRTUAL): 0.5, (VIRTUAL, UNKNOWN): 0.5}


def _rot90(v, left=True):
    return np.stack([-v[..., 1], v[..., 0]], -1) if left else np.stack([v[..., 1], -v[..., 0]], -1)


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def _ang(a, b):
    c = np.clip(np.dot(_unit(a), _unit(b)), -1, 1)
    return math.acos(float(c))


def _dist_to_polyline(pts, poly):
    """Distance of each point to a polyline; points whose closest spot is an END of the
    polyline (i.e. beyond its extent) get +inf so they don't bias the width."""
    out = np.full(len(pts), np.inf)
    for k, p in enumerate(pts):
        best = np.inf
        for a, b in zip(poly[:-1], poly[1:]):
            ab = b - a
            t = float(np.dot(p - a, ab) / max(ab @ ab, 1e-9))
            if t < 0 or t > 1:
                continue
            best = min(best, float(np.linalg.norm(a + t * ab - p)))
        out[k] = best
    return out


def _resample(poly, step):
    """Points every `step` metres along a polyline (keeps the first point)."""
    if len(poly) < 2:
        return poly.copy()
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))]
    if d[-1] < 1e-6:
        return poly[:1].copy()
    u = np.arange(0, d[-1] + 1e-9, step)
    return np.c_[np.interp(u, d, poly[:, 0]), np.interp(u, d, poly[:, 1])]


def _seg_dist(pts, poly):
    """Distance from each point to a polyline (segments clamped at the ends)."""
    if len(poly) == 0 or len(pts) == 0:
        return np.full(len(pts), np.inf)
    if len(poly) == 1:
        return np.linalg.norm(pts - poly[0], axis=1)
    a, b = poly[:-1], poly[1:]
    ab = b - a
    t = np.clip(((pts[:, None] - a[None]) * ab[None]).sum(2) / np.maximum((ab * ab).sum(1), 1e-9)[None], 0, 1)
    return np.linalg.norm(a[None] + t[..., None] * ab[None] - pts[:, None], axis=2).min(1)


class PCDPlanner:
    def __init__(self, cfg: Optional[PCDConfig] = None):
        self.cfg = cfg or PCDConfig()
        self.mem = {"blue": np.zeros((0, 2)), "yellow": np.zeros((0, 2)), "unknown": np.zeros((0, 2))}
        self.seen = {"blue": np.zeros(0), "yellow": np.zeros(0), "unknown": np.zeros(0)}   # odometer when last seen
        self.odo, self._last_car = 0.0, None
        self.width = self.cfg.default_width
        self.n_width = 0
        self.debug: Dict = {}

    # ------------------------------------------------------------------ FOV + memory
    def fov_mask(self, pts, car, cdir):
        c = self.cfg
        if len(pts) == 0:
            return np.zeros(0, bool)
        d = _unit(cdir)
        rel = pts - car
        fwd = rel @ d
        lat = rel @ _rot90(d)
        vx = fwd - c.fov_vertex
        r = np.hypot(vx, lat)
        return (r <= c.fov_radius) & (np.abs(np.arctan2(lat, vx)) <= math.radians(c.fov_angle_deg) / 2)

    def _merge(self, key, new):
        """Add this frame's cones of one colour to memory.
        - detections that match the same remembered cone are AVERAGED first, then that
          average nudges the remembered position (70 % old, 30 % new);
        - unmatched detections are added;
        - cones not seen during the last `recency` metres of driving are deleted."""
        c = self.cfg
        old, t = self.mem[key], self.seen[key]
        if len(new):
            if len(old) == 0:
                old, t = new.copy(), np.full(len(new), self.odo)
            else:
                d, i = cKDTree(old).query(new)
                hit = d < c.merge_radius
                for j in np.unique(i[hit]):
                    obs = new[hit & (i == j)].mean(axis=0)
                    old[j] = 0.7 * old[j] + 0.3 * obs
                    t[j] = self.odo
                old = np.vstack([old, new[~hit]])
                t = np.r_[t, np.full(int((~hit).sum()), self.odo)]
        keep = self.odo - t <= c.recency
        self.mem[key], self.seen[key] = old[keep], t[keep]

    def _window(self, pts, car, cdir):
        if len(pts) == 0:
            return pts
        fwd = (pts - car) @ _unit(cdir)
        r = np.linalg.norm(pts - car, axis=1)
        return pts[(fwd > -self.cfg.window_behind) & (r < self.cfg.window_ahead)]

    # ------------------------------------------------------------------ walls
    def _chain(self, pts, edges_same, car, cdir):
        """Order one colour into a wall: start near the car, walk along same-colour
        Delaunay edges (fallback: any same-colour cone in range) keeping the heading."""
        c = self.cfg
        if len(pts) == 0:
            return np.zeros((0, 2))
        fwd = (pts - car) @ _unit(cdir)
        dist = np.linalg.norm(pts - car, axis=1)
        # start AHEAD of the car and close to it (a wall that starts 10 m away is some
        # other part of the track, e.g. the far leg of a hairpin held in memory)
        cand = np.where((fwd > -0.5) & (dist < max(2.2 * self.width, 10.0)))[0]
        if len(cand) == 0:
            return np.zeros((0, 2))
        start = cand[np.argmin(np.linalg.norm(pts[cand] - car, axis=1))]
        nbr = {i: set() for i in range(len(pts))}
        for i, j in edges_same:
            nbr[i].add(j); nbr[j].add(i)
        order, used, heading = [start], {start}, _unit(cdir)
        prev_turn = None                                    # signed turn at the previous cone
        while True:
            first = len(order) == 1
            cur = order[-1]
            best, best_s, best_turn = None, np.inf, 0.0
            # pool 1: same-colour Delaunay neighbours; pool 2 (fallback): any cone, but
            # shorter and straighter, so the wall cannot jump to another part of the track
            for pool, max_L, lim_deg in ((nbr[cur], c.max_wall_edge, 100.0 if first else c.max_wall_turn_deg),
                                         (range(len(pts)), c.fallback_max_edge, 100.0 if first else c.fallback_max_turn_deg)):
                lim = math.radians(lim_deg)
                for j in pool:
                    if j in used:
                        continue
                    v = pts[j] - pts[cur]
                    L = np.linalg.norm(v)
                    if L < 0.3 or L > max_L:
                        continue
                    t = _ang(v, heading)
                    if t > lim:
                        continue
                    st = math.copysign(t, heading[0] * v[1] - heading[1] * v[0])
                    # walls bend smoothly: the turn may not change much from one cone to the next
                    if prev_turn is not None and abs(st - prev_turn) > math.radians(c.max_turn_change_deg):
                        continue
                    s = L * (1 + 1.5 * t)
                    if s < best_s:
                        best, best_s, best_turn = j, s, st
                if best is not None:
                    break
            if best is None and c.tip_follow and not first and prev_turn is not None:
                # v2 pool 3: the inner tip of a hairpin (comp_2021: R ~1.3 m, cones ~1 m apart)
                # turns more than the limits above allow. A SHORT step (<= tip_max_edge) may turn
                # up to tip_max_turn_deg in the SAME direction the wall is already turning, as long
                # as the cone does not lie back near the earlier part of the wall.
                lim = math.radians(c.tip_max_turn_deg)
                for j in range(len(pts)):
                    if j in used:
                        continue
                    v = pts[j] - pts[cur]
                    L = np.linalg.norm(v)
                    if L < 0.3 or L > c.tip_max_edge:
                        continue
                    t = _ang(v, heading)
                    st = math.copysign(t, heading[0] * v[1] - heading[1] * v[0])
                    if t > lim or st * prev_turn < 0 or abs(prev_turn) < math.radians(8):
                        continue
                    if len(order) >= 3 and np.min(np.linalg.norm(pts[order[:-1]] - pts[j], axis=1)) < 0.8 * L:
                        continue
                    sc = L * (1 + 0.5 * t)
                    if sc < best_s:
                        best, best_s, best_turn = j, sc, st
            if best is None:
                break
            if not first:
                prev_turn = best_turn
            heading = _unit(pts[best] - pts[cur])
            order.append(best); used.add(best)
        return pts[order]

    @staticmethod
    def _normals(chain, cdir, interior_left: bool):
        if len(chain) == 1:
            t = _unit(cdir)[None]
        else:
            t = np.gradient(chain, axis=0)
            t = _unit(t)
        return _rot90(t, left=interior_left)

    # ------------------------------------------------------------------ main
    def plan(self, blue, yellow, car, cdir, already_fov_filtered=False, unknown=None):
        c = self.cfg
        blue = np.asarray(blue, float).reshape(-1, 2)
        yellow = np.asarray(yellow, float).reshape(-1, 2)
        unknown = np.zeros((0, 2)) if unknown is None else np.asarray(unknown, float).reshape(-1, 2)
        car = np.asarray(car, float); cdir = _unit(np.asarray(cdir, float))
        if not already_fov_filtered:
            blue = blue[self.fov_mask(blue, car, cdir)]
            yellow = yellow[self.fov_mask(yellow, car, cdir)]
        if len(unknown):                                   # wider, longer sector: that is where they are
            rel = unknown - car
            fwd, lat = rel @ cdir, rel @ _rot90(cdir)
            ok = (np.hypot(fwd, lat) <= c.unknown_fov_radius) & \
                 (np.abs(np.degrees(np.arctan2(lat, fwd))) <= c.unknown_fov_angle_deg / 2)
            unknown = unknown[ok]

        # 1. memory
        if self._last_car is not None:
            self.odo += float(np.linalg.norm(car - self._last_car))
        self._last_car = car.copy()
        if c.use_memory:
            self._merge("blue", blue)
            self._merge("yellow", yellow)
            B = self._window(self.mem["blue"], car, cdir)
            Y = self._window(self.mem["yellow"], car, cdir)
        else:
            B, Y = blue, yellow
        U = np.zeros((0, 2))
        if c.use_unknown:
            if c.use_memory:
                self._merge("unknown", unknown)
                U = self._window(self.mem["unknown"], car, cdir)
            else:
                U = unknown
            col_all = np.vstack([self.mem["blue"], self.mem["yellow"], blue, yellow])
            if len(U) and len(col_all):                    # a cone that has its colour now is not unknown
                U = U[cKDTree(col_all).query(U)[0] > c.unknown_dedup_radius]
        self._car, self._cdir, self._B, self._Y = car, cdir, B, Y   # for subclasses / hooks
        self._on_frame(car, cdir, B, Y)

        # 2. walls from same-colour Delaunay edges
        P = np.vstack([B, Y]) if len(B) + len(Y) else np.zeros((0, 2))
        col = np.r_[np.zeros(len(B), int), np.ones(len(Y), int)]
        e_bb, e_yy = [], []
        if len(P) >= 3:
            try:
                for tri in _PCDDelaunay(P).simplices:
                    for i, j in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                        if col[i] == col[j] == 0:
                            e_bb.append((i, j))
                        elif col[i] == col[j] == 1:
                            e_yy.append((i - len(B), j - len(B)))
            except Exception:
                pass
        if c.use_walls:
            Lw = self._chain(B, e_bb, car, cdir)      # left wall  (blue)
            Rw = self._chain(Y, e_yy, car, cdir)      # right wall (yellow)
        else:                                         # ablation: unordered cones, no walls
            Lw, Rw = B, Y

        # 3a. measure width from real cones: distance from each cone to the OTHER
        #     wall's polyline (point-to-segment, correct on curves), ahead of the car,
        #     bootstrapped from the first frames, then gated so one bad frame can't drag it away
        if c.measure_width and len(Lw) >= 2 and len(Rw) >= 2:
            d = np.r_[_dist_to_polyline(Lw, Rw), _dist_to_polyline(Rw, Lw)]
            d = d[(d > c.width_clip[0] * 0.8) & (d < c.width_clip[1])]
            if len(d) >= 3:
                m = float(np.median(d))
                if self.n_width < 3:                       # bootstrap: trust the first measurements
                    self.width = m if self.n_width == 0 else 0.5 * self.width + 0.5 * m
                elif abs(m - self.width) < 0.4 * self.width:   # afterwards: slow, outlier-gated
                    self.width = 0.8 * self.width + 0.2 * m
                self.width = float(np.clip(self.width, *c.width_clip))
                self.n_width += 1
        w = self.width

        # 3c. v2: extend each wall with unknown-colour cones, constrained by the corridor
        #     (only when BOTH walls are traced, a cone must continue its wall's chain under the
        #     non-Delaunay step limits and keep 0.6 w .. 1.6 w from the other wall)
        Lu, Ru = np.zeros((0, 2)), np.zeros((0, 2))
        if c.use_unknown and c.use_walls and len(U):
            Lu, Ru = self._extend_with_unknown(Lw, Rw, U, w)
        LwX = np.vstack([Lw, Lu]) if len(Lu) else Lw      # extended chains (for prediction/completion)
        RwX = np.vstack([Rw, Ru]) if len(Ru) else Rw

        # 4. curvature prediction (extend each wall into the blind part of the bend)
        Lp, Rp = np.zeros((0, 2)), np.zeros((0, 2))
        if c.use_prediction and c.use_walls:
            self._pred_kappa = {}
            self._last_pred_kappa = None
            Lp = self._predict(LwX); self._pred_kappa["L"] = self._last_pred_kappa
            self._last_pred_kappa = None
            Rp = self._predict(RwX); self._pred_kappa["R"] = self._last_pred_kappa
            Lp, Rp = self._check_predictions(LwX, Lp, RwX, Rp, w)

        # 3b. corridor completion (virtual partner where the other wall has none)
        def complete(chain, other, interior_left):
            if len(chain) == 0 or not (c.use_completion and c.use_walls):
                return np.zeros((0, 2))
            n = self._normals(chain, cdir, interior_left)
            v = chain + n * w
            ok = np.ones(len(chain), bool)
            if len(other):
                ok &= cKDTree(other).query(chain)[0] > 1.35 * w
            # tight-bend guard: an offset of w TOWARDS the centre of a bend only exists if the
            # bend radius is larger than w; skip virtual cones where |kappa| * w is too big
            kap = self._vertex_curvature(chain)
            left_n = _rot90(_unit(np.gradient(chain, axis=0)) if len(chain) > 1 else _unit(cdir)[None], left=True)
            towards = (n * left_n).sum(1) * kap > 0
            ok &= ~(towards & (np.abs(kap) * w >= c.tight_bend_factor))
            return v[ok]

        Lfull = np.vstack([LwX, Lp]) if len(Lp) else LwX
        Rfull = np.vstack([RwX, Rp]) if len(Rp) else RwX
        vY = self._filter_virtual(complete(Lfull, Rfull, interior_left=False))   # left wall -> interior is to its right
        vB = self._filter_virtual(complete(Rfull, Lfull, interior_left=True))

        # 5. re-triangulate everything, keep width-consistent cross edges
        pts = [Lw, Lu, Lp, vB, Rw, Ru, Rp, vY]
        kinds = [REAL, UNKNOWN, PREDICTED, VIRTUAL, REAL, UNKNOWN, PREDICTED, VIRTUAL]
        cols = [0, 0, 0, 0, 1, 1, 1, 1]
        Q = np.vstack([p for p in pts if len(p)]) if any(len(p) for p in pts) else np.zeros((0, 2))
        qk = np.concatenate([np.full(len(p), k) for p, k in zip(pts, kinds)]) if len(Q) else np.zeros(0)
        qc = np.concatenate([np.full(len(p), k) for p, k in zip(pts, cols)]) if len(Q) else np.zeros(0)

        mids, wts = [], []
        if len(Q) >= 3:
            try:
                tris = _PCDDelaunay(Q).simplices
            except Exception:
                tris = []
            seen = set()
            for tri in tris:
                for i, j in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                    key = (min(i, j), max(i, j))
                    if key in seen or qc[i] == qc[j]:
                        continue
                    seen.add(key)
                    L = np.linalg.norm(Q[i] - Q[j])
                    if not (c.edge_min_frac * w <= L <= c.edge_max_frac * w):
                        continue
                    kk = (int(qk[i]), int(qk[j]))
                    wt = W_OF.get(kk, 0.3)
                    mids.append((Q[i] + Q[j]) / 2); wts.append(wt)
        mids = np.asarray(mids).reshape(-1, 2); wts = np.asarray(wts)

        path, info = self._order_and_fit(mids, wts, car, cdir)
        info["min_curvature"] = "off"
        base = path
        if c.min_curvature and len(path) >= 6:
            path, info["min_curvature"] = self._min_curvature(path, Lfull, Rfull, vB, vY)
        if c.safety_check and len(path) >= 2:
            p_mc, s_mc = self._safety(path, info.get("chain"), Lfull, Rfull, vB, vY)
            path, info["safety"] = p_mc, s_mc
            if s_mc != "ok" and path is not base and len(base) >= 2:
                # the curvature-minimised line failed the wall check: keep the original fit
                p0, s0 = self._safety(base, info.get("chain"), Lfull, Rfull, vB, vY)
                if s0 == "ok" or len(p0) > len(p_mc):
                    path, info["safety"] = p0, s0
                    info["min_curvature"] += " -> reverted (wall check)"
        # confirmed distance: arc length up to the farthest midpoint built only from perceived
        # cones (real or unknown-colour, weight >= 0.7) -- beyond it the path rests on guesses
        conf = 0.0
        if len(path) >= 2 and len(mids):
            arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
            good = mids[wts >= 0.7]
            if len(good):
                d, j = cKDTree(path).query(good)
                ok = d < 1.0
                if ok.any():
                    conf = float(arc[j[ok]].max())
        info["confirmed_m"] = conf
        self.debug = dict(left=LwX, right=RwX, left_pred=Lp, right_pred=Rp,
                          left_unknown=Lu, right_unknown=Ru,
                          virt_blue=vB, virt_yellow=vY, mids=mids, weights=wts,
                          width=w, mem_blue=B, mem_yellow=Y, mem_unknown=U, **info)
        return path

    # ------------------------------------------------------------------ v2 helpers
    def _extend_with_unknown(self, Lw, Rw, U, w):
        """Continue each traced wall with unknown-colour cones (AMZ 2020 idea, constrained by the
        corridor as suggested in the PP review): both walls must already exist; a cone must follow
        its wall like a non-Delaunay wall step (<= unknown_max_edge, turn <= unknown_max_turn_deg,
        turn change <= max_turn_change_deg) and lie 0.6 w .. 1.6 w from the other (extended) wall.
        Each cone is used once; the cheapest step over both walls is taken first."""
        c = self.cfg
        if len(Lw) < 2 or len(Rw) < 2 or len(U) == 0:
            return np.zeros((0, 2)), np.zeros((0, 2))
        walls = [list(Lw), list(Rw)]
        added = [[], []]
        used = np.zeros(len(U), bool)
        prev_turn = [None, None]
        for k in range(2):
            ch = walls[k]
            if len(ch) >= 3:
                a, b = _unit(ch[-2] - ch[-3]), _unit(ch[-1] - ch[-2])
                prev_turn[k] = math.copysign(_ang(b, a), a[0] * b[1] - a[1] * b[0])
        lim = math.radians(c.unknown_max_turn_deg)
        dlim = math.radians(c.max_turn_change_deg)
        while True:
            best = None
            for k in range(2):
                if len(added[k]) >= c.unknown_max_cones:
                    continue
                ch, other = walls[k], np.asarray(walls[1 - k])
                end, head = ch[-1], _unit(ch[-1] - ch[-2])
                for j in np.where(~used)[0]:
                    v = U[j] - end
                    L = float(np.linalg.norm(v))
                    if L < 0.5 or L > c.unknown_max_edge:
                        continue
                    t = _ang(v, head)
                    if t > lim:
                        continue
                    st = math.copysign(t, head[0] * v[1] - head[1] * v[0])
                    if prev_turn[k] is not None and abs(st - prev_turn[k]) > dlim:
                        continue
                    d_other = float(_dist_to_polyline(U[j][None], other)[0]) if len(other) >= 2 \
                        else float(np.linalg.norm(other - U[j], axis=1).min())
                    if not (0.6 * w <= d_other <= 1.6 * w):
                        continue
                    score = L * (1 + 1.5 * t)
                    if best is None or score < best[0]:
                        best = (score, k, j, st)
            if best is None:
                break
            _, k, j, st = best
            used[j] = True
            walls[k].append(U[j]); added[k].append(U[j])
            prev_turn[k] = st
        return np.asarray(added[0]).reshape(-1, 2), np.asarray(added[1]).reshape(-1, 2)

    def _min_curvature(self, path, Lfull, Rfull, vB, vY):
        """Curvature-minimised line inside the corridor (after KA-RaceIng 2022, who run a
        minimum-curvature optimisation online on the unclosed track). Lateral offsets a_i along
        the left normals minimise the squared second differences of the path (~ curvature),
        plus a small pull to the centre line, within [-right clearance, +left clearance] minus
        mc_margin. The first point (the car) stays fixed. Bounded linear least squares."""
        from scipy.optimize import lsq_linear
        c = self.cfg
        P = np.asarray(path, float)
        n = len(P)
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        h = np.unwrap(np.arctan2(np.diff(P[:, 1]), np.diff(P[:, 0])))
        kap = np.abs(np.diff(h)) / np.maximum(0.5 * (seg[:-1] + seg[1:]), 1e-3)
        if kap.max() < c.mc_trigger_kappa:
            return path, "not needed"
        t = _unit(np.gradient(P, axis=0))
        N = _rot90(t, left=True)
        def side_clear(poly, pts_extra):
            d = _seg_dist(P, poly) if len(poly) >= 2 else np.full(n, np.inf)
            if len(pts_extra):
                d = np.minimum(d, cKDTree(pts_extra).query(P)[0])
            return d
        dl, dr = side_clear(Lfull, vB), side_clear(Rfull, vY)
        if not (np.isfinite(dl).all() and np.isfinite(dr).all()):
            return path, "no corridor"
        ub = np.clip(dl - c.mc_margin, 0.0, 2.0)
        lb = -np.clip(dr - c.mc_margin, 0.0, 2.0)
        ub[0] = lb[0] = 0.0
        ub[0] = 1e-6                                       # lsq_linear needs lb < ub
        rows, rhs = [], []
        for i in range(1, n - 1):                          # D2 (P + N a) ~ 0
            r = np.zeros((2, n))
            r[:, i - 1] += N[i - 1]; r[:, i] -= 2 * N[i]; r[:, i + 1] += N[i + 1]
            rows.append(r); rhs.append(-(P[i - 1] - 2 * P[i] + P[i + 1]))
        A = np.vstack(rows + [math.sqrt(c.mc_ridge) * np.eye(n)])
        b = np.r_[np.concatenate(rhs), np.zeros(n)]
        try:
            res = lsq_linear(A, b, bounds=(lb, ub), method="bvls", max_iter=200)
        except Exception:
            return path, "solver error"
        if not res.success:
            return path, "solver failed"
        out = P + N * res.x[:, None]
        out[0] = P[0]
        return out, f"applied (max shift {np.abs(res.x).max():.2f} m)"

    # hooks (no-ops here; VisibilityPCD overrides them)
    def _on_frame(self, car, cdir, B, Y):
        pass

    def _filter_virtual(self, v):
        return v

    def _check_predictions(self, Lw, Lp, Rw, Rp, w):
        """The two walls of a track are parallel. If both walls were guessed and they bend in
        OPPOSITE directions, neither guess is used. A guess closer than 0.6 w to the other
        wall would cross the track: it is cut off, together with the guesses after it."""
        if not self.cfg.check_predictions:
            return Lp, Rp
        kL, kR = self._pred_kappa.get("L"), self._pred_kappa.get("R")
        if len(Lp) and len(Rp) and kL is not None and kR is not None and kL * kR < 0 \
                and min(abs(kL), abs(kR)) > 1 / 40:
            return np.zeros((0, 2)), np.zeros((0, 2))
        def keep(pred, other):
            if len(pred) == 0 or len(other) == 0:
                return pred
            d = cKDTree(other).query(pred)[0]
            return pred[np.cumprod(d > 0.6 * w).astype(bool)]
        oL = np.vstack([Rw, Rp]) if len(Rp) else Rw
        oR = np.vstack([Lw, Lp]) if len(Lp) else Lw
        return keep(Lp, oL), keep(Rp, oR)

    @staticmethod
    def _vertex_curvature(chain):
        """Signed curvature (+ = turning left) at every cone of an ordered wall."""
        n = len(chain)
        k = np.zeros(n)
        if n < 3:
            return k
        seg = np.diff(chain, axis=0); L = np.linalg.norm(seg, axis=1)
        h = np.arctan2(seg[:, 1], seg[:, 0])
        k[1:-1] = ((np.diff(h) + np.pi) % (2 * np.pi) - np.pi) / np.maximum(0.5 * (L[:-1] + L[1:]), 1e-6)
        k[0], k[-1] = k[1], k[-2]
        return k

    def _safety(self, path, chain, Lfull, Rfull, vB, vY):
        """Every path point (first `check_horizon` m) must be >= wall_margin from both walls.
        Walls = the traced wall polylines (incl. guesses) and the virtual cones.
        1st choice: the smoothed path; 2nd: the unsmoothed chain; else cut before the problem."""
        c = self.cfg
        def clearance(p):
            dl = np.minimum(_seg_dist(p, Lfull), _seg_dist(p, vB) if len(vB) else np.inf)
            dr = np.minimum(_seg_dist(p, Rfull), _seg_dist(p, vY) if len(vY) else np.inf)
            return np.minimum(dl, dr)
        def first_bad(p):
            a = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
            cl = clearance(p)
            # v2: a car that is already close to a wall must be allowed to steer away from it:
            # the required margin ramps from the car's own clearance up to wall_margin (0.5 m per m)
            need = np.minimum(c.wall_margin, max(cl[0] - 0.05, 0.0) + 0.5 * a)
            bad = np.where((cl < need) & (a <= c.check_horizon) & (a > 0.5))[0]
            return int(bad[0]) if len(bad) else None
        b = first_bad(path)
        if b is None:
            return path, "ok"
        if chain is not None and len(chain) >= 2:
            raw = _resample(chain, c.sample_step)
            if first_bad(raw) is None:
                return raw, "unsmoothed"
        return path[:max(b, 2)], "cut"

    def _predict(self, chain):
        """Clothoid continuation of a wall into the part of the bend the camera can't see.

        Discrete curvature at the last vertices of the wall:  kappa_i = turn angle / step.
        A clothoid is a curve whose curvature changes LINEARLY with distance (it is what
        roads and FS tracks are made of between straights and arcs), so the next
        curvature is extrapolated linearly from the last two:
            kappa_next = kappa_n + k * (kappa_n - kappa_{n-1})
        -> a bend that is tightening keeps tightening (entry), a bend that is opening
           relaxes toward straight (exit), and a sign flip (S-bend) is clamped to straight.
        With only one curvature sample, it decays geometrically instead.
        """
        c = self.cfg
        if len(chain) < max(3, c.predict_min_chain):
            return np.zeros((0, 2))
        seg = np.diff(chain, axis=0)
        L = np.linalg.norm(seg, axis=1)
        h = np.arctan2(seg[:, 1], seg[:, 0])
        turn = (np.diff(h) + np.pi) % (2 * np.pi) - np.pi
        kap = turn / np.maximum(0.5 * (L[:-1] + L[1:]), 1e-6)
        k_n = float(kap[-1])
        if abs(k_n) < 1.0 / c.predict_max_radius:
            return np.zeros((0, 2))                       # straight: nothing to predict
        # curvature change PER METRE: kap[-1] sits at chain[-2], kap[-2] at chain[-3],
        # which are L[-2] metres apart (uneven cone spacing no longer distorts the trend)
        dkds = float((kap[-1] - kap[-2]) / max(L[-2], 1e-6)) if len(kap) >= 2 else None
        step = float(np.median(L))
        k_cap = 1.0 / c.predict_min_radius
        if abs(k_n) > 1.25 * k_cap:
            # v2: this wall already turns tighter than predict_min_radius (the inner tip of a
            # hairpin, R ~1.3 m on comp_2021). A capped, wider arc would bulge into the lane
            # (seen at the comp_2021 left hairpin), so this wall is not continued; the other
            # wall's prediction and the virtual cones still describe the corner.
            self._last_pred_kappa = None
            return np.zeros((0, 2))
        cap = math.radians(c.predict_max_turn_deg)
        head0 = head = float(h[-1]); p, out = chain[-1].copy(), []
        s_from = float(L[-1])                             # arc length from chain[-2] to the last cone
        kk = 0.0
        for k in range(1, c.predict_cones + 1):
            if dkds is not None:
                # curvature trend continued to the START of this step (the last known cone for
                # the first guess). Continuing it to the middle of the step is the exact clothoid,
                # but extrapolates further and was less safe in testing (5 vs 1 off-track frames).
                kk = k_n + dkds * (s_from + (k - 1) * step)
                if kk * k_n <= 0:                         # would flip direction -> straight
                    kk = 0.0
                kk = float(np.clip(kk, -k_cap, k_cap))
                if abs(kk) > abs(k_n) * 1.5:
                    kk = math.copysign(abs(k_n) * 1.5, k_n)
            else:
                kk = k_n * (c.predict_decay ** k)
            if abs(head + kk * step - head0) > cap:
                break                                     # don't guess more turning than this
            mid = head + 0.5 * kk * step                  # midpoint rule: heading halfway along the step
            p = p + step * np.array([math.cos(mid), math.sin(mid)])
            head += kk * step
            out.append(p.copy())
        self._last_pred_kappa = kk if out else None
        return np.asarray(out).reshape(-1, 2)

    def _order_and_fit(self, mids, wts, car, cdir):
        """Midpoints -> ordered chain -> smoothed path. Returns (path, info).
        The path ALWAYS starts at the car position (or is empty)."""
        c = self.cfg
        empty = (np.zeros((0, 2)), dict(chain=None, n_chain=0))
        if len(mids) == 0:
            return empty
        # de-duplicate, keeping the more trustworthy midpoint
        order = np.argsort(-wts, kind="stable")
        K, KW = [], []
        for i in order:
            p = mids[i]
            if np.linalg.norm(p - car) < c.min_car_dist:
                continue
            if any(np.linalg.norm(p - q) < c.min_spacing for q in K):
                continue
            if (p - car) @ cdir < -1.0:
                continue
            K.append(p); KW.append(wts[i])
        if not K:
            return empty
        K, KW = np.asarray(K), np.asarray(KW)
        idx = self._order_graph(K, KW, car, cdir) if c.order == "graph" else self._order_greedy(K, car, cdir)
        if not idx:
            return empty
        seq = np.vstack([car, K[idx]]); sw = np.r_[1.0, KW[idx]]
        info = dict(chain=seq, n_chain=len(idx))
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(seq, axis=0), axis=1))]
        keep = np.r_[True, np.diff(d) > 1e-3]
        seq, d, sw = seq[keep], d[keep], sw[keep]
        if len(seq) < 3:                                   # too short to smooth: straight pieces
            return _resample(seq, c.sample_step), info
        k = min(3, len(seq) - 1)
        try:
            if not c.fit_through_car and len(seq) >= 4:
                # v2: fit the midpoints only; the car is joined by a straight piece, so an off-line
                # car does not bend the whole path into a hook (PP-07)
                q, dq, wq = seq[1:], d[1:] - d[1], sw[1:]
                kq = min(3, len(q) - 1)
                tck, _ = splprep(q.T, u=dq, w=wq * 2.0, k=kq, s=c.smoothing * len(q))
                uu = np.arange(0, dq[-1] + 1e-9, c.sample_step)
                body = np.stack(splev(uu, tck), 1)
                lead = _resample(np.vstack([car, body[0]]), c.sample_step)
                path = np.vstack([lead[:-1], body]) if len(lead) > 1 else np.vstack([car, body])
                path[0] = car
                return path, info
            tck, _ = splprep(seq.T, u=d, w=sw * 2.0, k=k, s=c.smoothing * len(seq))
            uu = np.arange(0, d[-1] + 1e-9, c.sample_step)
            path = np.stack(splev(uu, tck), 1)
            path[0] = car                                  # contract: first point = car
            return path, info
        except Exception:
            return _resample(seq, c.sample_step), info

    def _order_greedy(self, K, car, cdir):
        """v1 ordering: from the car, repeatedly take the best next point (heading-limited)."""
        c = self.cfg
        used = np.zeros(len(K), bool); cur, head, idx = car.copy(), cdir.copy(), []
        lim = math.radians(c.max_chain_turn_deg)
        while True:
            v = K - cur; d = np.linalg.norm(v, axis=1)
            ok = (~used) & (d <= c.max_hop) & (d > 1e-6)
            if not ok.any():
                break
            ang = np.array([_ang(v[i], head) if ok[i] else np.inf for i in range(len(K))])
            ok &= ang <= lim
            if not ok.any():
                break
            score = np.full(len(K), np.inf); score[ok] = d[ok] * (1 + ang[ok])
            i = int(np.argmin(score)); used[i] = True
            head = _unit(K[i] - cur); cur = K[i]; idx.append(i)
        return idx

    def _order_graph(self, K, KW, car, cdir):
        """Shortest-path ordering over ALL midpoints instead of a greedy walk.

        Nodes: the car (index -1) and the midpoints. A step a -> b is allowed if it is at most
        `max_hop` long and turns at most `max_chain_turn_deg` from the step before (from the
        car: from its heading). Because the allowed next steps depend on the previous one, the
        search runs over STEPS (pairs a -> b), with Dijkstra.
          cost of a step = length * (1 + turn_penalty * turn [rad])
                         + trust_penalty * length * (1 - weight of b)
        Every reachable step ends a candidate path; the one chosen maximises
          length - detour_weight * (cost - length)
        i.e. it goes as far as possible while turning little and using trusted midpoints."""
        import heapq
        c = self.cfg
        n = len(K)
        lim = math.radians(c.max_chain_turn_deg)
        D = np.linalg.norm(K[:, None] - K[None], axis=2)
        nb = [np.where((D[i] <= c.max_hop) & (D[i] > 1e-6))[0] for i in range(n)]
        best = {}                                          # (a, b) -> (cost, length, parent)
        heap = []
        for b in range(n):
            v = K[b] - car; L = float(np.linalg.norm(v))
            if L > c.max_hop or L < 1e-6:
                continue
            t = _ang(v, cdir)
            if t > lim:
                continue
            cost = L * (1 + c.turn_penalty * t) + c.trust_penalty * L * (1 - KW[b])
            best[(-1, b)] = (cost, L, None)
            heapq.heappush(heap, (cost, -1, b))
        done = set()
        while heap:
            cost, a, b = heapq.heappop(heap)
            if (a, b) in done or cost > best[(a, b)][0] + 1e-12:
                continue
            done.add((a, b))
            L0 = best[(a, b)][1]
            pa = car if a == -1 else K[a]
            hd = _unit(K[b] - pa)
            for k in nb[b]:
                if k == a:
                    continue
                v = K[k] - K[b]; L = D[b, k]
                t = _ang(v, hd)
                if t > lim:
                    continue
                nc = cost + L * (1 + c.turn_penalty * t) + c.trust_penalty * L * (1 - KW[k])
                if nc < best.get((b, k), (np.inf,))[0]:
                    best[(b, k)] = (nc, L0 + L, (a, b))
                    heapq.heappush(heap, (nc, b, k))
        if not best:
            return []
        end = max(best, key=lambda e: best[e][1] - c.detour_weight * (best[e][0] - best[e][1]))
        idx, e, seen = [], end, set()
        while e is not None and e not in seen:            # walk back (guard against loops)
            seen.add(e); idx.append(e[1]); e = best[e][2]
        idx = idx[::-1]
        out, used = [], set()                             # a midpoint is used once
        for i in idx:
            if i in used:
                break
            used.add(i); out.append(i)
        return out


# ============================================================
# END OF PCD PLANNER - original node code below
# ============================================================

# ============================================================
# Quaternion -> yaw
# ============================================================

def yaw_from_quat(
    qx,
    qy,
    qz,
    qw
) -> float:

    siny_cosp = (
        2.0 *
        (
            qw * qz
            +
            qx * qy
        )
    )

    cosy_cosp = (
        1.0
        -
        2.0 *
        (
            qy * qy
            +
            qz * qz
        )
    )

    return math.atan2(
        siny_cosp,
        cosy_cosp
    )


# ============================================================
# MAIN NODE
# ============================================================

class SlamPathSectorVisualizer(Node):

    def __init__(self):

        super().__init__(
            "slam_path_sector_visualiser"
        )

        # ====================================================
        # PARAMETERS
        # ====================================================

        self.declare_parameter(
            "topics.odom_in",
            "/slam/odom"
        )

        self.declare_parameter(
            "topics.map_in",
            "/slam/map_cones"
        )

        self.declare_parameter(
            "topics.path_out",
            "/path_points"
        )

        # ----------------------------------------------------
        # QoS
        # ----------------------------------------------------

        self.declare_parameter(
            "qos.best_effort",
            True
        )

        self.declare_parameter(
            "qos.depth",
            50
        )

        # ----------------------------------------------------
        # FOV
        # ----------------------------------------------------

        self.declare_parameter(
            "fov.vertex_offset_m",
            -5.0
        )

        self.declare_parameter(
            "fov.radius_m",
            30.0
        )

        self.declare_parameter(
            "fov.angle_deg",
            60.0
        )

        # ----------------------------------------------------
        # Candidate filtering
        # ----------------------------------------------------

        self.declare_parameter(
            "candidates.min_spacing_m",
            1.0
        )

        self.declare_parameter(
            "candidates.min_car_dist_m",
            2.0
        )

        # ----------------------------------------------------
        # Greedy path
        # ----------------------------------------------------

        self.declare_parameter(
            "greedy.max_hop_m",
            7.0
        )

        # ----------------------------------------------------
        # Publish
        # ----------------------------------------------------

        self.declare_parameter(
            "publish.rate_hz",
            20.0
        )

        # ----------------------------------------------------
        # Virtual colourless cone
        # ----------------------------------------------------

        self.declare_parameter(
            "fallback.virtual_cone_distance_m",
            1.0
        )

        # ----------------------------------------------------
        # PCD (Predictive Corridor Delaunay) - NEW
        # ----------------------------------------------------

        self.declare_parameter("pcd.enable", True)
        # "auto": find which colour is on the car's left from the cones
        #         (the controls_rl SLAM map has BLUE on the RIGHT)
        # "blue_left" / "blue_right": force it
        self.declare_parameter("pcd.color_mode", "auto")
        self.declare_parameter("pcd.track_width_m", 3.0)
        self.declare_parameter("pcd.measure_width", True)
        self.declare_parameter("pcd.use_memory", True)
        self.declare_parameter("pcd.use_prediction", True)
        self.declare_parameter("pcd.use_completion", True)
        self.declare_parameter("pcd.smoothing", 0.4)
        self.declare_parameter("pcd.wall_margin_m", 0.75)
        self.declare_parameter("pcd.min_path_points", 5)
        self.declare_parameter("pcd.min_path_length_m", 3.0)
        # cut the PCD path where it gets closer than this to a real
        # cone / wall segment of the map (last safety net, 0 = off)
        self.declare_parameter("pcd.cone_clearance_m", 0.5)
        # hand over to the original NORMAL/FALLBACK logic when this many
        # orange + big-orange cones are in the FOV (start line, skidpad lanes)
        # 0 = never hand over because of orange cones
        self.declare_parameter("pcd.defer_orange_count", 4)
        # v2: "validate" = ask PCD first even with orange cones in view and hand over only when
        #     PCD has no valid path or the scene is orange-only; "defer" = v1 behaviour
        self.declare_parameter("pcd.orange_policy", "validate")
        self.declare_parameter("pcd.use_unknown", True)          # v2: unknown-colour cones
        self.declare_parameter("pcd.min_curvature", True)        # v2: curvature-limited line
        self.declare_parameter("pcd.fit_through_car", False)     # v2: no spline hook to the car
        # v2 route memory (applies to every published path, PCD or team logic)
        self.declare_parameter("route.enable", True)
        self.declare_parameter("route.max_jump_m", 1.0)          # 4 m look-ahead point may move this much per tick
        self.declare_parameter("route.hold_ticks", 10)           # keep the previous route at most this many ticks (0.5 s at 20 Hz)
        self.declare_parameter("route.switch_ticks", 3)          # a different route must persist this many ticks
        self.declare_parameter("route.reset_jump_m", 5.0)        # car moved more than this in one tick -> reset (sim reset)
        # v2 gate pairing for the team NORMAL logic (big orange cones)
        self.declare_parameter("gate.min_width_m", 2.5)
        self.declare_parameter("gate.max_width_m", 6.5)

        # ====================================================
        # READ PARAMETERS
        # ====================================================

        gp = self.get_parameter

        odom_topic = str(
            gp(
                "topics.odom_in"
            ).value
        )

        map_topic = str(
            gp(
                "topics.map_in"
            ).value
        )

        path_topic = str(
            gp(
                "topics.path_out"
            ).value
        )

        best_effort = bool(
            gp(
                "qos.best_effort"
            ).value
        )

        depth = int(
            gp(
                "qos.depth"
            ).value
        )

        self.vertex_offset = float(
            gp(
                "fov.vertex_offset_m"
            ).value
        )

        self.fov_radius = float(
            gp(
                "fov.radius_m"
            ).value
        )

        self.fov_angle_deg = float(
            gp(
                "fov.angle_deg"
            ).value
        )

        self.fov_half_rad = math.radians(
            self.fov_angle_deg * 0.5
        )

        self.min_spacing = float(
            gp(
                "candidates.min_spacing_m"
            ).value
        )

        self.min_car_dist = float(
            gp(
                "candidates.min_car_dist_m"
            ).value
        )

        self.max_hop = float(
            gp(
                "greedy.max_hop_m"
            ).value
        )

        self.publish_rate_hz = float(
            gp(
                "publish.rate_hz"
            ).value
        )

        self.virtual_cone_distance = float(
            gp(
                "fallback.virtual_cone_distance_m"
            ).value
        )

        # ----------------------------------------------------
        # PCD parameters + planner - NEW
        # ----------------------------------------------------

        self.pcd_enable = bool(gp("pcd.enable").value)
        self.pcd_color_mode = str(gp("pcd.color_mode").value).strip().lower()
        self.pcd_min_points = int(gp("pcd.min_path_points").value)
        self.pcd_min_length = float(gp("pcd.min_path_length_m").value)
        self.pcd_defer_orange = int(gp("pcd.defer_orange_count").value)
        self.pcd_clearance = float(gp("pcd.cone_clearance_m").value)
        self.pcd_orange_policy = str(gp("pcd.orange_policy").value).strip().lower()
        self.route_enable = bool(gp("route.enable").value)
        self.route_max_jump = float(gp("route.max_jump_m").value)
        self.route_hold_ticks = int(gp("route.hold_ticks").value)
        self.route_switch_ticks = int(gp("route.switch_ticks").value)
        self.route_reset_jump = float(gp("route.reset_jump_m").value)
        self.gate_min_w = float(gp("gate.min_width_m").value)
        self.gate_max_w = float(gp("gate.max_width_m").value)

        self.pcd_cfg = PCDConfig(
            fov_vertex=self.vertex_offset,
            fov_radius=self.fov_radius,
            fov_angle_deg=self.fov_angle_deg,
            use_memory=bool(gp("pcd.use_memory").value),
            use_prediction=bool(gp("pcd.use_prediction").value),
            use_completion=bool(gp("pcd.use_completion").value),
            measure_width=bool(gp("pcd.measure_width").value),
            default_width=float(gp("pcd.track_width_m").value),
            min_spacing=self.min_spacing,
            min_car_dist=self.min_car_dist,
            max_hop=self.max_hop,
            smoothing=float(gp("pcd.smoothing").value),
            wall_margin=float(gp("pcd.wall_margin_m").value),
            use_unknown=bool(gp("pcd.use_unknown").value),
            min_curvature=bool(gp("pcd.min_curvature").value),
            fit_through_car=bool(gp("pcd.fit_through_car").value),
        )
        self.pcd = PCDPlanner(self.pcd_cfg)
        # v2 route memory state
        self._route = None              # last published route (N,2) map frame, without the car
        self._route_held = 0            # ticks the previous route has been re-used
        self._route_pending = None      # candidate route waiting to prove itself
        self._route_pending_n = 0
        self._route_state = "new"
        self._route_jump = 0.0
        self._last_car_xy = None
        self._cur_source = "TEAM"

        # colour-side latch for color_mode == "auto"
        self.pcd_blue_left = None
        if self.pcd_color_mode == "blue_left":
            self.pcd_blue_left = True
        elif self.pcd_color_mode == "blue_right":
            self.pcd_blue_left = False
        self._side_votes = 0
        self.pcd_last_reason = ""
        self._pcd_last_mode = None

        # debug topics (extra outputs only; /path_points is unchanged)
        #   /pp/mode            std_msgs/String  JSON: mode, reason, width, left colour
        #   /pp/debug_markers   visualization_msgs/MarkerArray (walls, guesses,
        #                       virtual cones, midpoints, edges) - also works in RViz
        self.declare_parameter("debug.publish", True)
        self.dbg_enable = bool(gp("debug.publish").value) and _HAS_VIZ_MSGS
        if self.dbg_enable:
            dbg_qos = QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST, depth=5,
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE)
            self.mode_pub = self.create_publisher(String, "/pp/mode", dbg_qos)
            self.marker_pub = self.create_publisher(MarkerArray, "/pp/debug_markers", dbg_qos)

        # ====================================================
        # SUBSCRIBER QoS
        # ====================================================

        sub_qos = QoSProfile(

            history=QoSHistoryPolicy.KEEP_LAST,

            depth=depth,

            reliability=(
                QoSReliabilityPolicy.BEST_EFFORT
                if best_effort
                else QoSReliabilityPolicy.RELIABLE
            ),

            durability=(
                QoSDurabilityPolicy.VOLATILE
            ),
        )

        # ====================================================
        # PUBLISHER QoS
        # ====================================================

        pub_qos = QoSProfile(

            history=QoSHistoryPolicy.KEEP_LAST,

            depth=depth,

            reliability=(
                QoSReliabilityPolicy.BEST_EFFORT
            ),

            durability=(
                QoSDurabilityPolicy.VOLATILE
            ),
        )

        # ====================================================
        # SUBSCRIBERS
        # ====================================================

        self.create_subscription(
            Odometry,
            odom_topic,
            self.cb_odom,
            sub_qos
        )

        self.create_subscription(
            ConeArrayWithCovariance,
            map_topic,
            self.cb_map,
            sub_qos
        )

        # ====================================================
        # PUBLISHER
        # ====================================================

        self.path_pub = (
            self.create_publisher(
                Path,
                path_topic,
                pub_qos
            )
        )

        # ====================================================
        # STATE
        # ====================================================

        self.data_lock = threading.Lock()

        self.car_x = None
        self.car_y = None
        self.car_yaw = None

        # Global cone positions
        self.blue_global = []
        self.yellow_global = []
        self.orange_global = []
        self.big_global = []
        self.unknown_global = []

        # ====================================================
        # LIVE VISUALIZATION STATE
        # ====================================================

        self.plot_lock = threading.Lock()

        self.plot_points_window = []
        self.plot_classes_window = []

        self.plot_edges = []

        self.plot_edge_points = []

        self.plot_candidates = []

        self.plot_path = []

        self.plot_virtual_cone = None

        self.plot_mode = "WAITING"

        self.plot_selected_side = "None"

        self.plot_blue_count = 0
        self.plot_yellow_count = 0

        # ====================================================
        # INITIALIZE PLOT
        # ====================================================

        self.fig = None
        if plt is not None and plt.get_backend().lower() != "agg":
            try:
                self._init_live_plot()
            except Exception as exc:
                self.get_logger().warn(f"Live plot unavailable: {exc}")
                if self.fig is not None:
                    plt.close(self.fig)
                self.fig = None

        # ====================================================
        # SCIpy warning
        # ====================================================

        if not _HAS_SCIPY:

            self.get_logger().warn(
                "SciPy not found. "
                "Delaunay will use k-NN fallback."
            )

        # ====================================================
        # LOG
        # ====================================================

        self.get_logger().info(
            f"odom={odom_topic}, "
            f"map={map_topic}, "
            f"path_out={path_topic}"
        )

        self.get_logger().info(
            f"FOV: radius={self.fov_radius} m, "
            f"angle={self.fov_angle_deg} deg"
        )

        self.get_logger().info(
            f"Virtual cone distance="
            f"{self.virtual_cone_distance} m"
        )

        # ====================================================
        # TIMER
        # ====================================================

        period = (
            1.0 /
            max(
                self.publish_rate_hz,
                1e-6
            )
        )

        self.timer = self.create_timer(
            period,
            self._tick
        )

    # ========================================================
    # LIVE PLOT INITIALIZATION
    # ========================================================

    def _init_live_plot(self):

        """
        Initialize the live Matplotlib visualization.

        This function does NOT affect the planner.
        """

        plt.ion()

        self.fig, self.ax = plt.subplots(
            figsize=(10, 8)
        )

        self.fig.canvas.manager.set_window_title(
            "SLAM Skidpad Path Planner - Live Visualization"
        )

        self.ax.set_aspect(
            "equal",
            adjustable="box"
        )

        self.ax.set_xlabel(
            "X [m]"
        )

        self.ax.set_ylabel(
            "Y [m]"
        )

        self.ax.set_title(
            "SLAM Skidpad Path Planner"
        )

        self.ax.grid(
            True,
            alpha=0.25
        )

        # ----------------------------------------------------
        # Plot handles
        # ----------------------------------------------------

        self.scatter_blue = (
            self.ax.scatter(
                [],
                [],
                s=45,
                marker="o",
                label="Blue"
            )
        )

        self.scatter_yellow = (
            self.ax.scatter(
                [],
                [],
                s=45,
                marker="o",
                label="Yellow"
            )
        )

        self.scatter_orange = (
            self.ax.scatter(
                [],
                [],
                s=50,
                marker="o",
                label="Orange"
            )
        )

        self.scatter_big = (
            self.ax.scatter(
                [],
                [],
                s=90,
                marker="s",
                label="Big Orange"
            )
        )

        # ----------------------------------------------------
        # Virtual cone
        # ----------------------------------------------------

        self.scatter_virtual = (
            self.ax.scatter(
                [],
                [],
                s=110,
                marker="X",
                label="Virtual Colourless"
            )
        )

        # ----------------------------------------------------
        # Candidate points
        # ----------------------------------------------------

        self.scatter_candidates = (
            self.ax.scatter(
                [],
                [],
                s=60,
                marker="x",
                label="Candidates"
            )
        )

        # ----------------------------------------------------
        # Car
        # ----------------------------------------------------

        self.scatter_car = (
            self.ax.scatter(
                [],
                [],
                s=120,
                marker="^",
                label="Car"
            )
        )

        # ----------------------------------------------------
        # Final path
        # ----------------------------------------------------

        self.path_line, = (
            self.ax.plot(
                [],
                [],
                linewidth=2.5,
                label="Greedy Path"
            )
        )

        # ----------------------------------------------------
        # Car heading
        # ----------------------------------------------------

        self.heading_line, = (
            self.ax.plot(
                [],
                [],
                linewidth=2.0
            )
        )

        # ----------------------------------------------------
        # FOV wedge
        # ----------------------------------------------------

        self.fov_patch = None

        # ----------------------------------------------------
        # Delaunay edge lines
        # ----------------------------------------------------

        self.edge_lines = []

        # ----------------------------------------------------
        # Status text
        # ----------------------------------------------------

        self.status_text = (
            self.ax.text(
                0.02,
                0.98,
                "",
                transform=self.ax.transAxes,
                verticalalignment="top",
                fontsize=11,
                bbox=dict(
                    boxstyle="round",
                    facecolor="white",
                    alpha=0.85
                )
            )
        )

        self.ax.legend(
            loc="upper right"
        )

        plt.show(
            block=False
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # ========================================================
    # ODOMETRY CALLBACK
    # ========================================================

    def cb_odom(
        self,
        msg: Odometry
    ):

        x = float(
            msg.pose.pose.position.x
        )

        y = float(
            msg.pose.pose.position.y
        )

        q = (
            msg.pose.pose.orientation
        )

        yaw = yaw_from_quat(
            q.x,
            q.y,
            q.z,
            q.w
        )

        with self.data_lock:

            self.car_x = x
            self.car_y = y
            self.car_yaw = yaw

    # ========================================================
    # MAP CALLBACK
    # ========================================================

    def cb_map(
        self,
        msg: ConeArrayWithCovariance
    ):

        blue = []
        yellow = []
        orange = []
        big = []

        # ----------------------------------------------------
        # Blue
        # ----------------------------------------------------

        for c in msg.blue_cones:

            blue.append(
                (
                    float(c.point.x),
                    float(c.point.y)
                )
            )

        # ----------------------------------------------------
        # Yellow
        # ----------------------------------------------------

        for c in msg.yellow_cones:

            yellow.append(
                (
                    float(c.point.x),
                    float(c.point.y)
                )
            )

        # ----------------------------------------------------
        # Orange
        # ----------------------------------------------------

        for c in msg.orange_cones:

            orange.append(
                (
                    float(c.point.x),
                    float(c.point.y)
                )
            )

        # ----------------------------------------------------
        # Big orange
        # ----------------------------------------------------

        for c in msg.big_orange_cones:

            big.append(
                (
                    float(c.point.x),
                    float(c.point.y)
                )
            )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        with self.data_lock:

            self.blue_global = blue
            self.yellow_global = yellow
            self.orange_global = orange
            self.big_global = big
            self.unknown_global = [
                (float(c.point.x), float(c.point.y))
                for c in getattr(msg, "unknown_color_cones", [])
            ]

    # ========================================================
    # WORLD -> LOCAL
    # ========================================================

    def _world_to_local(
        self,
        car_x,
        car_y,
        car_yaw,
        px,
        py
    ):

        """
        World/map frame -> car frame.

        Car frame:

            +X = forward
            +Y = left
        """

        c = math.cos(
            car_yaw
        )

        s = math.sin(
            car_yaw
        )

        dx = px - car_x
        dy = py - car_y

        lx = (
            c * dx
            +
            s * dy
        )

        ly = (
            -s * dx
            +
            c * dy
        )

        return (
            lx,
            ly
        )

    # ========================================================
    # COMPUTE FOV
    # ========================================================

    def _compute_fov_points(
        self,
        car_x,
        car_y,
        car_yaw
    ):

        """
        Select cones inside the FOV sector.
        """

        vertex_offset = (
            self.vertex_offset
        )

        R = self.fov_radius

        half_angle = (
            self.fov_half_rad
        )

        points_window = []
        classes_window = []

        # ----------------------------------------------------
        # Add list
        # ----------------------------------------------------

        def add_list(
            cones,
            cls_name
        ):

            for px, py in cones:

                lx, ly = (
                    self._world_to_local(
                        car_x,
                        car_y,
                        car_yaw,
                        px,
                        py
                    )
                )

                vx = (
                    lx -
                    vertex_offset
                )

                vy = ly

                r = math.hypot(
                    vx,
                    vy
                )

                if (
                    r > R
                    or
                    r < 1e-3
                ):

                    continue

                theta = math.atan2(
                    vy,
                    vx
                )

                if (
                    abs(theta)
                    <=
                    half_angle
                ):

                    points_window.append(
                        (
                            px,
                            py
                        )
                    )

                    classes_window.append(
                        cls_name
                    )

        # ----------------------------------------------------
        # All cone types
        # ----------------------------------------------------

        add_list(
            self.blue_global,
            "blue"
        )

        add_list(
            self.yellow_global,
            "yellow"
        )

        add_list(
            self.orange_global,
            "orange"
        )

        add_list(
            self.big_global,
            "big"
        )

        return (
            points_window,
            classes_window
        )

    # ========================================================
    # VIRTUAL COLOURLESS CONE
    # ========================================================

    def _get_virtual_cone(
        self,
        car_x,
        car_y,
        car_yaw
    ):

        """
        Create a virtual cone in front of the car.

        Returned in GLOBAL coordinates.
        """

        d = (
            self.virtual_cone_distance
        )

        virtual_x = (
            car_x
            +
            d * math.cos(
                car_yaw
            )
        )

        virtual_y = (
            car_y
            +
            d * math.sin(
                car_yaw
            )
        )

        return (
            virtual_x,
            virtual_y
        )

    # ========================================================
    # FALLBACK DELAUNAY
    # ========================================================

    def _build_fallback_edges(
        self,
        points_window,
        classes_window,
        car_x,
        car_y,
        car_yaw
    ):

        """
        Fallback mode:

            selected colour cones
                    +
            virtual colourless cone
                    ↓
                Delaunay
                    ↓
                  edges
        """

        # ----------------------------------------------------
        # Separate blue and yellow
        # ----------------------------------------------------

        blue_points = [
            points_window[i]
            for i, cls
            in enumerate(
                classes_window
            )
            if cls == "blue"
        ]

        yellow_points = [
            points_window[i]
            for i, cls
            in enumerate(
                classes_window
            )
            if cls == "yellow"
        ]

        blue_count = len(
            blue_points
        )

        yellow_count = len(
            yellow_points
        )

        # ----------------------------------------------------
        # Select side with MORE cones
        # ----------------------------------------------------

        if blue_count > yellow_count:

            selected_points = (
                list(
                    blue_points
                )
            )

            selected_colour = "blue"

        elif yellow_count > blue_count:

            selected_points = (
                list(
                    yellow_points
                )
            )

            selected_colour = "yellow"

        else:

            # Equal and neither side has enough
            if (
                blue_count < 2
                and
                yellow_count < 2
            ):

                return (
                    [],
                    [],
                    [],
                    None
                )

            return (
                [],
                [],
                [],
                None
            )

        # ----------------------------------------------------
        # Need >= 2 cones
        # ----------------------------------------------------

        if len(
            selected_points
        ) < 2:

            return (
                [],
                [],
                [],
                None
            )

        # ----------------------------------------------------
        # Virtual cone
        # ----------------------------------------------------

        virtual_cone = (
            self._get_virtual_cone(
                car_x,
                car_y,
                car_yaw
            )
        )

        # ----------------------------------------------------
        # Build working set
        # ----------------------------------------------------

        fallback_points = (
            list(
                selected_points
            )
        )

        fallback_classes = [
            selected_colour
            for _ in selected_points
        ]

        fallback_points.append(
            virtual_cone
        )

        fallback_classes.append(
            "colorless"
        )

        # ----------------------------------------------------
        # Delaunay
        # ----------------------------------------------------

        N = len(
            fallback_points
        )

        if N < 2:

            return (
                [],
                fallback_points,
                fallback_classes,
                selected_colour
            )

        pts = np.asarray(
            fallback_points,
            dtype=float
        )

        edges_set = set()

        # ----------------------------------------------------
        # Delaunay
        # ----------------------------------------------------

        if (
            _HAS_SCIPY
            and
            N >= 3
        ):

            try:

                tri = Delaunay(
                    pts
                )

                for simplex in (
                    tri.simplices
                ):

                    i = int(
                        simplex[0]
                    )

                    j = int(
                        simplex[1]
                    )

                    k = int(
                        simplex[2]
                    )

                    for a, b in (
                        (i, j),
                        (j, k),
                        (k, i)
                    ):

                        if a > b:

                            a, b = (
                                b,
                                a
                            )

                        edges_set.add(
                            (a, b)
                        )

            except Exception as e:

                self.get_logger().warn(
                    f"Fallback Delaunay failed: {e}"
                )

                # --------------------------------------------
                # k-NN fallback
                # --------------------------------------------

                k = min(
                    3,
                    N - 1
                )

                for i in range(N):

                    d2 = np.sum(
                        (
                            pts
                            -
                            pts[i]
                        ) ** 2,
                        axis=1
                    )

                    d2[i] = (
                        float("inf")
                    )

                    nn_idx = (
                        np.argsort(
                            d2
                        )[:k]
                    )

                    for j in nn_idx:

                        a = i
                        b = int(j)

                        if a > b:

                            a, b = (
                                b,
                                a
                            )

                        edges_set.add(
                            (a, b)
                        )

        else:

            # ------------------------------------------------
            # k-NN
            # ------------------------------------------------

            k = min(
                3,
                N - 1
            )

            for i in range(N):

                d2 = np.sum(
                    (
                        pts
                        -
                        pts[i]
                    ) ** 2,
                    axis=1
                )

                d2[i] = (
                    float("inf")
                )

                nn_idx = (
                    np.argsort(
                        d2
                    )[:k]
                )

                for j in nn_idx:

                    a = i
                    b = int(j)

                    if a > b:

                        a, b = (
                            b,
                            a
                        )

                    edges_set.add(
                        (a, b)
                    )

        # ----------------------------------------------------
        # Edge length filtering
        # ----------------------------------------------------

        edges = []

        max_len2 = (
            6.0 *
            6.0
        )

        for i, j in edges_set:

            x1, y1 = (
                fallback_points[i]
            )

            x2, y2 = (
                fallback_points[j]
            )

            d2 = (
                (x1 - x2) ** 2
                +
                (y1 - y2) ** 2
            )

            if (
                d2 >=
                max_len2
            ):

                continue

            edges.append(
                (i, j)
            )

        return (
            edges,
            fallback_points,
            fallback_classes,
            selected_colour
        )

    # ========================================================
    # NORMAL DELAUNAY
    # ========================================================

    def _build_edges(
        self,
        points_window,
        classes_window
    ):

        """
        Original Delaunay algorithm.
        """

        N = len(
            points_window
        )

        if N < 2:

            return []

        pts = np.asarray(
            points_window,
            dtype=float
        )

        edges_set = set()

        # ----------------------------------------------------
        # Delaunay
        # ----------------------------------------------------

        if (
            _HAS_SCIPY
            and
            N >= 3
        ):

            try:
                simplices = Delaunay(pts).simplices
            except QhullError:
                simplices = ()

            for simplex in simplices:

                i = int(
                    simplex[0]
                )

                j = int(
                    simplex[1]
                )

                k = int(
                    simplex[2]
                )

                for a, b in (
                    (i, j),
                    (j, k),
                    (k, i)
                ):

                    if a > b:

                        a, b = (
                            b,
                            a
                        )

                    edges_set.add(
                        (a, b)
                    )

        if not edges_set:

            # ------------------------------------------------
            # k-NN
            # ------------------------------------------------

            k = min(
                3,
                N - 1
            )

            for i in range(N):

                d2 = np.sum(
                    (
                        pts
                        -
                        pts[i]
                    ) ** 2,
                    axis=1
                )

                d2[i] = (
                    float("inf")
                )

                nn_idx = (
                    np.argsort(
                        d2
                    )[:k]
                )

                for j in nn_idx:

                    a = i
                    b = int(j)

                    if a > b:

                        a, b = (
                            b,
                            a
                        )

                    edges_set.add(
                        (a, b)
                    )

        # ----------------------------------------------------
        # Original constraints
        # ----------------------------------------------------

        orange_like = {
            "orange",
            "big"
        }

        blue_yellow = {
            "blue",
            "yellow"
        }

        edges = []

        max_len2 = (
            6.0 *
            6.0
        )

        for i, j in edges_set:

            ci = (
                classes_window[i]
            )

            cj = (
                classes_window[j]
            )

            x1, y1 = (
                points_window[i]
            )

            x2, y2 = (
                points_window[j]
            )

            d2 = (
                (x1 - x2) ** 2
                +
                (y1 - y2) ** 2
            )

            # ------------------------------------------------
            # Maximum edge length
            # ------------------------------------------------

            if (
                d2 >=
                max_len2
            ):

                continue

            # ------------------------------------------------
            # Blue-blue
            # ------------------------------------------------

            if (
                ci == "blue"
                and
                cj == "blue"
            ):

                continue

            # ------------------------------------------------
            # Yellow-yellow
            # ------------------------------------------------

            if (
                ci == "yellow"
                and
                cj == "yellow"
            ):

                continue

            # ------------------------------------------------
            # Orange <-> blue/yellow
            # ------------------------------------------------

            if (
                (
                    ci in orange_like
                    and
                    cj in blue_yellow
                )
                or
                (
                    cj in orange_like
                    and
                    ci in blue_yellow
                )
            ):

                continue

            edges.append(
                (i, j)
            )

        return edges

    # ========================================================
    # BIG ORANGE SPECIAL
    # ========================================================

    def _big_cone_candidate(
        self,
        points_window,
        classes_window
    ):
        """
        Big-orange gate point (v2).

        v1 averaged ANY two big cones. Each side of a start/finish gate has a pair of big cones
        ~0.4 m apart, so two cones of the SAME side gave a "gate centre" on the track edge
        (PP-04, Small Track frame 2929, a 2.3 m path jump). v2 splits the big cones by the side
        of the car they are on and returns the midpoint of the two side centroids only when both
        sides are present and they are a plausible track width apart (gate.min/max_width_m).
        Otherwise no gate point is returned and the walls decide the path.
        """
        big_pts = [
            points_window[i]
            for i, cls in enumerate(classes_window)
            if cls == "big"
        ]
        if len(big_pts) < 2 or self.car_x is None:
            return []
        c, s_ = math.cos(self.car_yaw), math.sin(self.car_yaw)
        left, right = [], []
        for x, y in big_pts:
            lat = -s_ * (x - self.car_x) + c * (y - self.car_y)
            (left if lat >= 0.0 else right).append((x, y))
        if not left or not right:
            return []
        lc = np.mean(np.asarray(left, float), axis=0)
        rc = np.mean(np.asarray(right, float), axis=0)
        width = float(np.linalg.norm(lc - rc))
        if not (self.gate_min_w <= width <= self.gate_max_w):
            return []
        m = 0.5 * (lc + rc)
        return [(float(m[0]), float(m[1]), True)]

    # ========================================================

    def _filter_centroids_min_spacing(
        self,
        centroids_raw,
        car_x,
        car_y,
        min_dist
    ):

        """
        Candidate filtering:

        1. Farthest-first.
        2. Reject candidates too close to car.
        3. Reject candidates too close to an already
           accepted candidate.
        """

        if not centroids_raw:

            return []

        pts = np.array(
            [
                [
                    c[0],
                    c[1]
                ]
                for c in centroids_raw
            ],
            dtype=float
        )

        car = np.array(
            [
                car_x,
                car_y
            ],
            dtype=float
        )

        d2_car = np.sum(
            (
                pts - car
            ) ** 2,
            axis=1
        )

        order = np.argsort(
            -d2_car
        )

        kept_pts = []
        kept = []

        for idx in order:

            p = pts[idx]

            # ------------------------------------------------
            # Car distance
            # ------------------------------------------------

            if (
                np.linalg.norm(
                    p - car
                )
                <
                self.min_car_dist
            ):

                continue

            # ------------------------------------------------
            # First candidate
            # ------------------------------------------------

            if not kept_pts:

                kept_pts.append(
                    p
                )

                kept.append(
                    (
                        float(
                            p[0]
                        ),
                        float(
                            p[1]
                        ),
                        bool(
                            centroids_raw[
                                idx
                            ][2]
                        )
                    )
                )

                continue

            # ------------------------------------------------
            # Spacing
            # ------------------------------------------------

            too_close = False

            for kp in kept_pts:

                if (
                    np.linalg.norm(
                        p - kp
                    )
                    <
                    min_dist
                ):

                    too_close = True
                    break

            if too_close:

                continue

            kept_pts.append(
                p
            )

            kept.append(
                (
                    float(
                        p[0]
                    ),
                    float(
                        p[1]
                    ),
                    bool(
                        centroids_raw[
                            idx
                        ][2]
                    )
                )
            )

        return kept

    # ========================================================
    # GREEDY PATH
    # ========================================================

    def _build_greedy_path(
        self,
        car_x,
        car_y,
        candidate_points
    ):

        """
        Greedy nearest-neighbour path.
        """

        if not candidate_points:

            return []

        pts = np.asarray(
            candidate_points,
            dtype=float
        )

        N = (
            pts.shape[0]
        )

        used = np.zeros(
            N,
            dtype=bool
        )

        current = np.array(
            [
                car_x,
                car_y
            ],
            dtype=float
        )

        max_step2 = (
            self.max_hop
            *
            self.max_hop
        )

        order = []

        for _ in range(N):

            diff = (
                pts -
                current
            )

            d2 = np.sum(
                diff * diff,
                axis=1
            )

            # Already used
            d2[
                used
            ] = float(
                "inf"
            )

            # Too far
            d2[
                d2 > max_step2
            ] = float(
                "inf"
            )

            idx = int(
                np.argmin(
                    d2
                )
            )

            if (
                not math.isfinite(
                    float(
                        d2[idx]
                    )
                )
                or
                float(
                    d2[idx]
                )
                ==
                float("inf")
            ):

                break

            used[idx] = True

            order.append(
                idx
            )

            current = pts[idx]

        return [
            tuple(
                pts[i]
            )
            for i in order
        ]

    # ========================================================
    # PUBLISH PATH
    # ========================================================

    # ========================================================
    # v2 ROUTE MEMORY (applies to every published path)
    # ========================================================

    @staticmethod
    def _poly_project(poly, p):
        """(distance from p to the polyline, arc length of the closest point, segment index)."""
        seg = np.diff(poly, axis=0)
        L2 = np.maximum((seg * seg).sum(1), 1e-12)
        f = np.clip(((p - poly[:-1]) * seg).sum(1) / L2, 0.0, 1.0)
        q = poly[:-1] + f[:, None] * seg
        d = np.linalg.norm(q - p, axis=1)
        i = int(np.argmin(d))
        arc = np.r_[0.0, np.cumsum(np.sqrt(L2))]
        return float(d[i]), float(arc[i] + f[i] * np.sqrt(L2[i])), i

    @staticmethod
    def _point_at(poly, s):
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))]
        s = float(np.clip(s, 0.0, arc[-1]))
        return np.array([np.interp(s, arc, poly[:, 0]), np.interp(s, arc, poly[:, 1])])

    @staticmethod
    def _plen(car, path):
        return float(np.linalg.norm(np.diff(np.vstack([car, path]), axis=0), axis=1).sum())

    def _route_remaining(self, car):
        """The previous route from the car's projection onward, or None if it is no longer usable
        (car > 3 m from it, < 4 points / pcd.min_path_length_m left, or cut short by real cones)."""
        if self._route is None or len(self._route) < 2:
            return None
        poly = self._route
        d, s_car, i = self._poly_project(poly, car)
        if d > 3.0:
            return None
        rest = poly[i + 1:]
        rest = rest[np.linalg.norm(rest - car, axis=1) > 0.3]
        if self.pcd_clearance > 0 and len(rest):
            rest = self._cut_at_walls(rest, car, self.blue_global, self.yellow_global)
        if len(rest) < 4 or self._plen(car, rest) < self.pcd_min_length:
            return None
        return np.vstack([car, rest])

    def _anchor(self, car, path, s=4.0):
        P = np.asarray(path, float).reshape(-1, 2)
        if np.linalg.norm(P[0] - car) > 1e-6:
            P = np.vstack([car, P])
        return self._point_at(P, s)

    def _publish_path(
        self,
        car_x,
        car_y,
        path_points
    ):
        """v2: route-memory filter in front of the original publisher.

        Each tick a candidate route arrives (PCD or the team logic). Team-logic routes are cut
        at real cones / wall segments first (PCD routes already are). The candidate replaces the
        previous route when it is valid and its 4 m look-ahead point moved <= route.max_jump_m.
        A jump or an invalid candidate keeps the previous route (trimmed to the car, re-checked
        against cones) for at most route.hold_ticks; a different route is taken over once it has
        persisted for route.switch_ticks, or when the hold expires. A car jump > route.reset_jump_m
        (simulator reset) clears the route and PCD's cone memory."""
        if not self.route_enable:
            return self._publish_path_raw(car_x, car_y, path_points)
        car = np.array([float(car_x), float(car_y)])
        if self._last_car_xy is not None and np.linalg.norm(car - self._last_car_xy) > self.route_reset_jump:
            self._route, self._route_pending, self._route_pending_n, self._route_held = None, None, 0, 0
            self.pcd = PCDPlanner(self.pcd_cfg)
        self._last_car_xy = car

        cand = np.asarray(path_points, dtype=float).reshape(-1, 2)
        note = ""
        if self._cur_source == "TEAM" and self.pcd_clearance > 0 and len(cand) > 1:
            # team paths have points 2-6 m apart: check a 0.25 m resampling so a segment that
            # jumps over a cone row is caught, and keep the original points before the hit
            dense = _resample(cand, 0.25)
            cutd = self._cut_at_walls(dense[1:], car, self.blue_global, self.yellow_global)
            if len(cutd) < len(dense) - 1:
                s_hit = float(np.linalg.norm(np.diff(dense[:len(cutd) + 1], axis=0), axis=1).sum())
                arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(cand, axis=0), axis=1))]
                kept = cand[arc < s_hit] if s_hit > 0 else cand[:0]
                kept = kept if len(kept) else cand[:1]
                end = dense[len(cutd)] if len(cutd) else dense[0]
                new = np.vstack([kept, end]) if np.linalg.norm(kept[-1] - end) > 0.05 else kept
                note = f", team path cut at cones {arc[-1]:.1f}->{s_hit:.1f} m"
                cand = new
        valid = len(cand) >= 4 and self._plen(car, cand) >= self.pcd_min_length
        prev = self._route_remaining(car)

        jump = 0.0
        if valid and prev is not None:
            jump = float(np.linalg.norm(self._anchor(car, cand) - self._anchor(car, prev)))
        self._route_jump = jump
        can_hold = prev is not None and self._route_held < self.route_hold_ticks

        def take(route, state):
            if len(route):
                self._route = np.asarray(route, float).reshape(-1, 2)
            self._route_held = 0
            self._route_pending, self._route_pending_n = None, 0
            self._route_state = state + note
            return self._publish_path_raw(car_x, car_y, [tuple(p) for p in route])

        def hold(why):
            self._route_held += 1
            self._route_state = f"hold {self._route_held}/{self.route_hold_ticks}: {why}" + note
            return self._publish_path_raw(car_x, car_y, [tuple(p) for p in prev])

        if prev is None:
            return take(cand, "new")
        if valid and jump <= self.route_max_jump:
            return take(cand, "follow")
        if valid:                                          # a different route
            if self._route_pending is not None and \
                    np.linalg.norm(self._anchor(car, cand) - self._anchor(car, self._route_pending)) <= self.route_max_jump:
                self._route_pending_n += 1
            else:
                self._route_pending_n = 1
            self._route_pending = cand
            if self._route_pending_n >= self.route_switch_ticks or not can_hold:
                return take(cand, f"switch (jump {jump:.1f} m)")
            return hold(f"candidate jumps {jump:.1f} m")
        if can_hold:
            return hold("no valid candidate")
        self._route_state = "no route" + note
        return self._publish_path_raw(car_x, car_y, [tuple(p) for p in cand])

    def _publish_path_raw(
        self,
        car_x,
        car_y,
        path_points
    ):

        msg = Path()

        msg.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        msg.header.frame_id = (
            "map"
        )

        poses = []

        # ----------------------------------------------------
        # Car pose
        # ----------------------------------------------------

        car_ps = PoseStamped()

        car_ps.header = (
            msg.header
        )

        car_ps.pose.position.x = (
            float(car_x)
        )

        car_ps.pose.position.y = (
            float(car_y)
        )

        car_ps.pose.position.z = 0.0

        car_ps.pose.orientation.w = 1.0

        poses.append(
            car_ps
        )

        # ----------------------------------------------------
        # Path points
        # ----------------------------------------------------

        for x, y in path_points:

            ps = PoseStamped()

            ps.header = (
                msg.header
            )

            ps.pose.position.x = (
                float(x)
            )

            ps.pose.position.y = (
                float(y)
            )

            ps.pose.position.z = 0.0

            ps.pose.orientation.w = 1.0

            poses.append(
                ps
            )

        msg.poses = poses

        self.path_pub.publish(
            msg
        )

    # ========================================================
    # UPDATE LIVE VISUALIZATION
    # ========================================================

    def _update_live_plot(
        self,
        car_x,
        car_y,
        car_yaw
    ):

        """
        Update the live Matplotlib visualization.

        This function ONLY visualizes already-computed
        algorithm data.
        """

        if self.fig is None or not plt.fignum_exists(
            self.fig.number
        ):

            return

        with self.plot_lock:

            points_window = list(
                self.plot_points_window
            )

            classes_window = list(
                self.plot_classes_window
            )

            edges = list(
                self.plot_edges
            )

            edge_points = list(
                self.plot_edge_points
            )

            candidates = list(
                self.plot_candidates
            )

            path = list(
                self.plot_path
            )

            virtual_cone = (
                self.plot_virtual_cone
            )

            mode = (
                self.plot_mode
            )

            selected_side = (
                self.plot_selected_side
            )

            blue_count = (
                self.plot_blue_count
            )

            yellow_count = (
                self.plot_yellow_count
            )

        # ====================================================
        # UPDATE CONE SCATTERS
        # ====================================================

        def get_points(
            cls_name
        ):

            return [
                p
                for p, c
                in zip(
                    points_window,
                    classes_window
                )
                if c == cls_name
            ]

        blue_pts = get_points(
            "blue"
        )

        yellow_pts = get_points(
            "yellow"
        )

        orange_pts = get_points(
            "orange"
        )

        big_pts = get_points(
            "big"
        )

        # ----------------------------------------------------
        # Helper for scatter
        # ----------------------------------------------------

        def set_scatter(
            scatter,
            points
        ):

            if points:

                scatter.set_offsets(
                    np.asarray(
                        points
                    )
                )

            else:

                scatter.set_offsets(
                    np.empty(
                        (0, 2)
                    )
                )

        set_scatter(
            self.scatter_blue,
            blue_pts
        )

        set_scatter(
            self.scatter_yellow,
            yellow_pts
        )

        set_scatter(
            self.scatter_orange,
            orange_pts
        )

        set_scatter(
            self.scatter_big,
            big_pts
        )

        # ====================================================
        # VIRTUAL CONE
        # ====================================================

        if virtual_cone is not None:

            self.scatter_virtual.set_offsets(
                np.asarray(
                    [virtual_cone]
                )
            )

        else:

            self.scatter_virtual.set_offsets(
                np.empty(
                    (0, 2)
                )
            )

        # ====================================================
        # CANDIDATES
        # ====================================================

        set_scatter(
            self.scatter_candidates,
            candidates
        )

        # ====================================================
        # CAR
        # ====================================================

        self.scatter_car.set_offsets(
            np.asarray(
                [
                    [
                        car_x,
                        car_y
                    ]
                ]
            )
        )

        # ====================================================
        # FINAL PATH
        # ====================================================

        if path:

            path_arr = np.asarray(
                path
            )

            self.path_line.set_data(
                path_arr[:, 0],
                path_arr[:, 1]
            )

        else:

            self.path_line.set_data(
                [],
                []
            )

        # ====================================================
        # CAR HEADING
        # ====================================================

        heading_length = 2.0

        hx = (
            car_x
            +
            heading_length *
            math.cos(
                car_yaw
            )
        )

        hy = (
            car_y
            +
            heading_length *
            math.sin(
                car_yaw
            )
        )

        self.heading_line.set_data(
            [
                car_x,
                hx
            ],
            [
                car_y,
                hy
            ]
        )

        # ====================================================
        # REMOVE OLD DELAUNAY LINES
        # ====================================================

        for line in (
            self.edge_lines
        ):

            try:

                line.remove()

            except Exception:

                pass

        self.edge_lines = []

        # ====================================================
        # DRAW CURRENT DELAUNAY EDGES
        # ====================================================

        for i, j in edges:

            try:

                x1, y1 = (
                    edge_points[i]
                )

                x2, y2 = (
                    edge_points[j]
                )

            except (
                IndexError,
                ValueError
            ):

                continue

            line, = self.ax.plot(
                [
                    x1,
                    x2
                ],
                [
                    y1,
                    y2
                ],
                linewidth=1.0,
                alpha=0.55
            )

            self.edge_lines.append(
                line
            )

        # ====================================================
        # REMOVE OLD FOV
        # ====================================================

        if self.fov_patch is not None:

            try:

                self.fov_patch.remove()

            except Exception:

                pass

            self.fov_patch = None

        # ====================================================
        # DRAW FOV SECTOR
        # ====================================================

        # FOV vertex is behind the car by vertex_offset.
        #
        # In global frame:
        #
        # vertex = car - vertex_offset * heading
        #
        # Since vertex_offset is normally -5:
        # vertex is 5 m behind the car.

        vertex_x = (
            car_x
            +
            self.vertex_offset *
            math.cos(
                car_yaw
            )
        )

        vertex_y = (
            car_y
            +
            self.vertex_offset *
            math.sin(
                car_yaw
            )
        )

        start_angle = (
            math.degrees(
                car_yaw
            )
            -
            self.fov_angle_deg / 2.0
        )

        self.fov_patch = Wedge(
            (
                vertex_x,
                vertex_y
            ),
            self.fov_radius,
            start_angle,
            start_angle
            +
            self.fov_angle_deg,
            fill=False,
            linewidth=1.2,
            alpha=0.35
        )

        self.ax.add_patch(
            self.fov_patch
        )

        # ====================================================
        # STATUS
        # ====================================================

        if mode == "FALLBACK":

            status = (
                f"MODE: FALLBACK\n"
                f"Blue in FOV: {blue_count}\n"
                f"Yellow in FOV: {yellow_count}\n"
                f"Selected side: "
                f"{selected_side}\n"
                f"Virtual cone: "
                f"{self.virtual_cone_distance:.1f} m"
            )

        elif mode == "PCD":

            status = (
                f"MODE: PCD\n"
                f"Blue in FOV: {blue_count}\n"
                f"Yellow in FOV: {yellow_count}\n"
                f"Left wall: {selected_side}\n"
                f"Track width: {self.pcd.width:.2f} m"
            )

        elif mode == "NORMAL":

            status = (
                f"MODE: NORMAL\n"
                f"Blue in FOV: {blue_count}\n"
                f"Yellow in FOV: {yellow_count}\n"
                f"Delaunay: all FOV cones"
            )

        else:

            status = (
                f"MODE: {mode}\n"
                f"Blue in FOV: {blue_count}\n"
                f"Yellow in FOV: {yellow_count}"
            )

        self.status_text.set_text(
            status
        )

        # ====================================================
        # AUTO VIEW
        # ====================================================

        # Keep car visible with a reasonable local window.
        view_radius = max(
            15.0,
            min(
                self.fov_radius + 5.0,
                35.0
            )
        )

        self.ax.set_xlim(
            car_x - view_radius,
            car_x + view_radius
        )

        self.ax.set_ylim(
            car_y - view_radius,
            car_y + view_radius
        )

        # ====================================================
        # TITLE
        # ====================================================

        self.ax.set_title(
            "SLAM Skidpad Path Planner — Live"
        )

        # ====================================================
        # REDRAW
        # ====================================================

        self.fig.canvas.draw_idle()

        self.fig.canvas.flush_events()

        plt.pause(
            0.001
        )

    # ========================================================
    # MAIN TICK
    # ========================================================

    # ========================================================
    # PCD HELPERS (NEW)
    # ========================================================

    def _log_mode(self, mode):

        if mode != self._pcd_last_mode:

            if mode == "PCD":
                self.get_logger().info("Planner: PCD")
            else:
                self.get_logger().info(
                    f"Planner: original NORMAL/FALLBACK "
                    f"({self.pcd_last_reason})"
                )

            self._pcd_last_mode = mode

    def _update_color_side(self, car_x, car_y, car_yaw, blue_global, yellow_global):
        """
        color_mode == "auto": decide which colour is the LEFT wall.

        Uses cones 0-15 m ahead of the car: if the blue cones
        are on average to the left of the yellow cones, blue is
        the left wall. The answer is latched after 5 agreeing
        frames, and only flipped after 40 agreeing frames the
        other way (then PCD memory is reset).
        """

        if self.pcd_color_mode in ("blue_left", "blue_right"):
            return

        c, s = math.cos(car_yaw), math.sin(car_yaw)

        def lateral(cones):
            out = []
            for px, py in cones:
                dx, dy = px - car_x, py - car_y
                fwd = c * dx + s * dy
                lat = -s * dx + c * dy           # + = left of the car
                if 0.0 < fwd < 15.0 and abs(lat) < 8.0:
                    out.append(lat)
            return out

        lb, ly = lateral(blue_global), lateral(yellow_global)

        if len(lb) < 2 or len(ly) < 2:
            return

        vote = float(np.mean(lb)) > float(np.mean(ly))   # True = blue left

        if self.pcd_blue_left is None:
            self._side_votes = self._side_votes + 1 if vote else self._side_votes - 1
            if abs(self._side_votes) >= 5:
                self.pcd_blue_left = self._side_votes > 0
                self._side_votes = 0
                self.get_logger().info(
                    "PCD colours: BLUE is on the "
                    + ("LEFT" if self.pcd_blue_left else "RIGHT")
                )
            return

        if vote != self.pcd_blue_left:
            self._side_votes += 1
            if self._side_votes >= 40:
                self.pcd_blue_left = vote
                self._side_votes = 0
                self.pcd = PCDPlanner(self.pcd_cfg)
                self.get_logger().warn(
                    "PCD colours flipped: BLUE is now on the "
                    + ("LEFT" if vote else "RIGHT")
                )
        else:
            self._side_votes = 0

    def _try_pcd(self, car_x, car_y, car_yaw, blue_global, yellow_global, classes_window):
        """
        Returns a list of (x, y) map-frame points (car point NOT
        included - _publish_path adds it), or None when the
        original logic should be used instead.
        """

        # v2: learn the colour side FIRST (v1 returned on the orange test before this, so the
        # side stayed unknown while the car stood among the start-line cones -> PP-04 opening jump)
        self._update_color_side(car_x, car_y, car_yaw, blue_global, yellow_global)

        n_orange = sum(1 for cls in classes_window if cls in ("orange", "big"))
        n_by = sum(1 for cls in classes_window if cls in ("blue", "yellow"))
        if self.pcd_defer_orange > 0 and n_orange >= self.pcd_defer_orange:
            if self.pcd_orange_policy == "defer" or n_by < 2:
                # v1 behaviour, or a genuinely orange-only scene (start lane, skidpad lanes)
                self.pcd_last_reason = f"{n_orange} orange cones in FOV" + ("" if self.pcd_orange_policy == "defer" else ", no blue/yellow")
                return None
            # "validate": fall through and let PCD try; the checks below decide

        if self.pcd_blue_left is None:
            self.pcd_last_reason = "colour side not known yet"
            return None

        if self.pcd_blue_left:
            left, right = blue_global, yellow_global
        else:
            left, right = yellow_global, blue_global

        car = np.array([car_x, car_y], dtype=float)
        cdir = np.array([math.cos(car_yaw), math.sin(car_yaw)], dtype=float)

        try:
            path = self.pcd.plan(
                np.asarray(left, dtype=float).reshape(-1, 2),
                np.asarray(right, dtype=float).reshape(-1, 2),
                car,
                cdir,
                unknown=np.asarray(getattr(self, "unknown_global", []), dtype=float).reshape(-1, 2),
            )
        except Exception as exc:                              # never kill the node
            self.get_logger().error(f"PCD failed: {exc}", throttle_duration_sec=2.0)
            self.pcd_last_reason = "PCD exception"
            return None

        path = np.asarray(path, dtype=float).reshape(-1, 2)

        # the car pose is prepended by _publish_path -> drop points on top of the car
        if len(path):
            path = path[np.linalg.norm(path - car, axis=1) > 0.3]

        # last safety net: cut where the path comes too close to the
        # REAL walls of the map = every blue/yellow cone plus every
        # same-colour Delaunay edge shorter than 6.5 m (the wall
        # segments between neighbouring cones), whether PCD traced
        # that wall or not
        if self.pcd_clearance > 0 and len(path):
            path = self._cut_at_walls(path, car, blue_global, yellow_global)

        if len(path) < self.pcd_min_points:
            self.pcd_last_reason = f"PCD path too short ({len(path)} pts)"
            return None

        length = float(np.linalg.norm(np.diff(np.vstack([car, path]), axis=0), axis=1).sum())

        if length < self.pcd_min_length:
            self.pcd_last_reason = f"PCD path too short ({length:.1f} m)"
            return None

        self.pcd_last_reason = ""
        return [(float(x), float(y)) for x, y in path]

    def _cut_at_walls(self, path, car, blue_global, yellow_global):

        B = np.asarray(blue_global, dtype=float).reshape(-1, 2)
        Y = np.asarray(yellow_global, dtype=float).reshape(-1, 2)
        near = self.fov_radius + 10.0
        B = B[np.linalg.norm(B - car, axis=1) < near] if len(B) else B
        Y = Y[np.linalg.norm(Y - car, axis=1) < near] if len(Y) else Y
        P = np.vstack([B, Y])

        if len(P) == 0:
            return path

        d = cKDTree(P).query(path)[0]                         # cones

        if len(P) >= 3:
            col = np.r_[np.zeros(len(B), int), np.ones(len(Y), int)]
            segs = set()
            try:
                for tri in _PCDDelaunay(P).simplices:
                    for i, j in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                        if col[i] == col[j] and np.linalg.norm(P[i] - P[j]) < 6.5:
                            segs.add((min(i, j), max(i, j)))
            except Exception:
                segs = set()
            if segs:
                S = np.array(sorted(segs))
                a, b = P[S[:, 0]], P[S[:, 1]]
                ab = b - a
                t = np.clip(((path[:, None] - a[None]) * ab[None]).sum(2)
                            / np.maximum((ab * ab).sum(1), 1e-9)[None], 0.0, 1.0)
                dseg = np.linalg.norm(a[None] + t[..., None] * ab[None] - path[:, None], axis=2).min(1)
                d = np.minimum(d, dseg)                       # wall segments

        # v2: required clearance ramps from the car's own clearance (0.5 m per m driven), so a
        # car that is close to a cone can still get a path that leads away from it
        d_car = float(cKDTree(P).query(car)[0])
        if len(P) >= 3 and segs:
            tc = np.clip(((car - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0.0, 1.0)
            d_car = min(d_car, float(np.linalg.norm(a + tc[:, None] * ab - car, axis=1).min()))
        arc = np.linalg.norm(path[0] - car) + np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        need = np.minimum(self.pcd_clearance, max(d_car - 0.05, 0.0) + 0.5 * arc)
        hit = np.nonzero(d < need)[0]

        return path[:hit[0]] if len(hit) else path

    def _set_pcd_plot(self, points_window, classes_window, path, blue_count, yellow_count):
        """
        Live plot in PCD mode: edges = traced walls (+ predicted
        and virtual cones chained along them), candidates =
        PCD midpoints, path = PCD path.
        """

        d = getattr(self.pcd, "debug", {}) or {}
        pts, edges = [], []

        def add_chain(*arrays):
            arr = [np.asarray(a, float).reshape(-1, 2) for a in arrays if a is not None and len(a)]
            if not arr:
                return
            P = np.vstack(arr)
            base = len(pts)
            pts.extend((float(x), float(y)) for x, y in P)
            for k in range(len(P) - 1):
                if np.linalg.norm(P[k + 1] - P[k]) < 7.0:
                    edges.append((base + k, base + k + 1))

        add_chain(d.get("left"), d.get("left_pred"))
        add_chain(d.get("right"), d.get("right_pred"))
        add_chain(d.get("virt_yellow"))
        add_chain(d.get("virt_blue"))

        mids = np.asarray(d.get("mids", np.zeros((0, 2))), float).reshape(-1, 2)

        side = "?"
        if self.pcd_blue_left is not None:
            side = "BLUE" if self.pcd_blue_left else "YELLOW"

        with self.plot_lock:
            self.plot_points_window = points_window
            self.plot_classes_window = classes_window
            self.plot_edges = edges
            self.plot_edge_points = pts
            self.plot_candidates = [(float(x), float(y)) for x, y in mids]
            self.plot_path = path
            self.plot_virtual_cone = None
            self.plot_mode = "PCD"
            self.plot_selected_side = side
            self.plot_blue_count = blue_count
            self.plot_yellow_count = yellow_count

    # ========================================================
    # DEBUG TOPICS (NEW)
    # ========================================================

    def _tick(self):

        self._tick_core()

        if self.dbg_enable:
            try:
                self._publish_debug()
            except Exception as exc:
                self.get_logger().warn(f"debug publish failed: {exc}", throttle_duration_sec=5.0)

    def _publish_debug(self):

        with self.plot_lock:
            mode = self.plot_mode
            edges = list(self.plot_edges)
            edge_points = list(self.plot_edge_points)
            candidates = list(self.plot_candidates)

        if mode == "WAITING":
            return

        left = None
        if self.pcd_blue_left is not None:
            left = "BLUE" if self.pcd_blue_left else "YELLOW"

        dbg = getattr(self.pcd, "debug", {}) or {}
        route = self._route if self._route is not None else np.zeros((0, 2))
        kmax = 0.0
        if len(route) >= 3:
            sg = np.linalg.norm(np.diff(route, axis=0), axis=1)
            keep = np.r_[True, sg > 0.05]
            r2 = route[keep]
            if len(r2) >= 3:
                sg = np.linalg.norm(np.diff(r2, axis=0), axis=1)
                hd = np.unwrap(np.arctan2(np.diff(r2[:, 1]), np.diff(r2[:, 0])))
                kmax = float((np.abs(np.diff(hd)) / np.maximum(0.5 * (sg[:-1] + sg[1:]), 1e-3)).max())
        info = dict(
            mode=mode,
            reason=("" if mode == "PCD" else self.pcd_last_reason),
            width=round(float(self.pcd.width), 3),
            left=left,
            route=self._route_state,                      # v2: new / follow / hold n/N / switch / no route
            route_jump_m=round(float(self._route_jump), 2),
            confirmed_m=round(float(dbg.get("confirmed_m", 0.0)), 2) if mode == "PCD" else None,
            kappa_max=round(kmax, 3),                     # of the published route [1/m]
            r_min_m=(round(1.0 / kmax, 2) if kmax > 1e-6 else None),
            tight_for_20deg=bool(kmax > 1.0 / 4.34),      # tighter than R 4.34 m (controller STEER_MAX 20 deg)
            tight_for_full_lock=bool(kmax > 1.0 / 2.76),  # tighter than R 2.76 m (EUFS lock 0.52 rad)
            unknown_used=int(len(dbg.get("left_unknown", [])) + len(dbg.get("right_unknown", []))) if mode == "PCD" else 0,
            min_curvature=dbg.get("min_curvature") if mode == "PCD" else None,
        )
        m = String()
        m.data = json.dumps(info)
        self.mode_pub.publish(m)

        stamp = self.get_clock().now().to_msg()
        arr = MarkerArray()

        clear = Marker()
        clear.header.frame_id = "map"
        clear.header.stamp = stamp
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        def mk(ns, mid, mtype, pts, rgba, scale):
            pts = np.asarray(pts, dtype=float).reshape(-1, 2)
            if len(pts) == 0:
                return
            k = Marker()
            k.header.frame_id = "map"
            k.header.stamp = stamp
            k.ns = ns
            k.id = mid
            k.type = mtype
            k.action = Marker.ADD
            k.pose.orientation.w = 1.0
            k.scale.x = scale
            k.scale.y = scale
            k.scale.z = scale
            k.color.r, k.color.g, k.color.b, k.color.a = rgba
            for x, y in pts:
                q = Point()
                q.x, q.y, q.z = float(x), float(y), 0.0
                k.points.append(q)
            arr.markers.append(k)

        BLUE = (0.18, 0.42, 1.0, 1.0)
        YEL = (1.0, 0.83, 0.0, 1.0)

        if mode == "PCD":
            d = getattr(self.pcd, "debug", {}) or {}
            lcol, rcol = (BLUE, YEL) if self.pcd_blue_left else (YEL, BLUE)
            mk("left_wall", 0, Marker.LINE_STRIP, d.get("left", []), lcol, 0.12)
            mk("right_wall", 1, Marker.LINE_STRIP, d.get("right", []), rcol, 0.12)
            pred = [a for a in (d.get("left_pred"), d.get("right_pred")) if a is not None and len(a)]
            if pred:
                mk("predicted", 2, Marker.SPHERE_LIST, np.vstack(pred), (1.0, 0.3, 1.0, 1.0), 0.45)
            virt = [a for a in (d.get("virt_blue"), d.get("virt_yellow")) if a is not None and len(a)]
            if virt:
                mk("virtual", 3, Marker.SPHERE_LIST, np.vstack(virt), (1.0, 1.0, 1.0, 0.9), 0.35)
            mk("midpoints", 4, Marker.SPHERE_LIST, d.get("mids", []), (0.24, 1.0, 0.43, 0.9), 0.22)
            unk = [x for x in (d.get("left_unknown"), d.get("right_unknown")) if x is not None and len(x)]
            if unk:
                mk("unknown_used", 6, Marker.SPHERE_LIST, np.vstack(unk), (0.6, 0.6, 0.6, 1.0), 0.35)
        else:
            seg = []
            for i, j in edges:
                if i < len(edge_points) and j < len(edge_points):
                    seg.append(edge_points[i]); seg.append(edge_points[j])
            mk("edges", 5, Marker.LINE_LIST, seg, (0.85, 0.85, 0.85, 0.6), 0.05)
            mk("midpoints", 4, Marker.SPHERE_LIST, candidates, (0.24, 1.0, 0.43, 0.9), 0.22)

        self.marker_pub.publish(arr)

    def _tick_core(self):
        self._cur_source = "TEAM"

        # ====================================================
        # GET CURRENT STATE
        # ====================================================

        with self.data_lock:

            car_x = self.car_x
            car_y = self.car_y
            car_yaw = self.car_yaw

            blue_global = list(
                self.blue_global
            )

            yellow_global = list(
                self.yellow_global
            )

            orange_global = list(
                self.orange_global
            )

            big_global = list(
                self.big_global
            )

        # ----------------------------------------------------
        # No odometry yet
        # ----------------------------------------------------

        if (
            car_x is None
            or
            car_y is None
            or
            car_yaw is None
        ):

            return

        # ====================================================
        # KEEP LATEST DATA
        # ====================================================

        with self.data_lock:

            self.blue_global = (
                blue_global
            )

            self.yellow_global = (
                yellow_global
            )

            self.orange_global = (
                orange_global
            )

            self.big_global = (
                big_global
            )

        # ====================================================
        # STEP 1:
        # COMPUTE FOV
        # ====================================================

        (
            points_window,
            classes_window
        ) = self._compute_fov_points(
            car_x,
            car_y,
            car_yaw
        )

        # ====================================================
        # COUNTS
        # ====================================================

        blue_count = sum(
            1
            for cls
            in classes_window
            if cls == "blue"
        )

        yellow_count = sum(
            1
            for cls
            in classes_window
            if cls == "yellow"
        )

        # ====================================================
        # VIRTUAL CONE
        #
        # Only visualized/used in fallback.
        # ====================================================

        virtual_cone = None

        # ====================================================
        # TOO FEW TOTAL POINTS
        # ====================================================

        if len(
            points_window
        ) < 2:

            with self.plot_lock:

                self.plot_points_window = (
                    points_window
                )

                self.plot_classes_window = (
                    classes_window
                )

                self.plot_edges = []
                self.plot_edge_points = []
                self.plot_candidates = []
                self.plot_path = []

                self.plot_virtual_cone = None

                self.plot_mode = (
                    "INSUFFICIENT FOV POINTS"
                )

                self.plot_selected_side = (
                    "None"
                )

                self.plot_blue_count = (
                    blue_count
                )

                self.plot_yellow_count = (
                    yellow_count
                )

            self._publish_path(
                car_x,
                car_y,
                []
            )

            self._update_live_plot(
                car_x,
                car_y,
                car_yaw
            )

            return

        # ====================================================
        # STEP 2:
        # SELECT MODE
        # ====================================================

        # ----------------------------------------------------
        # STEP 1b (NEW): PCD first
        #
        # If PCD gives a long-enough, wall-checked path, it is
        # published and the original logic is skipped.
        # Otherwise fall through to the original NORMAL /
        # FALLBACK logic below (unchanged).
        # ----------------------------------------------------

        if self.pcd_enable:

            pcd_path = self._try_pcd(
                car_x,
                car_y,
                car_yaw,
                blue_global,
                yellow_global,
                classes_window,
            )

            self._log_mode("PCD" if pcd_path is not None else "ORIGINAL")

            if pcd_path is not None:
                self._cur_source = "PCD"

                self._publish_path(
                    car_x,
                    car_y,
                    pcd_path
                )

                self._set_pcd_plot(
                    points_window,
                    classes_window,
                    pcd_path,
                    blue_count,
                    yellow_count,
                )

                self._update_live_plot(
                    car_x,
                    car_y,
                    car_yaw
                )

                return

        fallback_mode = (
            blue_count <= 1
            or
            yellow_count <= 1
        )

        # ====================================================
        # FALLBACK MODE
        # ====================================================

        if fallback_mode:

            (
                edges,
                working_points,
                working_classes,
                selected_side
            ) = (
                self._build_fallback_edges(
                    points_window,
                    classes_window,
                    car_x,
                    car_y,
                    car_yaw
                )
            )

            # ------------------------------------------------
            # Virtual cone for visualization
            # ------------------------------------------------

            if selected_side is not None:

                virtual_cone = (
                    self._get_virtual_cone(
                        car_x,
                        car_y,
                        car_yaw
                    )
                )

            # ------------------------------------------------
            # Edge midpoint candidates
            # ------------------------------------------------

            centroids_raw = []

            for i, j in edges:

                x1, y1 = (
                    working_points[i]
                )

                x2, y2 = (
                    working_points[j]
                )

                mx = (
                    0.5 *
                    (
                        x1 + x2
                    )
                )

                my = (
                    0.5 *
                    (
                        y1 + y2
                    )
                )

                centroids_raw.append(
                    (
                        mx,
                        my,
                        False
                    )
                )

            # ------------------------------------------------
            # Filtering
            # ------------------------------------------------

            centroids = (
                self._filter_centroids_min_spacing(
                    centroids_raw,
                    car_x,
                    car_y,
                    min_dist=(
                        self.min_spacing
                    )
                )
            )

            candidate_points = [
                (
                    x,
                    y
                )
                for x, y, _
                in centroids
            ]

            # ------------------------------------------------
            # Greedy path
            # ------------------------------------------------

            path_points = (
                self._build_greedy_path(
                    car_x,
                    car_y,
                    candidate_points
                )
            )

            # ------------------------------------------------
            # Publish
            # ------------------------------------------------

            self._publish_path(
                car_x,
                car_y,
                path_points
            )

            # ------------------------------------------------
            # Visualization state
            # ------------------------------------------------

            with self.plot_lock:

                self.plot_points_window = (
                    points_window
                )

                self.plot_classes_window = (
                    classes_window
                )

                self.plot_edges = (
                    edges
                )

                self.plot_edge_points = (
                    working_points
                )

                self.plot_candidates = (
                    candidate_points
                )

                self.plot_path = (
                    path_points
                )

                self.plot_virtual_cone = (
                    virtual_cone
                )

                self.plot_mode = (
                    "FALLBACK"
                )

                self.plot_selected_side = (
                    selected_side
                    if selected_side
                    is not None
                    else "None"
                )

                self.plot_blue_count = (
                    blue_count
                )

                self.plot_yellow_count = (
                    yellow_count
                )

        # ====================================================
        # NORMAL MODE
        # ====================================================

        else:

            # ------------------------------------------------
            # Normal Delaunay
            # ------------------------------------------------

            edges = (
                self._build_edges(
                    points_window,
                    classes_window
                )
            )

            # ------------------------------------------------
            # Big-orange special
            # ------------------------------------------------

            big_special = (
                self._big_cone_candidate(
                    points_window,
                    classes_window
                )
            )

            # ------------------------------------------------
            # Midpoint candidates
            # ------------------------------------------------

            centroids_raw = []

            for i, j in edges:

                x1, y1 = (
                    points_window[i]
                )

                x2, y2 = (
                    points_window[j]
                )

                ci = (
                    classes_window[i]
                )

                cj = (
                    classes_window[j]
                )

                # Big-big handled separately
                if (
                    ci == "big"
                    and
                    cj == "big"
                ):

                    continue

                mx = (
                    0.5 *
                    (
                        x1 + x2
                    )
                )

                my = (
                    0.5 *
                    (
                        y1 + y2
                    )
                )

                centroids_raw.append(
                    (
                        mx,
                        my,
                        False
                    )
                )

            # ------------------------------------------------
            # Add big-orange candidate
            # ------------------------------------------------

            for bx, by, _ in (
                big_special
            ):

                centroids_raw.append(
                    (
                        bx,
                        by,
                        True
                    )
                )

            # ------------------------------------------------
            # Candidate filtering
            # ------------------------------------------------

            centroids = (
                self._filter_centroids_min_spacing(
                    centroids_raw,
                    car_x,
                    car_y,
                    min_dist=(
                        self.min_spacing
                    )
                )
            )

            candidate_points = [
                (
                    x,
                    y
                )
                for x, y, _
                in centroids
            ]

            # ------------------------------------------------
            # Greedy path
            # ------------------------------------------------

            path_points = (
                self._build_greedy_path(
                    car_x,
                    car_y,
                    candidate_points
                )
            )

            # ------------------------------------------------
            # Publish
            # ------------------------------------------------

            self._publish_path(
                car_x,
                car_y,
                path_points
            )

            # ------------------------------------------------
            # Visualization state
            # ------------------------------------------------

            with self.plot_lock:

                self.plot_points_window = (
                    points_window
                )

                self.plot_classes_window = (
                    classes_window
                )

                self.plot_edges = (
                    edges
                )

                self.plot_edge_points = (
                    points_window
                )

                self.plot_candidates = (
                    candidate_points
                )

                self.plot_path = (
                    path_points
                )

                self.plot_virtual_cone = (
                    None
                )

                self.plot_mode = (
                    "NORMAL"
                )

                self.plot_selected_side = (
                    "None"
                )

                self.plot_blue_count = (
                    blue_count
                )

                self.plot_yellow_count = (
                    yellow_count
                )

        # ====================================================
        # UPDATE VISUALIZATION
        # ====================================================

        self._update_live_plot(
            car_x,
            car_y,
            car_yaw
        )


# ============================================================
# MAIN
# ============================================================

def main():

    rclpy.init()

    node = (
        SlamPathSectorVisualizer()
    )

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    finally:

        if node.fig is not None:
            plt.close(node.fig)

        node.destroy_node()

        rclpy.shutdown()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()