#!/usr/bin/env python3
"""
pp_skidpad_tels.py  -  skidpad path planner: fit the known layout, then follow it

    ros2 run pathplanning pp_skidpad_tels
    ros2 launch bringup stack_skidpad.launch.py      (whole stack with this planner)

Same inputs and output as pp_tls / pp_tels_new (/slam/odom_raw + /slam/map_cones in,
/path_points out in "map", car pose first, then points every 0.5 m), so the controller
needs no change. Run it INSTEAD of pp_tls / pp_tels_new on the skidpad only.

Why a separate planner
    The skidpad layout is fixed by the rules (FS Rules 2026, D4 / DV skidpad):
      * two circles, centres 18.25 m apart
      * inner circle diameter 15.25 m (R 7.625 m), track 3 m wide -> outer R 10.625 m
      * driving line = middle of the track, R = 9.125 m, both circles tangent to the
        entry/exit line at the crossing point (the timing line)
      * procedure: enter, 2 laps of the RIGHT circle (clockwise), 2 laps of the LEFT circle
        (counter-clockwise), leave along the exit lane
    Cone-by-cone planning (Delaunay/PCD) has to rediscover this every tick and is ambiguous
    at the crossing, where both circles overlap. Here only three numbers are unknown: where
    the layout sits in the map and how it is rotated (x, y, yaw). They are fitted to every
    cone seen so far; the path is then the exact rule geometry.

How it works (every tick)
    1. Memory: every cone of /slam/map_cones is merged into a map (0.5 m merge radius); the
       SLAM map is local, the fit uses everything seen.
    2. Prior: the layout's crossing point is assumed `skidpad.start_to_centre_m` (15 m) ahead
       of the car's first pose, along its heading (the car starts in the entry lane).
    3. Fit (x, y, yaw), colour-agnostic: each blue / yellow / unknown cone should lie on one
       of the four circles (inner / outer of the right and left circle); each orange cone on
       one of the two lane lines (|lateral| = lane half width) or near the crossing. Robust
       least squares (soft L1, 0.3 m), started from a coarse grid around the prior until the
       fit locks, then from the current estimate.
    4. Orientation and colour: the figure of eight looks the same rotated by 180 deg, so the
       fit must put the car in the entry lane heading along it. The colour convention (blue
       left / yellow right, or swapped as in some SLAM maps) is then read off the fitted pose
       and used as a SOFT association term (a boundary of the wrong colour costs 0.6 m).
    5. Lock, only when the pose is actually determined: >= 12 inliers, RMS <= 0.25 m, >= 4
       inliers on EACH circle, >= 60 deg angular coverage on each circle, 1-sigma uncertainty
       from rms^2 (J^T J)^-1 <= 0.05 m and 0.3 deg, and 3 consecutive fits agreeing. Until
       then the path is straight along the car's heading. After locking, a new fit is blended
       in (EMA) only if it is as good and moves the layout by < 0.5 m / 3 deg; dragged or
       false cones cannot pull it away.
    6. Progress: the reference (entry -> right x2 -> left x2 -> exit) is one polyline in the
       layout frame. The car's arc length s only moves forward and is searched in a window
       [s - 1 m, s + 6 m], so the five passes through the crossing are never confused.
    7. Output: car pose, then the reference from the car's projection for
       `skidpad.horizon_m` (20 m) at 0.5 m spacing, transformed into "map".
    /pp/mode (JSON): mode SKIDPAD, phase, lap, fit RMS, cones used, locked, s, pose
    uncertainty, angular coverage, colour convention, and which lock checks are still failing.
    /pp/debug_markers: the fitted circles (ns "edges") for the bird view / RViz.

Parameters (defaults = FS rules)
    skidpad.path_radius_m 9.125, skidpad.inner_radius_m 7.625, skidpad.outer_radius_m 10.625,
    skidpad.centre_distance_m 18.25, skidpad.lane_half_width_m 1.5, skidpad.laps_per_side 2,
    skidpad.first_side "right", skidpad.start_to_centre_m 15.0, skidpad.entry_length_m 30,
    skidpad.exit_length_m 25, skidpad.horizon_m 20, skidpad.lock_min_cones 12,
    skidpad.lock_rms_m 0.25, skidpad.lock_min_per_circle 4, skidpad.lock_min_arc_deg 60,
    skidpad.lock_sigma_xy_m 0.05, skidpad.lock_sigma_deg 0.3, skidpad.lock_stable_fits 3,
    skidpad.colour_mode auto (normal / swapped / off), skidpad.colour_penalty_m 0.6,
    skidpad.gate_half_width_m 1.78
"""
from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Point
from eufs_msgs.msg import ConeArrayWithCovariance
from std_msgs.msg import String
try:
    from visualization_msgs.msg import Marker, MarkerArray
    _HAS_VIZ = True
except ImportError:                                    # pragma: no cover
    _HAS_VIZ = False


# ============================================================ pure geometry (no ROS)
@dataclass
class SkidpadConfig:
    path_radius: float = 9.125
    inner_radius: float = 7.625
    outer_radius: float = 10.625
    centre_distance: float = 18.25
    lane_half_width: float = 1.5        # entry/exit lane 3 m wide (EUFS places its orange cones at 1.64)
    gate_half_width: float = 1.78       # timing-line (big orange) cones, lateral offset at the crossing
    laps_per_side: int = 2
    first_side: str = "right"           # which circle is driven first
    start_to_centre: float = 15.0       # car's first pose -> crossing point [m]
    entry_length: float = 30.0
    exit_length: float = 25.0
    horizon: float = 20.0
    step: float = 0.5
    merge_radius: float = 0.5
    lock_min_cones: int = 12
    lock_rms: float = 0.25
    lock_min_per_circle: int = 4        # inliers on EACH circle
    lock_min_arc_deg: float = 60.0      # angular coverage of the inliers on EACH circle
    lock_sigma_xy: float = 0.05         # [m] 1-sigma position uncertainty from (J^T J)^-1
    lock_sigma_deg: float = 0.3         # [deg] 1-sigma yaw uncertainty
    lock_stable_fits: int = 3           # consecutive fits agreeing within 0.05 m / 0.3 deg
    provisional_sigma_xy: float = 0.15  # [m] weaker bound to use an unlocked fit for the path
    provisional_sigma_deg: float = 1.0
    colour_mode: str = "auto"           # auto | normal | swapped | off
    colour_penalty: float = 0.6         # [m] added to a boundary of the wrong colour (soft)
    fit_radius: float = 35.0            # only cones within this distance of the crossing


def _rot(th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]])


class SkidpadPlanner:
    """Layout frame: origin = crossing point, +x = driving direction entry -> exit, +y = left.
    The first circle is centred at (0, -h) when it is the right one (clockwise), h = D / 2."""

    def __init__(self, cfg: SkidpadConfig | None = None):
        self.cfg = c = cfg or SkidpadConfig()
        self.h = c.centre_distance / 2.0
        sgn = -1.0 if c.first_side == "right" else 1.0      # first circle's centre side
        self.centres = np.array([[0.0, sgn * self.h], [0.0, -sgn * self.h]])
        self.ref, self.ref_s, self.phase_bounds = self._build_reference()
        # memory: position, kind (0 unknown-colour circle cone, 1 orange, 2 blue, 3 yellow) and
        # colour votes (blue, yellow) so that one wrong colour does not flip a cone
        self.mem = np.zeros((0, 2)); self.mem_kind = np.zeros(0, int); self.mem_votes = np.zeros((0, 2))
        # expected colour per boundary [first inner, first outer, second inner, second outer] for
        # blue-left / yellow-right: on a clockwise circle the inner edge is on the car's right
        exp = []
        for ctr in self.centres:
            cw = ctr[1] < 0
            exp += [3, 2] if cw else [2, 3]
        self.exp_normal = np.array(exp)
        self.colour_conv = {"normal": 1, "swapped": -1}.get(self.cfg.colour_mode, 0)
        self.colour_frac = float("nan")
        self.history = []                # recent consistent fits (for the stability test)
        self.lock_checks = {}
        self.sigma = (float("nan"), float("nan"))
        self.arcs = (0.0, 0.0)
        self.pose = None                 # (tx, ty, th): layout -> map
        self.locked = False
        self.rms = float("nan")
        self.n_used = 0
        self.s = None
        self.prior = None
        self.consistent = False
        self.n_disagree = 0

    # ---------------------------------------------------------------- reference
    def _build_reference(self):
        c, R, ds = self.cfg, self.cfg.path_radius, 0.05
        parts, names = [], []
        x = np.arange(-c.entry_length, 0.0, ds)
        parts.append(np.c_[x, np.zeros_like(x)]); names.append("entry")
        for k, (cx, cy) in enumerate(self.centres):
            turn = -1.0 if cy < 0 else 1.0               # centre on the right -> clockwise
            phi = np.arange(0.0, 2 * math.pi * c.laps_per_side, ds / R)
            # start at the crossing heading +x: point = centre + R * (sin(phi), -turn*cos(phi)) * ...
            px = R * np.sin(phi)
            py = cy - turn * R * np.cos(phi)
            for lap in range(c.laps_per_side):
                m = (phi >= 2 * math.pi * lap) & (phi < 2 * math.pi * (lap + 1))
                parts.append(np.c_[px[m], py[m]])
                side = "right" if cy < 0 else "left"
                names.append(f"{side} {lap + 1}")
        x = np.arange(0.0, c.exit_length + ds, ds)
        parts.append(np.c_[x, np.zeros_like(x)]); names.append("exit")
        ref = np.vstack(parts)
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(ref, axis=0), axis=1))]
        bounds, i = [], 0
        for p, n in zip(parts, names):
            bounds.append((s[i], n)); i += len(p)
        return ref, s, bounds

    def phase(self, s):
        name = self.phase_bounds[0][1]
        if s is None:
            return name
        for s0, n in self.phase_bounds:
            if s >= s0 - 1e-6:
                name = n
        return name

    # ---------------------------------------------------------------- memory
    def add_cones(self, blue=(), yellow=(), unknown=(), orange=()):
        for pts, kind in ((blue, 2), (yellow, 3), (unknown, 0), (orange, 1)):
            P = np.asarray(pts, float).reshape(-1, 2)
            if not len(P):
                continue
            if len(self.mem):
                d, i = cKDTree(self.mem).query(P)
                hit = d < self.cfg.merge_radius
                for j in np.unique(i[hit]):
                    self.mem[j] = 0.8 * self.mem[j] + 0.2 * P[hit & (i == j)].mean(0)
                    if kind in (2, 3):
                        self.mem_votes[j, kind - 2] += 1
                    elif kind == 1:
                        self.mem_kind[j] = 1
                P = P[~hit]
            if len(P):
                self.mem = np.vstack([self.mem, P])
                self.mem_kind = np.r_[self.mem_kind, np.full(len(P), kind)]
                v = np.zeros((len(P), 2))
                if kind in (2, 3):
                    v[:, kind - 2] = 1
                self.mem_votes = np.vstack([self.mem_votes, v])
        circ = self.mem_kind != 1
        col = np.where(self.mem_votes[:, 0] > self.mem_votes[:, 1], 2,
                       np.where(self.mem_votes[:, 1] > self.mem_votes[:, 0], 3, 0))
        self.mem_kind = np.where(circ, col, 1)

    # ---------------------------------------------------------------- fit
    def _boundary_dist(self, Q):
        """|distance| of layout-frame points to the 4 boundaries [1st inner, 1st outer, 2nd inner, 2nd outer]."""
        d1 = np.linalg.norm(Q - self.centres[0], axis=1)
        d2 = np.linalg.norm(Q - self.centres[1], axis=1)
        ri, ro = self.cfg.inner_radius, self.cfg.outer_radius
        return np.abs(np.c_[d1 - ri, d1 - ro, d2 - ri, d2 - ro])

    def _residuals(self, params, P, K, use_colour=True):
        tx, ty, th = params
        Q = (P - [tx, ty]) @ _rot(th)                     # map -> layout (R^T applied on the right)
        r = np.empty(len(Q))
        circ = K != 1
        if circ.any():
            B = self._boundary_dist(Q[circ])
            if use_colour and self.colour_conv != 0:
                # soft colour association: a boundary of the wrong colour costs colour_penalty
                k = K[circ]
                exp = self.exp_normal if self.colour_conv > 0 else 5 - self.exp_normal
                wrong = (k[:, None] >= 2) & (k[:, None] != exp[None, :])
                B = B + self.cfg.colour_penalty * wrong
            r[circ] = B.min(axis=1)
        orn = ~circ
        if orn.any():
            q = Q[orn]
            lane = np.abs(np.abs(q[:, 1]) - self.cfg.lane_half_width)
            gate = np.abs(np.abs(q[:, 1]) - self.cfg.gate_half_width)
            r[orn] = np.where(np.abs(q[:, 0]) < 4.0, np.minimum(lane, gate), lane)
        return r

    def _car_agrees(self, pose, car_xy, car_yaw):
        """Is the car where this layout pose says it can be? Before any progress: in the entry
        lane, behind the crossing, heading along it. Once a provisional pose is in use: within
        3 m of the reference near its current arc length, heading along the reference."""
        q = (np.asarray(car_xy, float) - pose[:2]) @ _rot(pose[2])
        if self.s is None:
            dh = abs((car_yaw - pose[2] + math.pi) % (2 * math.pi) - math.pi)
            return -self.cfg.entry_length <= q[0] <= -1.0 and abs(q[1]) <= 3.0 and dh <= math.radians(35)
        lo, hi = np.searchsorted(self.ref_s, [self.s - 3.0, self.s + 8.0])
        hi = max(hi, lo + 2)
        d = np.linalg.norm(self.ref[lo:hi] - q, axis=1)
        i = lo + int(np.argmin(d))
        if d.min() > 3.0:
            return False
        j = min(i + 1, len(self.ref) - 1); i0 = j - 1
        tang = math.atan2(self.ref[j, 1] - self.ref[i0, 1], self.ref[j, 0] - self.ref[i0, 0]) + pose[2]
        return abs((car_yaw - tang + math.pi) % (2 * math.pi) - math.pi) <= math.radians(35)

    def _colour_agreement(self, pose, P, K):
        """Share of coloured inlier cones whose colour matches blue-left/yellow-right at this pose."""
        col = K >= 2
        if col.sum() < 6:
            return float("nan")
        Q = (P[col] - pose[:2]) @ _rot(pose[2])
        B = self._boundary_dist(Q)
        b = B.argmin(axis=1)
        inl = B.min(axis=1) < 0.4
        if inl.sum() < 6:
            return float("nan")
        return float(np.mean(K[col][inl] == self.exp_normal[b[inl]]))

    def _quality(self, pose, P, K):
        """Lock checks: inliers and angular coverage per circle, and the pose covariance
        rms^2 (J^T J)^-1 from a numerical Jacobian of the inlier residuals."""
        c = self.cfg
        circ = K != 1
        Pc, Kc = P[circ], K[circ]
        r = self._residuals(pose, Pc, Kc)
        inl = r < 0.6
        Q = (Pc[inl] - pose[:2]) @ _rot(pose[2])
        B = self._boundary_dist(Q)
        which = B.argmin(axis=1) // 2                     # 0 first circle, 1 second circle
        n_c, arcs = [], []
        for k in (0, 1):
            q = Q[which == k] - self.centres[k]
            n_c.append(int(len(q)))
            if len(q) < 2:
                arcs.append(0.0); continue
            a = np.sort(np.degrees(np.arctan2(q[:, 1], q[:, 0])) % 360.0)
            gaps = np.diff(np.r_[a, a[0] + 360.0])
            arcs.append(float(360.0 - gaps.max()))
        rms = float(np.sqrt(np.mean(r[inl] ** 2))) if inl.any() else float("inf")
        sig_xy, sig_th = float("inf"), float("inf")
        if inl.sum() >= 3:
            J = np.empty((int(inl.sum()), 3))
            for j, e in enumerate((1e-4, 1e-4, 1e-5)):
                dp = np.zeros(3); dp[j] = e
                J[:, j] = (self._residuals(pose + dp, Pc[inl], Kc[inl]) -
                           self._residuals(pose - dp, Pc[inl], Kc[inl])) / (2 * e)
            H = J.T @ J
            if np.linalg.cond(H) < 1e10:
                C = max(rms, 0.05) ** 2 * np.linalg.inv(H)
                sig_xy = float(np.sqrt(np.linalg.eigvalsh(C[:2, :2]).max()))
                sig_th = float(np.degrees(np.sqrt(C[2, 2])))
        return dict(n_in=int(inl.sum()), rms=rms, n_circle=n_c, arcs=arcs, sigma_xy=sig_xy, sigma_deg=sig_th)

    def fit(self, car_xy, car_yaw):
        c = self.cfg
        if not self.locked and not self.consistent:
            # the prior follows the car until a fit agrees with it (the car waits at the start;
            # a stale first odometry message must not fix the prior). Afterwards the search is
            # centred on the provisional pose, not on a prior that moves with the car.
            o = np.asarray(car_xy) + c.start_to_centre * np.array([math.cos(car_yaw), math.sin(car_yaw)])
            self.prior = np.array([o[0], o[1], car_yaw])
        guess = self.pose if (self.locked or self.consistent) else self.prior
        if not len(self.mem):
            self.pose = self.pose if self.locked else self.prior.copy()
            return
        sel = np.linalg.norm(self.mem - guess[:2], axis=1) < c.fit_radius
        P, K = self.mem[sel], self.mem_kind[sel]
        if (K != 1).sum() < 8:                            # too few cones: keep the prior
            self.pose = self.pose if self.locked else self.prior.copy()
            return
        loss = lambda p: np.sum(np.log1p((self._residuals(p, P, K) / 0.3) ** 2))
        if not self.locked:
            # coarse grid around the prior: along +-6 m, lateral +-2 m, yaw +-12 deg
            best, bp = np.inf, guess
            ctr = guess
            ux = np.array([math.cos(ctr[2]), math.sin(ctr[2])])
            uy = np.array([-ux[1], ux[0]])
            if self.consistent:
                best, bp = loss(ctr), ctr
            for da in np.arange(-6.0, 6.01, 1.0):
                for dl in np.arange(-2.0, 2.01, 1.0):
                    for dt in np.radians(np.arange(-12, 12.1, 3.0)):
                        p = np.r_[ctr[:2] + da * ux + dl * uy, ctr[2] + dt]
                        v = loss(p)
                        if v < best:
                            best, bp = v, p
            guess = bp
        res = least_squares(self._residuals, guess, args=(P, K), loss="soft_l1", f_scale=0.3)
        r = self._residuals(res.x, P, K)
        rc = r[K != 1]
        inl = rc < 0.6
        rms = float(np.sqrt(np.mean(rc[inl] ** 2))) if inl.any() else float("inf")
        n_in = int(inl.sum())
        if not self.locked:
            # the figure of eight looks the same rotated by 180 deg; the car decides: it must be
            # in the entry lane (behind the crossing, within 3 m of the line) heading along +x
            best = None
            for th in (res.x[2], res.x[2] + math.pi):
                cand = np.r_[res.x[:2], (th + math.pi) % (2 * math.pi) - math.pi]
                if self._car_agrees(cand, car_xy, car_yaw):
                    best = cand
            if best is None:
                self.n_disagree += 1
                if self.consistent and self.n_disagree > 20:
                    self.consistent, self.s = False, None  # provisional pose lost: start again
                if not self.consistent:
                    self.pose = self.prior.copy()         # fit disagrees with the car: keep waiting
                    self.history = []
                self.rms, self.n_used = rms, n_in         # (a provisional pose is kept)
                return
            self.n_disagree = 0
            if self.colour_conv == 0 or c.colour_mode == "auto":
                # colour convention from the orientation the car chose (blue may be on the right
                # in this stack's SLAM map); used as a soft term only once it is clear
                f = self._colour_agreement(best, P, K)
                self.colour_frac = f
                if c.colour_mode == "auto":
                    self.colour_conv = 1 if f >= 0.8 else (-1 if f <= 0.2 else 0) if f == f else 0
                if self.colour_conv != 0:                 # refine once more with the colour term
                    best = least_squares(self._residuals, best, args=(P, K), loss="soft_l1", f_scale=0.3).x
            qd = self._quality(best, P, K)
            if not (qd["n_in"] >= 8 and qd["sigma_xy"] <= c.provisional_sigma_xy
                    and qd["sigma_deg"] <= c.provisional_sigma_deg):
                # too weakly determined even for provisional use: keep what we had
                if not self.consistent:
                    self.pose = self.prior.copy()
                self.rms, self.n_used = qd["rms"], qd["n_in"]
                return
            self.consistent = True
            self.pose = best
            self.rms, self.n_used = qd["rms"], qd["n_in"]
            self.sigma, self.arcs = (qd["sigma_xy"], qd["sigma_deg"]), tuple(qd["arcs"])
            self.history = (self.history + [best.copy()])[-c.lock_stable_fits:]
            stable = len(self.history) >= c.lock_stable_fits and all(
                np.linalg.norm(h[:2] - best[:2]) <= 0.05 and
                abs((h[2] - best[2] + math.pi) % (2 * math.pi) - math.pi) <= math.radians(0.3)
                for h in self.history)
            self.lock_checks = dict(
                cones=qd["n_in"] >= c.lock_min_cones,
                rms=qd["rms"] <= c.lock_rms,
                both_circles=min(qd["n_circle"]) >= c.lock_min_per_circle,
                coverage=min(qd["arcs"]) >= c.lock_min_arc_deg,
                observable=qd["sigma_xy"] <= c.lock_sigma_xy and qd["sigma_deg"] <= c.lock_sigma_deg,
                stable=bool(stable))
            if all(self.lock_checks.values()):
                self.locked = True
        else:
            dpos = np.linalg.norm(res.x[:2] - self.pose[:2])
            dth = abs((res.x[2] - self.pose[2] + math.pi) % (2 * math.pi) - math.pi)
            if rms <= max(c.lock_rms, 1.2 * self.rms) and dpos < 0.5 and dth < math.radians(3):
                a = 0.2
                th = self.pose[2] + a * ((res.x[2] - self.pose[2] + math.pi) % (2 * math.pi) - math.pi)
                self.pose = np.r_[(1 - a) * self.pose[:2] + a * res.x[:2], th]
                self.rms, self.n_used = 0.8 * self.rms + 0.2 * rms, n_in

    # ---------------------------------------------------------------- path
    def to_map(self, Q):
        tx, ty, th = self.pose
        return np.asarray(Q) @ _rot(th).T + [tx, ty]

    def to_layout(self, P):
        tx, ty, th = self.pose
        return (np.asarray(P) - [tx, ty]) @ _rot(th)

    def plan(self, car_xy, car_yaw):
        """Returns the path (N,2) in map frame starting after the car, or empty."""
        self.fit(car_xy, car_yaw)
        if not self.locked and not self.consistent:
            # no fit that agrees with the car yet: straight ahead along its heading (entry lane)
            self.s = None
            u = np.arange(self.cfg.step, self.cfg.horizon + 1e-9, self.cfg.step)
            return np.asarray(car_xy, float) + u[:, None] * np.array([math.cos(car_yaw), math.sin(car_yaw)])
        q = self.to_layout(np.asarray(car_xy, float)[None])[0]
        # (a consistent but not yet locked fit is used provisionally: on the entry line it only
        #  shifts the straight a little, and it keeps being refitted every tick until it locks)
        if self.s is None:
            d = np.linalg.norm(self.ref - q, axis=1)
            entry = self.ref_s <= self.phase_bounds[1][0]  # first pick: on the entry line
            i = int(np.argmin(np.where(entry, d, np.inf)))
            self.s = float(self.ref_s[i])
        lo, hi = np.searchsorted(self.ref_s, [self.s - 1.0, self.s + 6.0])
        hi = max(hi, lo + 2)
        i = lo + int(np.argmin(np.linalg.norm(self.ref[lo:hi] - q, axis=1)))
        self.s = max(self.s, float(self.ref_s[i]))
        u = np.arange(self.s + self.cfg.step, min(self.s + self.cfg.horizon, self.ref_s[-1]) + 1e-9, self.cfg.step)
        if len(u) == 0:
            return np.zeros((0, 2))
        Q = np.c_[np.interp(u, self.ref_s, self.ref[:, 0]), np.interp(u, self.ref_s, self.ref[:, 1])]
        return self.to_map(Q)

    def info(self):
        lap = {"entry": 0, "exit": 2 * self.cfg.laps_per_side + 1}
        ph = self.phase(self.s) if self.s is not None else "entry"
        names = [n for _, n in self.phase_bounds]
        return dict(mode="SKIDPAD", phase=ph, lap=names.index(ph) if ph in names else 0,
                    locked=bool(self.locked), layout=("locked" if self.locked else
                                                      "provisional" if self.consistent else "none (straight ahead)"), fit_rms_m=round(float(self.rms), 3) if np.isfinite(self.rms) else None,
                    cones_used=int(self.n_used), cones_in_memory=int(len(self.mem)),
                    s_m=round(float(self.s or 0.0), 2), total_m=round(float(self.ref_s[-1]), 1),
                    pose=[round(float(v), 3) for v in self.pose] if self.pose is not None else None,
                    sigma_xy_m=round(self.sigma[0], 4) if np.isfinite(self.sigma[0]) else None,
                    sigma_yaw_deg=round(self.sigma[1], 3) if np.isfinite(self.sigma[1]) else None,
                    arc_deg=[round(a) for a in self.arcs],
                    colour={1: "normal", -1: "swapped", 0: "unused"}[self.colour_conv],
                    reason="" if self.locked else ("fitting the layout"
                                                   + ("" if self.consistent else " (fit does not match the car's pose yet)")
                                                   + ("; waiting for: " + ", ".join(k for k, v in self.lock_checks.items() if not v)
                                                      if self.lock_checks else "")))

    def circles_map(self, n=90):
        out = []
        for ctr in self.centres:
            for r in (self.cfg.inner_radius, self.cfg.outer_radius):
                a = np.linspace(0, 2 * math.pi, n)
                out.append(self.to_map(ctr + r * np.c_[np.cos(a), np.sin(a)]))
        return out


# ============================================================ ROS node
def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class SkidpadNode(Node):
    def __init__(self):
        super().__init__("pp_skidpad_tels")
        dp = self.declare_parameter
        dp("topics.odom_in", "/slam/odom")
        dp("topics.map_in", "/slam/map_cones")
        dp("topics.path_out", "/path_points")
        dp("qos.best_effort", True)
        dp("qos.depth", 50)
        dp("publish.rate_hz", 20.0)
        d = SkidpadConfig()
        for k, v in (("path_radius_m", d.path_radius), ("inner_radius_m", d.inner_radius),
                     ("outer_radius_m", d.outer_radius), ("centre_distance_m", d.centre_distance),
                     ("lane_half_width_m", d.lane_half_width), ("laps_per_side", d.laps_per_side),
                     ("first_side", d.first_side), ("start_to_centre_m", d.start_to_centre),
                     ("entry_length_m", d.entry_length), ("exit_length_m", d.exit_length),
                     ("horizon_m", d.horizon), ("lock_min_cones", d.lock_min_cones),
                     ("lock_rms_m", d.lock_rms), ("gate_half_width_m", d.gate_half_width),
                     ("lock_min_per_circle", d.lock_min_per_circle), ("lock_min_arc_deg", d.lock_min_arc_deg),
                     ("lock_sigma_xy_m", d.lock_sigma_xy), ("lock_sigma_deg", d.lock_sigma_deg),
                     ("lock_stable_fits", d.lock_stable_fits), ("colour_mode", d.colour_mode),
                     ("colour_penalty_m", d.colour_penalty)):
            dp("skidpad." + k, v)
        dp("skidpad.use_unknown", True)
        dp("debug.publish", True)
        g = lambda k: self.get_parameter(k).value
        self.cfg = SkidpadConfig(
            path_radius=float(g("skidpad.path_radius_m")), inner_radius=float(g("skidpad.inner_radius_m")),
            outer_radius=float(g("skidpad.outer_radius_m")), centre_distance=float(g("skidpad.centre_distance_m")),
            lane_half_width=float(g("skidpad.lane_half_width_m")), laps_per_side=int(g("skidpad.laps_per_side")),
            first_side=str(g("skidpad.first_side")).strip().lower(),
            start_to_centre=float(g("skidpad.start_to_centre_m")), entry_length=float(g("skidpad.entry_length_m")),
            exit_length=float(g("skidpad.exit_length_m")), horizon=float(g("skidpad.horizon_m")),
            lock_min_cones=int(g("skidpad.lock_min_cones")), lock_rms=float(g("skidpad.lock_rms_m")),
            gate_half_width=float(g("skidpad.gate_half_width_m")),
            lock_min_per_circle=int(g("skidpad.lock_min_per_circle")),
            lock_min_arc_deg=float(g("skidpad.lock_min_arc_deg")),
            lock_sigma_xy=float(g("skidpad.lock_sigma_xy_m")), lock_sigma_deg=float(g("skidpad.lock_sigma_deg")),
            lock_stable_fits=int(g("skidpad.lock_stable_fits")),
            colour_mode=str(g("skidpad.colour_mode")).strip().lower(),
            colour_penalty=float(g("skidpad.colour_penalty_m")))
        self.use_unknown = bool(g("skidpad.use_unknown"))
        self.planner = SkidpadPlanner(self.cfg)
        qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=int(g("qos.depth")),
                         reliability=QoSReliabilityPolicy.BEST_EFFORT if bool(g("qos.best_effort"))
                         else QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.VOLATILE)
        pub_qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=int(g("qos.depth")),
                             reliability=QoSReliabilityPolicy.BEST_EFFORT, durability=QoSDurabilityPolicy.VOLATILE)
        self.create_subscription(Odometry, str(g("topics.odom_in")), self.cb_odom, qos)
        self.create_subscription(ConeArrayWithCovariance, str(g("topics.map_in")), self.cb_map, qos)
        self.path_pub = self.create_publisher(Path, str(g("topics.path_out")), pub_qos)
        self.dbg = bool(g("debug.publish"))
        if self.dbg:
            dq = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=5,
                            reliability=QoSReliabilityPolicy.BEST_EFFORT, durability=QoSDurabilityPolicy.VOLATILE)
            self.mode_pub = self.create_publisher(String, "/pp/mode", dq)
            self.marker_pub = self.create_publisher(MarkerArray, "/pp/debug_markers", dq) if _HAS_VIZ else None
        self.lock = threading.Lock()
        self.car = None
        self.new = {"blue": [], "yellow": [], "unknown": [], "orange": []}
        self.n_tick = 0
        self.create_timer(1.0 / max(float(g("publish.rate_hz")), 1e-6), self._tick)
        self.get_logger().info(
            f"skidpad planner: R path {self.cfg.path_radius} m, circles {self.cfg.centre_distance} m apart, "
            f"{self.cfg.laps_per_side} laps per side, {self.cfg.first_side} first, "
            f"crossing assumed {self.cfg.start_to_centre} m ahead of the start pose")

    def cb_odom(self, m):
        p = m.pose.pose
        with self.lock:
            self.car = (float(p.position.x), float(p.position.y), _yaw(p.orientation))

    def cb_map(self, m):
        xy = lambda arr: [(c.point.x, c.point.y) for c in arr]
        unk = xy(getattr(m, "unknown_color_cones", [])) if self.use_unknown else []
        with self.lock:
            self.new["blue"] += xy(m.blue_cones)
            self.new["yellow"] += xy(m.yellow_cones)
            self.new["unknown"] += unk
            self.new["orange"] += xy(m.orange_cones) + xy(getattr(m, "big_orange_cones", []))

    def _tick(self):
        with self.lock:
            car = self.car
            new = self.new
            self.new = {"blue": [], "yellow": [], "unknown": [], "orange": []}
        if car is None:
            return
        self.planner.add_cones(**new)
        path = self.planner.plan(car[:2], car[2])
        self._publish(car, path)
        self.n_tick += 1
        if self.dbg and self.n_tick % 2 == 0:
            self._publish_debug()

    def _publish(self, car, path):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        for x, y in [car[:2]] + [tuple(p) for p in path]:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = float(x), float(y), 0.0
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)

    def _publish_debug(self):
        info = self.planner.info()
        s = String(); s.data = json.dumps(info); self.mode_pub.publish(s)
        if self.marker_pub is None:
            return
        stamp = self.get_clock().now().to_msg()
        arr = MarkerArray()
        clr = Marker(); clr.header.frame_id = "map"; clr.header.stamp = stamp; clr.action = Marker.DELETEALL
        arr.markers.append(clr)
        k = Marker(); k.header.frame_id = "map"; k.header.stamp = stamp; k.ns = "edges"; k.id = 5
        k.type = Marker.LINE_LIST; k.action = Marker.ADD; k.pose.orientation.w = 1.0
        k.scale.x = 0.05; k.color.r = k.color.g = k.color.b = 0.85; k.color.a = 0.6
        for C in self.planner.circles_map():
            for a, b in zip(C[:-1], C[1:]):
                for x, y in (a, b):
                    q = Point(); q.x, q.y = float(x), float(y); k.points.append(q)
        arr.markers.append(k)
        self.marker_pub.publish(arr)


def main():
    rclpy.init()
    node = SkidpadNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
