#!/usr/bin/env python3
import math
import numpy as np
from scipy.interpolate import splprep, splev

# -------------------- Defaults / Constants --------------------

ROUTE_IS_LOOP = False
SEARCH_BACK = 10
SEARCH_FWD = 250
MAX_STEP = 60

WHEELBASE_M = 1.5
MAX_STEER_RAD = math.pi / 2  # 90 degrees
LD_BASE = 3.5
LD_GAIN = 0.6
LD_MIN = 2.0
LD_MAX = 7.0

V_MAX = 12.0
AY_MAX = 4.0
AX_MAX = 5.0
AX_MIN = -4.0

PROFILE_WINDOW_M = 100.0
NUM_ARC_POINTS = 800
PROFILE_HZ = 10
BRAKE_GAIN = 0.7

STOP_SPEED_THRESHOLD = 0.1

# Jerk-limited velocity profile defaults
V_MIN = 5.0
A_MAX = 1.0
D_MAX = 20.0
J_MAX = 70.0
CURVATURE_MAX = 0.9

# -------------------- Utility Functions --------------------

def preprocess_path(xs, ys, loop=True):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if loop:
        x_next = np.roll(xs, -1)
        y_next = np.roll(ys, -1)
    else:
        x_next = np.concatenate((xs[1:], xs[-1:]))
        y_next = np.concatenate((ys[1:], ys[-1:]))
    seglen = np.hypot(x_next - xs, y_next - ys)
    if not loop:
        seglen[-1] = 0.0
    s = np.concatenate(([0.0], np.cumsum(seglen[:-1])))
    return xs, ys, s, float(seglen.sum())


def resample_track(x_raw, y_raw, num_arc_points=NUM_ARC_POINTS):
    if len(x_raw) < 2:
        return np.array(x_raw), np.array(y_raw)

    tck, _ = splprep([x_raw, y_raw], s=0, k=min(3, max(1, len(x_raw) - 1)))
    tt_dense = np.linspace(0, 1, 2000)
    xx, yy = splev(tt_dense, tck)
    s_dense = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(xx), np.diff(yy)))))

    if s_dense[-1] == 0:
        x_res = np.interp(np.linspace(0, 1, num_arc_points), np.linspace(0, 1, len(xx)), xx)
        y_res = np.interp(np.linspace(0, 1, num_arc_points), np.linspace(0, 1, len(yy)), yy)
        return x_res, y_res

    s_dense /= s_dense[-1]
    s_uniform = np.linspace(0, 1, num_arc_points)
    return np.interp(s_uniform, s_dense, xx), np.interp(s_uniform, s_dense, yy)


def segment_distances(xs, ys, loop=True):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    x_next = np.roll(xs, -1)
    y_next = np.roll(ys, -1)
    ds = np.hypot(x_next - xs, y_next - ys)
    if not loop:
        ds[-1] = 0.0
    return ds


def preprocess_path_and_seg(xs, ys, loop=True):
    return preprocess_path(xs, ys, loop)


def compute_signed_curvature(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = np.power(dx * dx + dy * dy, 1.5) + 1e-12
    kappa = (dx * ddy - dy * ddx) / denom
    kappa = np.clip(kappa, -CURVATURE_MAX, CURVATURE_MAX)
    kappa[~np.isfinite(kappa)] = 0.0
    return kappa


def ackermann_curv_speed_limit(kappa, wheelbase=WHEELBASE_M, v_max=V_MAX, d_max=D_MAX):
    delta = np.arctan(kappa * wheelbase)
    denom = np.abs(np.tan(delta)) + 1e-6
    v = np.sqrt(np.maximum(0.0, (d_max * wheelbase) / denom))
    return np.minimum(v, v_max)


def calc_lookahead(speed_mps):
    return max(LD_MIN, min(LD_MAX, LD_BASE + LD_GAIN * speed_mps))


def forward_index_by_distance(near_idx, Ld, s, total_len, loop=True):
    if len(s) == 0:
        return near_idx
    if loop:
        target = (s[near_idx] + Ld) % total_len
        return int(np.searchsorted(s, target, side="left") % len(s))
    else:
        target = min(s[near_idx] + Ld, s[-1])
        return int(np.searchsorted(s, target, side="left"))


def local_closest_index(xy, xs, ys, cur_idx, loop=True):
    x0, y0 = xy
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    N = len(xs)
    if N == 0:
        return 0

    if loop:
        start = (cur_idx - SEARCH_BACK) % N
        count = min(N, SEARCH_BACK + SEARCH_FWD + 1)
        idxs = (np.arange(start, start + count) % N)
        dx, dy = xs[idxs] - x0, ys[idxs] - y0
        j = int(np.argmin(dx * dx + dy * dy))
        return int(idxs[j])
    else:
        i0 = max(0, cur_idx - SEARCH_BACK)
        i1 = min(N, cur_idx + SEARCH_FWD + 1)
        dx, dy = xs[i0:i1] - x0, ys[i0:i1] - y0
        j = int(np.argmin(dx * dx + dy * dy))
        return i0 + j


def calc_lookahead_point(speed, xs, ys, near_idx, s, total_len, loop=True):
    Ld = calc_lookahead(speed)
    tgt_idx = forward_index_by_distance(near_idx, Ld, s, total_len, loop)
    tx, ty = xs[tgt_idx], ys[tgt_idx]
    return tx, ty, Ld, tgt_idx


def pure_pursuit_steer(pos_xy, yaw, tx, ty, Ld):
    dx, dy = tx - pos_xy[0], ty - pos_xy[1]
    cy, sy = math.cos(yaw), math.sin(yaw)
    x_rel, y_rel = cy * dx + sy * dy, -sy * dx + cy * dy
    denom = max(0.5, Ld) ** 2
    kappa = 2.0 * y_rel / denom
    delta = math.atan(WHEELBASE_M * kappa)
    return max(-1.0, min(1.0, delta / MAX_STEER_RAD))


def cross_track_error(cx, cy, xs, ys, idx, loop=True):
    theta_ref = path_heading(xs, ys, idx, loop)
    dx = cx - xs[idx]
    dy = cy - ys[idx]
    e_lat = -math.sin(theta_ref) * dx + math.cos(theta_ref) * dy
    return e_lat, theta_ref


def path_heading(xs, ys, idx, loop=True):
    n = len(xs)
    if n == 0:
        return 0.0
    i2 = (idx + 1) % n if loop else min(idx + 1, n - 1)
    dx = xs[i2] - xs[idx]
    dy = ys[i2] - ys[idx]
    return math.atan2(dy, dx)


def jerk_limited_velocity_profile(v_limit, ds, v0, vf, v_min, v_max, a_max, d_max, j_max):
    v_limit = np.asarray(v_limit, dtype=float)
    ds = np.asarray(ds, dtype=float)
    N = len(v_limit)
    if N == 0:
        return np.array([])
    if len(ds) != N:
        raise ValueError("ds length must equal v_limit length (ds[0] is 0 for the first point)")

    # Forward pass (acceleration limited)
    v_forward = np.zeros(N, dtype=float)
    v_forward[0] = min(max(v0, 0.0), v_limit[0], v_max)
    a_prev = 0.0
    for i in range(1, N):
        ds_i = max(ds[i], 1e-9)
        v_avg = max(v_min, v_forward[i - 1])
        dt = ds_i / max(v_avg, 1e-6)
        a_curr = min(a_prev + j_max * dt, a_max)
        v_possible = math.sqrt(max(0.0, v_forward[i - 1] ** 2 + 2.0 * a_curr * ds_i))
        v_forward[i] = min(v_possible, v_limit[i], v_max)
        a_prev = (v_forward[i] ** 2 - v_forward[i - 1] ** 2) / (2.0 * ds_i)

    # Backward pass (deceleration limited)
    v_profile = v_forward.copy()
    v_profile[-1] = min(v_profile[-1], max(0.0, vf))
    a_prev = 0.0
    for i in range(N - 2, -1, -1):
        ds_i = max(ds[i + 1], 1e-9)
        v_avg = max(v_min, v_profile[i + 1])
        dt = ds_i / max(v_avg, 1e-6)
        a_curr = min(a_prev + j_max * dt, d_max)
        v_possible = math.sqrt(max(0.0, v_profile[i + 1] ** 2 + 2.0 * a_curr * ds_i))
        v_profile[i] = min(v_profile[i], v_possible, v_max)
        a_prev = (v_profile[i + 1] ** 2 - v_profile[i] ** 2) / (2.0 * ds_i)

    v_profile = np.minimum(v_profile, v_max)
    if N > 1:
        v_profile[:-1] = np.maximum(v_profile[:-1], v_min)
    if vf <= 0.0:
        v_profile[-1] = 0.0
    return v_profile


# -------------------- PID helpers --------------------

class PID:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self._i = 0.0
        self._prev_err = None

    def reset(self):
        self._i, self._prev_err = 0.0, None

    def update(self, err, dt):
        dt = max(dt, 1e-3)
        self._i += err * dt
        d = 0 if self._prev_err is None else (err - self._prev_err) / dt
        self._prev_err = err
        return max(0, min(1, self.kp * err + self.ki * self._i + self.kd * d))


class PIDRange:
    def __init__(self, kp, ki, kd, out_min=-1.0, out_max=1.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self._i = 0.0
        self._prev_err = None

    def reset(self):
        self._i, self._prev_err = 0.0, None

    def update(self, err, dt):
        dt = max(dt, 1e-3)
        self._i += err * dt
        d = 0.0 if self._prev_err is None else (err - self._prev_err) / dt
        self._prev_err = err
        u = self.kp * err + self.ki * self._i + self.kd * d
        return max(self.out_min, min(self.out_max, u))
