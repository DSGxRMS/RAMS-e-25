#!/usr/bin/env python3
"""
mppi_controller.py — steer-only MPPI for RAMS-e-25 (EUFS Gazebo).

Pipeline per cycle (compute()):
  1. Splines the rolling-buffer path (B-spline, 40 pts) -> smooth reference,
     analytic curvature, corner-speed profile.
  2. Speed target = corner profile (v = min(V_MAX, sqrt(A_LAT/|kappa|)) with a
     backward braking pass), capped by remaining visible path length.
  3. Steering: K=1024 rollouts of a grey-box Neural ODE (1 s horizon), scored
     on crosstrack + heading; softmax-weighted average of the best samples is
     the steering command (K_FF=0, MPPI carries all steering).
  4. Longitudinal: PI throttle on the speed target, sent as acceleration.

Performance: the K=1024 x T=200 RK4 rollout is captured into a CUDA graph and
replayed as a single GPU submission (~80 ms in WSL2; ~3.9 s without the graph
because WSL2 CUDA passthrough adds ~0.6 ms per kernel launch).

Guards (set flags consumed by control_node):
  path_is_bad     path heading > HD_BAD_RAD off the nose AND one-sided cones
  path_exhausted  smoothed remaining path <= STOP_PATH_M (deadlock breaker)

AUTHOR: Soumil (RMS controls team)
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import splprep, splev

from controls.path_reference import project_reference
from controls.cone_clearance import (
    CLEARANCE_CRAWL_MPS, ClearanceSpeedGovernor, clearance_log_penalty,
    cone_clearance_enabled, nearby_cones, swept_clearance,
)
from controls.random_seed import sampling_seed
from controls.steering_actuator import SteeringActuator, steering_actuator_enabled, steering_option

# =========================================================================
# Checkpoints are package data, available in source and colcon installs.
def _find_ckpt(filename="best_model_600.pt"):
    return Path(__file__).parent / "fs_model" / filename

CKPT_PATH = _find_ckpt()

# Rollout state indices (matches export_fs_to_matlab.py)
VX_I, VY_I, YR_I, SIN_I, COS_I, SLIP_I = 0, 1, 2, 3, 4, 5


# =========================================================================
# Grey-box Neural ODE (exact port of the MATLAB-side GB class)
# =========================================================================
# Exp13 evaluation bounds from data2 training runs 0-3, with the Vx floor.
_EXP13_X_MIN = (0.0, -1.17195213, -2.06207252, -1.25007439, -1.25000715, -0.27041510)
_EXP13_X_MAX = (13.90166473, 1.17151773, 2.06071639, 1.25007391, 1.25006676, 0.27883193)


class _Func(nn.Module):
    """Learn the four body-dynamics channels, with exact yaw kinematics."""

    def __init__(self, mirror_symmetric=False):
        super().__init__()
        self.mirror_symmetric = mirror_symmetric
        self.net = nn.Sequential(
            nn.Linear(6 if mirror_symmetric else 8, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 4),
        )

    def forward(self, x, u, x_mean, x_std, u_mean, u_std):
        xn = (x - x_mean) / x_std
        un = (u - u_mean) / u_std
        if self.mirror_symmetric:
            even = torch.stack((xn[..., VX_I], un[..., 1]), dim=-1)
            odd = torch.stack((xn[..., VY_I], xn[..., YR_I],
                               xn[..., SLIP_I], un[..., 0]), dim=-1)
            original = torch.cat((even, odd), dim=-1)
            reflected = torch.cat((even, -odd), dim=-1)
            g_original, g_reflected = self.net(
                torch.stack((original, reflected), dim=0)
            ).unbind(dim=0)
            g = torch.cat((
                0.5 * (g_original[..., :1] + g_reflected[..., :1]),
                0.5 * (g_original[..., 1:] - g_reflected[..., 1:]),
            ), dim=-1)
        else:
            g = self.net(torch.cat((xn, un), dim=-1))
        yr  = x[..., YR_I]
        ys  = x[..., SIN_I]
        yc  = x[..., COS_I]
        # ENU: d(sin)/dt = cos*w , d(cos)/dt = -sin*w
        return torch.stack([
            g[..., 0] * x_std[VX_I],    # dVx
            g[..., 1] * x_std[VY_I],    # dVy
            g[..., 2] * x_std[YR_I],    # dyawRate
            yc * yr,                    # d(yaw_sin)  exact
            -ys * yr,                   # d(yaw_cos)  exact
            g[..., 3] * x_std[SLIP_I],  # dslip
        ], dim=-1)


def _rk4(f, x, u, h, *s):
    k1 = f(x,             u, *s)
    k2 = f(x + 0.5*h*k1, u, *s)
    k3 = f(x + 0.5*h*k2, u, *s)
    k4 = f(x + h*k3,     u, *s)
    return x + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


class _GB(nn.Module):
    """Grey-box ODE wrapper: RK4 integration + yaw-vector renormalisation."""

    def __init__(self, ck, mirror_symmetric=False):
        super().__init__()
        self.func = _Func(mirror_symmetric)
        self.nr   = int(ck["num_rk4_steps"])
        for name in ("x_mean", "x_std", "u_mean", "u_std", "x_min", "x_max"):
            self.register_buffer(name, torch.as_tensor(ck[name], dtype=torch.float32))

    def step(self, x, u, dt, euler=False):
        """x: (...,6)  u: (...,2)  dt: scalar tensor."""
        if euler:
            dx = self.func(x, u, self.x_mean, self.x_std, self.u_mean, self.u_std)
            x  = x + dt * dx
        else:
            h = dt / self.nr
            for _ in range(self.nr):
                x = _rk4(self.func, x, u, h,
                         self.x_mean, self.x_std, self.u_mean, self.u_std)
        a   = x[..., SIN_I]
        b   = x[..., COS_I]
        nrm = torch.sqrt(a * a + b * b).clamp_min(1e-6)
        x   = x.clone()
        x[..., SIN_I] = a / nrm
        x[..., COS_I] = b / nrm
        return torch.clamp(x, self.x_min, self.x_max)


# =========================================================================
# Small utilities
# =========================================================================
def _angle_diff(a, b):
    d = a - b
    if isinstance(d, torch.Tensor):
        return torch.atan2(torch.sin(d), torch.cos(d))
    return math.atan2(math.sin(d), math.cos(d))


def _world_vel(x_K6):
    """Body (vx, vy) -> world (East, North).

    TRAINING convention for the heading slots (verified on data row 1:
    yaw_sin=-1, yaw_cos=0 at theta=0):
        yaw_sin_slot = -cos(theta)   yaw_cos_slot = sin(theta)
    """
    S = x_K6[:, SIN_I]
    C = x_K6[:, COS_I]
    vx = x_K6[:, VX_I]
    vy = x_K6[:, VY_I]
    return (-(vx * S + vy * C), (vx * C - vy * S))


def _interp_1d(s_knots, values, s_query):
    """GPU linear interpolation; s_knots must be sorted."""
    idx = torch.searchsorted(s_knots.contiguous(), s_query.contiguous())
    idx = idx.clamp(1, s_knots.shape[0] - 1)
    s0, s1 = s_knots[idx - 1], s_knots[idx]
    v0, v1 = values[idx - 1], values[idx]
    alpha = ((s_query - s0) / (s1 - s0 + 1e-8)).clamp(0.0, 1.0)
    return v0 + alpha * (v1 - v0)


# =========================================================================
# MPPIController
# =========================================================================
class MPPIController:
    """Steer-only MPPI with Neural ODE rollouts, CUDA-graph accelerated."""

    # ---- rollout / sampling ----
    K           = 1024   # rollouts per cycle
    T_NEAR      = 200    # near horizon: 1.0 s (used BELOW T_SWITCH_SPEED: already
                         #   inside the corner -- the 6.5 m reference covers it and
                         #   full command rate matters more than foresight)
    T_FAR       = 400    # far horizon: 2.0 s (used AT/ABOVE T_SWITCH_SPEED: fast on
                         #   a straight the corner must enter the scored window
                         #   EARLY -- at 5 m/s a 1 s rollout sees only ~9 m of
                         #   reference and turn-in comes too late. We can afford
                         #   late acceleration; never late braking / turning.)
    T_SWITCH_SPEED = 3.0 # m/s: at/above this use the far horizon
    T_MAX       = 400    # buffer width (must equal max(T_NEAR, T_FAR))
    DT_EFF      = 0.005  # rollout dt MUST equal the training-data dt
    USE_EULER   = False  # RK4 (CUDA graph makes it affordable)
    LAMBDA_FRAC = 0.15   # adaptive softmax temp = frac of (cost_max - cost_min)
    LAMBDA_MIN  = 1e-3   # floor so lam never reaches 0
    SIG_STEER   = 0.08   # per-step steer jitter (rad): fine exploration
    SIG_BIAS    = 0.05   # one sustained steer offset per sample (rad): without it
                         # all 1024 rollouts converge to the same trajectory and
                         # MPPI goes degenerate (nEff -> K). 0.12 caused violent
                         # oscillation; 0.05 is the working compromise.
    STEER_MAX   = 0.349  # 20 deg, model range

    # ---- rollout cost weights ----
    W_CT        = 1.0    # crosstrack (Huber)
    W_HEAD      = 0.6    # pure-pursuit heading term; >0 makes MPPI approach the
                         # line tangentially instead of slamming across it
    CT_SAT      = 5.0    # Huber knee (m)
    K_FF        = 0      # curvature feedforward gain. DISABLED: MPPI drives the
                         # wheel (corr(net,MPPI)=0.76 vs corr(net,FF)=0.19) and
                         # the FF fought MPPI in ~half the frames. Kept as a
                         # tunable + CSV diagnostic (ff_part).
    SPLINE_NPTS   = 40   # dense points resampled from the B-spline
    SPLINE_SMOOTH = 0.12 # spline smoothing per point (irons out buffer point noise)

    # ---- longitudinal (PI throttle on accel output) ----
    KP          = 0.50
    KI          = 0.10
    I_CLAMP     = 1.5    # max |integral| contribution KI*I = 0.15 of full throttle
    I_BAND      = 1.0    # integrate only within this error band (no windup in transients)
    KP_ROLL     = 0.40   # P-only longitudinal INSIDE the rollout
    ACCEL_MAX   = 3.0    # m/s^2 (EUFS sim clip -3..3)

    # ---- speed profile / path-length caps ----
    V_MAX       = 4.0    # m/s
    A_LAT_MAX   = 2.5    # m/s^2 lateral budget: v = min(V_MAX, sqrt(A_LAT/|kappa|)
                         # + backward braking pass (brake BEFORE the corner)
    BRAKE_A     = 2.5    # m/s^2 braking used by the backward pass and path-end ramp
    HORIZON_CAP_FRAC = 0.5  # keep >= this fraction of the horizon on visible path
    STOP_PATH_M = 1.5    # remaining path below this -> command a full stop
    V_CREEP     = 1.0    # seed value for the smoothed speed target

    # ---- P19: horizon-switch gate + launch governor ----
    # P19-c: the two thresholds are DECOUPLED. The simulator's camera envelope
    # (cone colour survives only within 15 m / +-60 deg) caps chain span at
    # ~16 m, so ahead_eff ceilings out around 11-12 m on straights -- a 12 m
    # release threshold kept the car governed for 67% of the 2026-08-26 run
    # (915/1364 cycles pinned at exactly 2.50, released only at t=187 s).
    FAR_MIN_AHEAD_M  = 12.0  # min smoothed path-ahead to engage/keep FAR: a
                             #   shorter spline saturates s_tgt at its end for
                             #   most of the 400-step rollout -> degenerate
                             #   steering scores (the 2026-08-26 dive)
    LAUNCH_RELEASE_M = 9.0   # path-ahead that releases the governor; fast
                             #   NEAR-only driving below FAR is proven safe
                             #   (post-release sprint to 3.96 m/s was NEAR)
    LAUNCH_CAP_MPS   = 2.5   # target-speed ceiling while the path is immature
    CT0         = 8.0    # CTE (m) at which the off-line speed reduction maxes out
    K_CT        = 0.4    # off-line speed reduction strength (soft: keeps enough
                         # speed for MPPI to see far enough to steer back)

    # ---- guards ----
    HD_BAD_RAD  = 1.4   # ~80 deg: path heading beyond this = geometry-garbage
                         # (combined with one_sided cone evidence -> path_is_bad)
    LOOKAHEAD_M = 4.0   # fixed arc-length preview (m). Do NOT make speed-adaptive:
                         # shrinking it in corners removed anticipation -> ran wide.

    # ---- speed fed to the model ----
    V_PLAN_MAX  = 13.0   # exp10 model training ceiling (~13.9 m/s)
    SPEED_SANE_MAX = 16.0  # reject absurd odom readings
    L_WB_KIN    = 1.535  # wheelbase (m) for kinematic feedforward / viz

    def __init__(self, ckpt_path=None, force_cpu=False):
        # ---- device ----
        if torch.cuda.is_available() and not force_cpu:
            self.dev  = torch.device("cuda")
            self.use_graph = True
            d = torch.cuda.get_device_properties(0)
            print(f"[MPPI] GPU {d.name} ({d.total_memory/1e9:.1f} GB) + CUDA graph")
        else:
            self.dev  = torch.device("cpu")
            self.use_graph = False
            print("[MPPI] CPU mode (no CUDA graph)")

        self.seed = sampling_seed()
        self._rng = torch.Generator(device=self.dev).manual_seed(self.seed)
        print(f"[MPPI] sampling seed={self.seed}", flush=True)

        # ---- load checkpoint (weights AND norm buffers) ----
        model_name = os.environ.get("MPPI_MODEL", "exp13")
        self.steering_actuator_enabled = steering_actuator_enabled(model_name)
        self.elapsed_warm_start = steering_option("MPPI_ELAPSED_WARM_START", model_name == "exp13")
        self.cone_clearance = cone_clearance_enabled(model_name)
        self.clearance_speed = (ClearanceSpeedGovernor(self.V_MAX)
                                if self.cone_clearance and steering_option("MPPI_CLEARANCE_SPEED", True)
                                else None)
        if self.cone_clearance and (model_name != "exp13" or self.K_FF
                                    or os.environ.get("MPPI_FORCE_STEER0") == "1"):
            raise ValueError("MPPI_CONE_CLEARANCE requires the standard Exp13 steering output")
        print(f"[MPPI] EUFS steering actuator model={'on' if self.steering_actuator_enabled else 'off'}",
              flush=True)
        print(f"[MPPI] cone clearance={'on' if self.cone_clearance else 'off'}, "
              f"clearance speed={'on' if self.clearance_speed is not None else 'off'}, "
              f"V_MAX={self.V_MAX:g} m/s", flush=True)
        if model_name == "exp13":
            cp = ckpt_path or _find_ckpt("exp13_stage1_pretrained.pt")
            state = torch.load(str(cp), map_location="cpu", weights_only=True)
            state["x_min"] = torch.tensor(_EXP13_X_MIN, dtype=torch.float32)
            state["x_max"] = torch.tensor(_EXP13_X_MAX, dtype=torch.float32)
            self.model = _GB({**state, "num_rk4_steps": 1}, mirror_symmetric=True)
        elif model_name == "exp10":
            cp = ckpt_path or CKPT_PATH
            ck = torch.load(str(cp), map_location="cpu", weights_only=False)
            self.model = _GB(ck)
            state = ck["model_state_dict"]
        else:
            raise ValueError(f"Unknown MPPI_MODEL={model_name!r}. Use exp10 or exp13.")
        self.model.load_state_dict(state)
        self.model = self.model.to(self.dev).eval()
        print(f"[MPPI] dynamics={model_name} checkpoint={cp}", flush=True)
        self._dt_t = torch.tensor(self.DT_EFF, dtype=torch.float32, device=self.dev)

        # ---- static buffers (fixed addresses, required for CUDA graph capture) ----
        dev = self.dev
        TM = self.T_MAX
        self._A      = torch.zeros(self.K, TM, device=dev)  # steer samples
        self._wheel_angle = torch.zeros((), device=dev)
        self._steer_index = torch.arange(TM, device=dev)
        self._queued_steer = torch.zeros(TM, device=dev)
        self._refE   = torch.zeros(TM, device=dev)
        self._refN   = torch.zeros(TM, device=dev)
        self._nE     = torch.zeros(TM, device=dev)
        self._nN     = torch.zeros(TM, device=dev)
        self._vtgt   = torch.zeros(TM, device=dev)          # target speed over horizon
        self._ff     = torch.zeros(TM, device=dev)          # curvature feedforward
        self._x0     = torch.zeros(6, device=dev)           # initial state
        self._p0     = torch.zeros(2, device=dev)           # initial position
        self._cost   = torch.zeros(self.K, device=dev)
        self._trajE  = torch.zeros(TM, self.K, device=dev)  # per-step East pos of every
        self._trajN  = torch.zeros(TM, self.K, device=dev)  # rollout (live viz, P14-c)
        self._trajYaw = torch.zeros_like(self._trajN) if self.cone_clearance else None
        self._selected_actions = torch.zeros(1, TM, device=dev) if self.cone_clearance else None
        self._selected_graphs = {}
        self._graphs = {}                                    # horizon -> captured graph
        self.horizon = self.T_NEAR                           # active horizon this cycle

        print(f"[MPPI] Loaded {Path(cp).name}  K={self.K}  "
              f"T={self.T_NEAR}/{self.T_FAR} (@<{self.T_SWITCH_SPEED} m/s)  "
              f"dt_eff={self.DT_EFF}s  RK4={not self.USE_EULER}")

        # ---- warm-start nominal steer (full buffer width; only the active
        # horizon columns are consumed by the rollout each cycle) ----
        self.nom = torch.zeros(1, TM, device=dev)
        self._previous_steering_plan = None

        # ---- longitudinal PI state ----
        self.I_spd  = 0.0
        self._ahead_lp = 8.0           # smoothed path-ahead (m)
        self._vr_lp = self.V_CREEP     # smoothed speed target
        self.launch_done = False       # P19-b: latches True once the path matures

        # ---- path cache ----
        self._path_key  = None
        self._path_data = None
        self.cur        = 0            # closest-point index (exposed in CSV)
        self._progress_s = None

        # ---- yaw rate: finite difference + low-pass (seeds rollout x0) ----
        self.prev_yaw = None
        self.yaw_rate = 0.0

        self._dbg_ctr = 0
        self.audit_enabled = os.environ.get("MPPI_AUDIT") == "1"

        # ---- optional CSV trace (set MPPI_CSV=/path/to.csv) ----
        self._csv = None
        csv_path = os.environ.get("MPPI_CSV", "")
        if csv_path:
            self._csv = open(csv_path, "w")
            self._csv.write(
                "time,x,y,yaw,odom_spd,use_spd,v_plan,vr_eff,steer,accel,lon0,"
                "cte,heading_err,cost_min,cost_max,n_eff,a0_std,pace_m,roll_ms,"
                "I_spd,yaw_rate,cur,N,ff_part,mppi_part,"
                "clearance_infeasible_pct,clearance_weighted_margin_m,clearance_selected_margin_m,"
                "clearance_feasible_pct,clearance_speed_cap_mps,clearance_speed_next_cap_mps,"
                "clearance_speed_base_target_mps,clearance_initial_feasible_pct,"
                "clearance_retry_target_mps,clearance_retry_roll_ms,clearance_rollout_target_mps\n")
            print(f"[MPPI] CSV trace -> {csv_path}", flush=True)

        # ---- P18-g: eager dual-horizon warm-up (closes tracker item P4) ----
        # Capture BOTH CUDA graphs HERE, while the car is stationary. The old
        # lazy path captured inside compute() the first time the speed crossed
        # the horizon switch -- measured 79.2 s mid-run (kernel JIT for the new
        # horizon's tensor shapes): no commands were published, the sim bridge
        # held the last one, and the car drove ~23 m blind into a corner
        # (2026-08-25 run, t=2.2->81.4, roll_ms=79168). Same failure class as
        # the 41.9 s first-call stall at t=0. All one-time compilation /
        # graph instantiation now lands at startup where it costs nothing.
        if self.use_graph:
            for h in (self.T_NEAR, self.T_FAR):
                self._capture_graph(h)
                if self.cone_clearance:
                    self._capture_selected_graph(h, self.steering_actuator_enabled)
                    # Exercise the variable-size geometry path before the first command.
                    initial = torch.zeros((1, 1, 3), device=dev)
                    cones = torch.tensor([[float(i), 3., .16] for i in range(16)], device=dev)
                    poses = torch.stack((self._trajE[:h], self._trajN[:h], self._trajYaw[:h]), dim=-1)
                    swept_clearance(torch.cat((initial.expand(1, self.K, 3), poses)), cones)
                    self._check_selected(h, self.steering_actuator_enabled,
                                         self._selected_actions, initial, cones)
            print(f"[MPPI] startup warm-up complete: graphs live for "
                  f"T={self.T_NEAR}/{self.T_FAR}", flush=True)

    # =====================================================================
    #  Rollout — reads ONLY the static buffers, writes self._cost.
    #  This whole method is captured into a CUDA graph (one per horizon).
    # =====================================================================
    def _set_steering_schedule(self, wheel_angle, queued, solve_s):
        steps = np.arange(self.T_MAX)
        offset = steps - len(queued)
        # Hold the first candidate command for one estimated command interval.
        hold_steps = max(1, round(solve_s / self.DT_EFF))
        indices = np.where(offset < 0, -1, np.maximum(offset - hold_steps + 1, 0))
        queued_full = np.zeros(self.T_MAX, dtype=np.float32)
        queued_full[:len(queued)] = queued
        self._wheel_angle.fill_(wheel_angle)
        self._steer_index.copy_(torch.as_tensor(indices, device=self.dev))
        self._queued_steer.copy_(torch.as_tensor(queued_full, device=self.dev))

    def reset_steering_plan(self):
        self.nom = torch.zeros_like(self.nom)
        self._previous_steering_plan = None

    def _warm_start(self, candidate_start_s, hold_steps):
        """Resample the previous scored sequence on today's action timeline."""
        previous = self._previous_steering_plan
        if previous is None:
            return self.nom, None
        previous_start, previous_hold, available_steps, scored_steps = previous
        elapsed_steps = (candidate_start_s - previous_start) / self.DT_EFF
        if elapsed_steps >= available_steps:
            # The previous plan no longer covers the next command.
            return torch.zeros_like(self.nom), elapsed_steps

        # Candidate zero is held for several integration steps. All subsequent
        # candidates are spaced one model step apart.
        offsets = np.arange(self.T_MAX, dtype=np.float64)
        offsets[1:] += hold_steps - 1
        aligned_offsets = offsets + elapsed_steps
        indices = np.clip(aligned_offsets - previous_hold + 1, 0, scored_steps - 1)
        indices[aligned_offsets < previous_hold] = 0
        lower = np.floor(indices).astype(np.int64)
        index = torch.as_tensor(lower, device=self.dev)
        weight = torch.as_tensor(indices - lower, dtype=self.nom.dtype, device=self.dev)
        shifted = torch.lerp(self.nom[:, index], self.nom[:, (index + 1).clamp_max(scored_steps - 1)], weight)
        return shifted, elapsed_steps

    def _scheduled_steer(self, horizon, actions=None):
        indices = self._steer_index[:horizon]
        actions = self._A if actions is None else actions
        candidates = actions.index_select(1, indices.clamp_min(0))
        if self.K_FF:
            candidates = candidates + self.K_FF * self._ff.index_select(0, indices.clamp_min(0))
        return torch.where(indices < 0, self._queued_steer[:horizon], candidates)

    @torch.no_grad()
    def _rollout(self, horizon, model_actuator=False, selected=None):
        actions = self._A if selected is None else selected
        K = actions.shape[0]
        dev  = self.dev
        xk = self._x0.unsqueeze(0).expand(K, 6).contiguous()      # (K,6)
        pe = self._p0[0].expand(K).contiguous()
        pn = self._p0[1].expand(K).contiguous()
        cost = torch.zeros(K, device=dev)
        if model_actuator:
            requested_steer = self._scheduled_steer(horizon, actions)
            wheel = self._wheel_angle.expand(K)
            max_travel = SteeringActuator.RATE_RAD_S * self.DT_EFF

        selected_poses = [] if selected is not None else None
        for t in range(horizon):
            vx   = xk[:, VX_I]
            smax = (self.STEER_MAX
                    - (self.STEER_MAX - 0.14) * (vx - 13.0) / 5.0).clamp(0.14, self.STEER_MAX)
            if model_actuator:
                # Pending requests were already clipped when issued. Only limit
                # new candidates using predicted speed.
                requested = requested_steer[:, t]
                target = torch.where(self._steer_index[t] < 0, requested,
                                     requested.clamp(-smax, smax))
                wheel = wheel + (target - wheel).clamp(-max_travel, max_travel)
                steer = wheel
            else:
                steer = (self.K_FF * self._ff[t] + actions[:, t]).clamp(-smax, smax)
            lon   = (self.KP_ROLL * (self._vtgt[t] - vx)).clamp(-1.0, 1.0)
            u     = torch.stack([steer, lon], dim=1)
            # NOTE: no kinematic yaw injection — yaw evolves from the learned model

            Ve1, Vn1 = _world_vel(xk)
            xk = self.model.step(xk, u, self._dt_t, euler=self.USE_EULER)
            Ve2, Vn2 = _world_vel(xk)
            pe = pe + 0.5 * (Ve1 + Ve2) * self.DT_EFF
            pn = pn + 0.5 * (Vn1 + Vn2) * self.DT_EFF
            car_th = torch.atan2(xk[:, COS_I], -xk[:, SIN_I])
            if selected_poses is not None:
                selected_poses.append(torch.stack((pe, pn, car_th), dim=-1))
                continue
            self._trajE[t].copy_(pe)
            self._trajN[t].copy_(pn)
            if self._trajYaw is not None:
                self._trajYaw[t].copy_(car_th)

            # C1: Huber crosstrack to the reference point at t
            perp = (pe - self._refE[t]) * self._nE[t] + (pn - self._refN[t]) * self._nN[t]
            ap   = perp.abs()
            ctc  = ap.clamp(max=self.CT_SAT)**2 + 2*self.CT_SAT*(ap - self.CT_SAT).clamp(min=0.0)

            # C2: pure-pursuit heading toward the reference point (gated when the
            # point is ~on the car: the bearing is meaningless there and injects spin)
            dE = self._refE[t] - pe
            dN = self._refN[t] - pn
            bear = torch.atan2(dN, dE)
            dh   = torch.atan2(torch.sin(car_th - bear), torch.cos(car_th - bear))
            dist = torch.sqrt(dE * dE + dN * dN)
            head_gate = (dist > 1.0).float()
            cost = cost + self.W_CT * ctc + self.W_HEAD * head_gate * (dh * dh)

        if selected_poses is not None:
            return torch.stack(selected_poses)
        self._cost.copy_(cost / horizon)

    def _capture_graph(self, horizon, model_actuator=None):
        """Capture _rollout(horizon) into a CUDA graph (once per horizon;
        warmup in a side stream)."""
        if model_actuator is None:
            model_actuator = self.steering_actuator_enabled
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._rollout(horizon, model_actuator)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._rollout(horizon, model_actuator)
        torch.cuda.synchronize()
        self._graphs[horizon, model_actuator] = g
        print(f"[MPPI] CUDA graph captured (horizon={horizon} "
              f"= {horizon * self.DT_EFF:.1f}s)")

    def _capture_selected_graph(self, horizon, model_actuator):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self._rollout(horizon, model_actuator, self._selected_actions)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            poses = self._rollout(horizon, model_actuator, self._selected_actions)
        self._selected_graphs[horizon, model_actuator] = graph, poses

    def _check_selected(self, horizon, model_actuator, actions, initial_pose, cones):
        if self.use_graph:
            key = horizon, model_actuator
            if key not in self._selected_graphs:
                self._capture_selected_graph(*key)
            self._selected_actions.copy_(actions)
            graph, poses = self._selected_graphs[key]
            graph.replay()
        else:
            poses = self._rollout(horizon, model_actuator, actions)
        return float(swept_clearance(torch.cat((initial_pose, poses)), cones)[0])

    def _run_rollout_batch(self, horizon, model_actuator):
        start = time.perf_counter()
        if self.use_graph:
            key = horizon, model_actuator
            if key not in self._graphs:
                self._capture_graph(*key)
            self._graphs[key].replay()
            torch.cuda.synchronize()
        else:
            self._rollout(horizon, model_actuator)
        return (time.perf_counter() - start) * 1000.0

    def _screen_rollouts(self, horizon, initial_pose, cones):
        poses = torch.stack((self._trajE[:horizon], self._trajN[:horizon],
                             self._trajYaw[:horizon]), dim=-1)
        margins = swept_clearance(torch.cat((initial_pose.expand(1, self.K, 3), poses)), cones)
        valid = torch.isfinite(margins) & torch.isfinite(self._cost)
        clear = margins > 0
        candidates = torch.nonzero(valid & clear).flatten()
        return margins, candidates, int((valid & ~clear).sum())

    # =====================================================================
    #  update_path — spline the buffer, build reference + speed profile
    # =====================================================================
    def update_path(self, path_pts):
        if not path_pts or len(path_pts) < 2:
            self._path_key = self._path_data = self._progress_s = None
            return
        key = tuple(map(tuple, path_pts))
        if key == self._path_key:
            return
        self._path_key = key

        raw = np.array(path_pts, dtype=np.float64)
        rx0, ry0 = raw[:, 0], raw[:, 1]

        # B-spline smoothing + dense resampling: PP/buffer points are sparse and
        # slightly offset; raw they wiggle the reference and spike the curvature.
        try:
            keep = np.concatenate([[True], np.hypot(np.diff(rx0), np.diff(ry0)) > 1e-3])
            rxk, ryk = rx0[keep], ry0[keep]
            k = min(3, len(rxk) - 1)                    # cubic if >=4 pts
            if k >= 1 and len(rxk) >= 2:
                tck, _ = splprep([rxk, ryk], s=self.SPLINE_SMOOTH * len(rxk), k=k)
                u = np.linspace(0.0, 1.0, self.SPLINE_NPTS)
                rx, ry = np.asarray(splev(u, tck))
                dx1, dy1 = splev(u, tck, der=1)
                dx2, dy2 = splev(u, tck, der=2) if k >= 2 else (0.0, 0.0)
                denom = np.power(dx1*dx1 + dy1*dy1, 1.5) + 1e-9
                kap = (dx1*dy2 - dy1*dx2) / denom       # signed curvature
                heading = np.arctan2(dy1, dx1)
            else:
                raise ValueError("too few unique pts")
        except Exception:
            # fallback: raw points + finite-difference curvature
            rx, ry = rx0, ry0
            dxg, dyg = np.gradient(rx), np.gradient(ry)
            heading = np.arctan2(dyg, dxg)
            dth = np.arctan2(np.sin(np.diff(heading)), np.cos(np.diff(heading)))
            seg0 = np.maximum(np.hypot(np.diff(rx), np.diff(ry)), 1e-3)
            kap = np.concatenate([dth / seg0, [0.0]])

        N   = len(rx)
        seg = np.maximum(np.hypot(np.diff(rx), np.diff(ry)), 1e-6)
        s_np = np.concatenate([[0.0], np.cumsum(seg)])
        nE = -np.sin(heading)
        nN = np.cos(heading)

        # Feedforward steer from curvature (active only if K_FF > 0).
        # Clip at 0.3: tightest real corner ~R>3 m; unclipped noise spiked full lock.
        kap = np.clip(kap, -0.3, 0.3)
        ff_steer = np.arctan(self.L_WB_KIN * kap)

        # Corner speed profile: lateral-accel limit per point, then a backward
        # braking pass (x2 iterations to converge) so the car slows BEFORE corners.
        vt = np.minimum(self.V_MAX,
                        np.sqrt(self.A_LAT_MAX / np.maximum(np.abs(kap), 1e-3)))
        for _ in range(2):
            for i in range(N - 2, -1, -1):
                ds = max(s_np[i + 1] - s_np[i], 1e-3)
                vt[i] = min(vt[i], math.sqrt(vt[i + 1] ** 2 + 2.0 * self.BRAKE_A * ds))

        dev = self.dev
        self._path_data = {
            "N":  N,
            "s":  torch.tensor(s_np, dtype=torch.float32, device=dev),
            "rx": torch.tensor(rx,   dtype=torch.float32, device=dev),
            "ry": torch.tensor(ry,   dtype=torch.float32, device=dev),
            "nE": torch.tensor(nE,   dtype=torch.float32, device=dev),
            "nN": torch.tensor(nN,   dtype=torch.float32, device=dev),
            "vt": torch.tensor(vt,   dtype=torch.float32, device=dev),
            "ff": torch.tensor(ff_steer, dtype=torch.float32, device=dev),
            "rx_np": rx, "ry_np": ry, "s_np": s_np,
            "xy_np": np.column_stack((rx, ry)),
            "kap_np": np.asarray(kap, dtype=np.float64),
            "vt_np":  np.asarray(vt,  dtype=np.float64),
        }
        self.cur = 0
        self._progress_s = None

    # =====================================================================
    #  compute — one MPPI planning step
    # =====================================================================
    def compute(self, x, y, yaw, speed, dt_ros, one_sided=True, speed_cap=None,
                steering_prediction=None, steering_state_s=None, clearance_cones=None):
        """Returns (steer, accel, info). `one_sided` = only one cone colour
        visible (from control_node's FOV counts); gates the bad-path guard.
        `speed_cap` = optional hard cap (m/s) on the speed target -- used to
        slow the car to a crawl while it follows a freshly fed PSEUDO bridge
        (a straight-line guess; taking it at 4-5 m/s risks leaving the track)."""
        if self._path_data is None:
            return 0.0, -self.ACCEL_MAX, {
                "target_speed": 0.0, "cte": 0.0,
                "heading_err": 0.0, "mean_traj": [],
            }

        pd     = self._path_data
        dev    = self.dev
        dt_ros = max(dt_ros, 1e-3)

        # ---- speed: GT twist passed in; clamp sane + model ceiling ----
        _speed = float(np.clip(float(speed), 0.0, self.SPEED_SANE_MAX))
        v_plan = float(np.clip(_speed, 0.0, self.V_PLAN_MAX))

        # ---- P15: speed-adaptive horizon (with hysteresis) ----
        # FAST -> FAR: on a straight the car is quick and the upcoming corner
        # must enter the scored window early (2 s = ~10 m at 5 m/s). Late
        # braking / turn-in is unrecoverable; late acceleration is not.
        # SLOW -> NEAR: already inside the corner, the 6.5 m reference covers
        # it and full command rate matters more than foresight.
        # HYSTERESIS (P15-c): the speed oscillates around the switch point
        # when accelerating out of corners; without a gap the horizon (and
        # the v_path_cap it scales) flipped every few cycles -- 24 flips in
        # one run, each a steering + speed-cap transient.
        # P19-a: FAR needs ENOUGH UNSATURATED reference, not just speed. A short
        # spline (camera-limited chains are only ~13-15 m at spawn) clamps the
        # arc-length reference at s_max for most of the 2 s rollout. Measured
        # 2026-08-26 run: switching FAR mid-acceleration there produced an
        # instant right-steer transient (-0.14 rad held ~1.2 s at 5 m/s) that
        # threw the car off a straight reference at 25 deg/s. Gate BOTH
        # directions on the previous cycle's smoothed path-ahead.
        far_ok = self._ahead_lp >= self.FAR_MIN_AHEAD_M
        if self.horizon == self.T_FAR:
            horizon = self.T_FAR if (_speed >= self.T_SWITCH_SPEED - 0.3 and far_ok) else self.T_NEAR
        else:
            horizon = self.T_FAR if (_speed >= self.T_SWITCH_SPEED + 0.3 and far_ok) else self.T_NEAR
        self.horizon = horizon

        # ---- yaw rate: finite difference + low-pass ----
        if self.prev_yaw is not None:
            raw_yr = _angle_diff(yaw, self.prev_yaw) / dt_ros
            self.yaw_rate += 0.3 * (raw_yr - self.yaw_rate)
        self.prev_yaw = yaw

        # ---- closest path point + crosstrack ----
        self.cur, s_cur, cte_val, hd_ref = project_reference(
            x, y, yaw, pd["xy_np"], pd["s_np"], self._progress_s,
            forward_window=max(2.0, 1.0 + _speed * dt_ros))
        self._progress_s = s_cur
        ct_abs = abs(cte_val)

        # ---- bad-path guard ----
        # One-sided cones make PP emit full-length but WRONG paths (~100 deg off).
        # Geometry alone also false-positives legit hairpins, so the flag requires
        # BOTH: heading wildly off AND one-sided cone evidence.
        heading_err = _angle_diff(yaw, hd_ref)
        _hd_off     = abs(heading_err)
        hd_bad      = (_hd_off > self.HD_BAD_RAD)
        path_is_bad = hd_bad and bool(one_sided)

        # Bail out early on a garbage path WITHOUT touching MPPI memory
        # (nom warm start, _vr_lp, I_spd) so handback resumes from the last
        # good plan. control_node's recovery takes the returned command over.
        if path_is_bad:
            audit = {}
            if self.audit_enabled:
                audit["spline"] = np.column_stack((pd["rx_np"], pd["ry_np"])).tolist()
            return 0.0, 0.0, {
                "target_speed": 0.0,
                "cte":          cte_val,
                "heading_err":  heading_err,
                "mean_traj":    [],
                "path_bad":     True,
                "hd_bad":       bool(hd_bad),
                "audit":        audit,
            }

        # ---- speed target: corner profile, then caps, then rate limit ----
        s_max = float(pd["s_np"][-1])
        lookahead_eff = min(self.LOOKAHEAD_M, 0.5 * max(s_max - s_cur, 0.0))
        path_ahead = max(s_max - s_cur - lookahead_eff, 0.5)   # m of usable path

        # Smooth path-ahead: PP path length jitters frame to frame; a running
        # low-pass ignores transient dips but still slows on sustained shortage.
        self._ahead_lp += 0.25 * (path_ahead - self._ahead_lp)
        ahead_eff = self._ahead_lp
        # P13: sustained exhaustion -> control_node latches creep mode.
        exhausted = bool(ahead_eff <= self.STOP_PATH_M)
        v_path_cap = ahead_eff / (self.HORIZON_CAP_FRAC * horizon * self.DT_EFF)

        # CTE reduction (slow when off-line) on top of the corner profile.
        scale    = float(np.clip(1.0 - self.K_CT * min(ct_abs / self.CT0, 1.0), 0.0, 1.0))
        vt_np    = pd.get("vt_np")
        v_corner = float(np.interp(s_cur, pd["s_np"], vt_np))
        vr_uncapped = v_corner * scale
        v_brake_cap = math.sqrt(2.0 * self.BRAKE_A * max(path_ahead, 0.0))
        vr_raw = min(vr_uncapped, v_path_cap, v_brake_cap)
        # P19-b/P19-c: launch governor -- hold target speed down until the
        # rolling buffer has matured ONCE (ahead_eff >= LAUNCH_RELEASE_M).
        # Latched: corners with transiently short chains are NOT re-governed
        # after maturity. FAR engagement stays gated separately on
        # FAR_MIN_AHEAD_M (see horizon block above).
        if not self.launch_done:
            if ahead_eff >= self.LAUNCH_RELEASE_M:
                self.launch_done = True
                print(f"[MPPI] path matured ({ahead_eff:.1f} m ahead) -> "
                      f"launch governor released", flush=True)
            else:
                vr_raw = min(vr_raw, self.LAUNCH_CAP_MPS)
        if speed_cap is not None:              # P14-c: crawl on pseudo bridges
            vr_raw = min(vr_raw, float(speed_cap))
        if path_ahead <= self.STOP_PATH_M:
            vr_raw = 0.0                       # don't drive blind past the path end
        if path_is_bad:
            vr_raw = 0.0
        vr_raw = float(np.clip(vr_raw, 0.0, self.V_MAX))
        clearance_audit = {}

        # Rate-limit: slow to slow down (kills stop-start stutter), quick up.
        if vr_raw < self._vr_lp:
            self._vr_lp += 0.15 * (vr_raw - self._vr_lp)
        else:
            self._vr_lp += 0.6 * (vr_raw - self._vr_lp)
        if speed_cap is not None:
            # Apply the bridge ceiling to the filtered target as well as its input.
            self._vr_lp = min(self._vr_lp, float(speed_cap))
        vr_eff = float(self._vr_lp)
        if self.clearance_speed is not None:
            clearance_cap = self.clearance_speed.cap_mps
            clearance_audit.update(clearance_speed_base_target_mps=vr_raw,
                                   clearance_speed_cap_mps=clearance_cap)
            # Keep the original profiler's filter state independent of the add-on.
            vr_raw = min(vr_raw, clearance_cap)
            vr_eff = min(vr_eff, clearance_cap)

        # ---- arc-length reference horizon (marches at the planning speed) ----
        steps = torch.arange(horizon, dtype=torch.float32, device=dev)

        def stage_reference(target_speed):
            pace = max(target_speed, 0.5) * self.DT_EFF
            s_tgt = torch.clamp(s_cur + lookahead_eff + steps * pace, 0.0, s_max)
            knots = pd["s"]
            self._refE[:horizon].copy_(_interp_1d(knots, pd["rx"], s_tgt))
            self._refN[:horizon].copy_(_interp_1d(knots, pd["ry"], s_tgt))
            self._nE[:horizon].copy_(_interp_1d(knots, pd["nE"], s_tgt))
            self._nN[:horizon].copy_(_interp_1d(knots, pd["nN"], s_tgt))
            self._ff[:horizon].copy_(_interp_1d(knots, pd["ff"], s_tgt))
            self._vtgt[:horizon].copy_(_interp_1d(knots, pd["vt"], s_tgt).clamp(max=target_speed))
            return pace

        # ---- steer samples: sustained per-sample bias + fine jitter ----
        model_actuator = self.steering_actuator_enabled and steering_prediction is not None
        prefix_steps, hold_steps = 0, 1
        if model_actuator:
            wheel_angle, queued, solve_s, measured_wheel = steering_prediction
            prefix_steps = len(queued)
            hold_steps = max(1, round(solve_s / self.DT_EFF))
            self._set_steering_schedule(wheel_angle, queued, solve_s)

        use_elapsed = self.elapsed_warm_start and steering_state_s is not None
        elapsed_steps = None
        if use_elapsed:
            candidate_start_s = steering_state_s + prefix_steps * self.DT_EFF
            nominal, elapsed_steps = self._warm_start(candidate_start_s, hold_steps)
        else:
            nominal = self.nom
        bias = self.SIG_BIAS  * torch.randn(self.K, 1, device=dev, generator=self._rng)
        jit  = self.SIG_STEER * torch.randn(self.K, self.T_MAX, device=dev, generator=self._rng)
        eps  = bias + jit
        A    = (nominal + eps).clamp(-self.STEER_MAX, self.STEER_MAX)
        # When clearance checking is active, score the clipped first request
        # that will actually be sent to the simulator.
        smax0 = float(np.clip(
            self.STEER_MAX - (self.STEER_MAX - 0.14) * (_speed - 13.0) / 5.0,
            0.14, self.STEER_MAX))
        if self.cone_clearance:
            A[:, 0].clamp_(-smax0, smax0)

        # ---- fill static buffers (slice to the ACTIVE horizon: buffers are
        # T_MAX wide, the interp outputs are `horizon` wide, and the rollout
        # only ever reads [0, horizon)) ----
        self._A.copy_(A)
        pace_m = stage_reference(vr_eff)
        # Heading slots use the TRAINING convention: yaw_sin=-cos(yaw), yaw_cos=sin(yaw)
        self._x0.copy_(torch.tensor(
            [v_plan, 0.0, self.yaw_rate, -math.cos(yaw), math.sin(yaw), 0.0],
            dtype=torch.float32, device=dev))
        self._p0.copy_(torch.tensor([x, y], dtype=torch.float32, device=dev))

        # ---- rollout (CUDA graph replay, or plain call on CPU) ----
        roll_ms = self._run_rollout_batch(horizon, model_actuator)
        cost = self._cost

        # ---- softmax -> updated nominal (adaptive temperature) ----
        # Fixed lambda degenerates when the cost spread is tiny (all samples
        # equal-weighted -> MPPI = noise). Scale it to the actual spread.
        blocked = False
        crawl = False
        candidates = None
        if self.cone_clearance:
            clearance_start = time.perf_counter()
            retry_roll_ms = 0.0
            initial_feasible_pct = None
            # Use the deployed state bounds over the full horizon, including the footprint.
            reach = math.hypot(13.902, 1.173) * horizon * self.DT_EFF + 3.0
            cone_data = nearby_cones(() if clearance_cones is None else clearance_cones, x, y, reach)
            clearance_audit["clearance_cone_count"] = len(cone_data)
            blocked = clearance_cones is None
            if len(cone_data):
                cones = torch.as_tensor(cone_data, device=dev)
                initial_pose = torch.as_tensor([x, y, yaw], dtype=self._trajE.dtype, device=dev).view(1, 1, 3)
                margins, candidates, infeasible = self._screen_rollouts(horizon, initial_pose, cones)
                initial_feasible_pct = 100.0 * len(candidates) / self.K
                crawl_target = min(CLEARANCE_CRAWL_MPS, vr_raw, vr_eff)
                if (self.clearance_speed is not None and not len(candidates) and infeasible
                        and crawl_target > 0.0 and not exhausted):
                    if vr_eff > crawl_target:
                        # Reuse the measured state, queued steering and candidate bank.
                        vr_eff = vr_raw = crawl_target
                        pace_m = stage_reference(vr_eff)
                        retry_roll_ms = self._run_rollout_batch(horizon, model_actuator)
                        roll_ms += retry_roll_ms
                        margins, candidates, infeasible = self._screen_rollouts(horizon, initial_pose, cones)
                        clearance_audit.update(clearance_retry_target_mps=vr_eff,
                                               clearance_retry_roll_ms=retry_roll_ms)
                    crawl = not len(candidates) and infeasible > 0
                clearance_audit["clearance_infeasible_pct"] = 100.0 * infeasible / self.K
                clearance_audit["clearance_feasible_count"] = len(candidates)
                clearance_audit["clearance_invalid_count"] = self.K - len(candidates) - infeasible
                clearance_audit["clearance_best_margin_m"] = float(
                    margins.masked_fill(~torch.isfinite(margins), -math.inf).max())
                blocked = not len(candidates)
            else:
                clearance_audit["clearance_infeasible_pct"] = None if blocked else 0.0
                if not blocked:
                    candidates = torch.nonzero(torch.isfinite(cost)).flatten()
                    clearance_audit["clearance_feasible_count"] = len(candidates)
                    clearance_audit["clearance_invalid_count"] = self.K - len(candidates)
                    blocked = not len(candidates)
            feasible_pct = None if candidates is None else 100.0 * len(candidates) / self.K
            if initial_feasible_pct is None:
                initial_feasible_pct = feasible_pct
            clearance_audit["clearance_feasible_pct"] = feasible_pct
            clearance_audit["clearance_initial_feasible_pct"] = initial_feasible_pct
            clearance_audit["clearance_rollout_target_mps"] = vr_eff
            if self.clearance_speed is not None:
                # A successful slower retry does not cancel the original warning.
                self.clearance_speed.observe(initial_feasible_pct, dt_ros)
                if crawl or (retry_roll_ms and not blocked):
                    self.clearance_speed.cap_mps = min(self.clearance_speed.cap_mps, CLEARANCE_CRAWL_MPS)
                clearance_audit["clearance_speed_next_cap_mps"] = self.clearance_speed.cap_mps
            if crawl:
                candidates = torch.nonzero(torch.isfinite(margins) & torch.isfinite(cost)).flatten()
                blocked = False

        weighted = None
        n_eff, lam = 0.0, None
        top_trajs = []
        if blocked:
            # An infeasible batch has no steering distribution to average.
            nom_new = torch.zeros_like(self.nom[0])
        else:
            scored_cost = cost if candidates is None else cost[candidates]
            c_lo = scored_cost.min()
            c_spread = float(scored_cost.max() - c_lo)
            lam = max(self.LAMBDA_FRAC * c_spread, self.LAMBDA_MIN)
            logits = -(scored_cost - c_lo) / lam
            if self.cone_clearance and len(cone_data):
                logits = logits - clearance_log_penalty(margins[candidates])
            if crawl:
                # Use an evaluated sample so averaging cannot create a worse overlap.
                best = candidates[logits.argmax()]
                nom_new = A[best]
                n_eff = 1.0
                clearance_audit.update(clearance_decision="crawl",
                                       clearance_selected_margin_m=float(margins[best]))
            else:
                wts = torch.exp(logits)
                wts = wts / wts.sum()
                scored_actions = A if candidates is None else A[candidates]
                weighted = (wts.unsqueeze(1) * scored_actions).sum(dim=0)
                nom_new = weighted
                if self.cone_clearance and len(cone_data):
                    margin = self._check_selected(horizon, model_actuator, weighted.unsqueeze(0), initial_pose, cones)
                    clearance_audit["clearance_weighted_margin_m"] = margin
                    if not margin > 0:
                        best = candidates[logits.argmax()]
                        nom_new = A[best]
                        margin = float(margins[best])
                        clearance_audit["clearance_decision"] = "best_feasible"
                    clearance_audit["clearance_selected_margin_m"] = margin
                n_eff = float(1.0 / (wts * wts).sum())
            top = torch.topk(logits, k=min(10, len(logits))).indices
            if candidates is not None:
                top = candidates[top]
            for idx in top.tolist():
                top_trajs.append(list(zip(self._trajE[:horizon:5, idx].tolist(),
                                          self._trajN[:horizon:5, idx].tolist())))
        if self.cone_clearance:
            clearance_audit.setdefault("clearance_decision",
                                       "missing_cone_map" if clearance_cones is None else
                                       "brake" if blocked else "weighted" if len(cone_data) else "no_nearby_cones")
            clearance_audit["clearance_ms"] = (time.perf_counter() - clearance_start) * 1000.0 - retry_roll_ms
        self.nom = nom_new.unsqueeze(0)

        # ---- diagnostics ----
        cost_min = float(cost.min())
        cost_max = float(cost.max())
        a0_std   = float(A[:, 0].std())             # sample diversity at t=0

        # ---- first action: feedforward + MPPI correction ----
        ff0       = float(self._ff[0])
        ff_part   = self.K_FF * ff0
        mppi_part = float(nom_new[0])
        steer_out = float(np.clip(ff_part + mppi_part, -smax0, smax0))

        if os.environ.get("MPPI_FORCE_STEER0") == "1":
            steer_out = 0.0

        if use_elapsed:
            available_steps = horizon - prefix_steps
            scored_steps = max(1, available_steps - hold_steps + 1) if available_steps > 0 else 0
            if scored_steps:
                self._previous_steering_plan = candidate_start_s, hold_steps, available_steps, scored_steps
            else:
                self.reset_steering_plan()
        else:
            self.nom = torch.cat([self.nom[:, 1:], self.nom[:, -1:]], dim=1)

        # ---- PI longitudinal with anti-windup ----
        if blocked:
            self.reset_steering_plan()
            self.I_spd = 0.0
            vr_eff = 0.0
        err  = vr_eff - _speed
        raw  = self.KP * err + self.KI * self.I_spd
        lon0 = float(np.clip(raw, -1.0, 1.0))
        # Integrate only when unsaturated AND near the setpoint; integrating
        # through transients wound up a bias that resisted braking into corners.
        saturated = (raw >= 1.0 and err > 0.0) or (raw <= -1.0 and err < 0.0)
        if not blocked and not saturated and abs(err) < self.I_BAND:
            self.I_spd = float(np.clip(self.I_spd + err * dt_ros, -self.I_CLAMP, self.I_CLAMP))
        accel_out = lon0 * self.ACCEL_MAX

        # ---- telemetry ----
        mean_traj   = self._viz_kinematic(x, y, yaw, steer_out, _speed)

        self._dbg_ctr += 1
        if self._dbg_ctr % 20 == 1:
            print(f"[MPPI] t={roll_ms:4.0f}ms odom={speed:5.2f} use={_speed:5.2f} "
                  f"vplan={v_plan:4.1f} vr={vr_eff:4.1f} vcap={v_path_cap:4.1f} vcrnr={v_corner:4.1f} look={lookahead_eff:3.1f} | "
                  f"steer={steer_out:+.3f} ff0={ff0:+.3f} accel={accel_out:+.2f} lon0={lon0:+.2f} | "
                  f"cur={self.cur}/{pd['N']} ahead={path_ahead:4.1f}m CTE={cte_val:+.2f} "
                  f"hd_err={float(heading_err):+.2f} | "
                  f"cost[{cost_min:.2f},{cost_max:.2f}] nEff={n_eff:.0f}/{self.K} "
                  f"lam={lam or 0.0:.3f} a0std={a0_std:.3f} I={self.I_spd:.2f}",
                  flush=True)

        # ---- optional CSV trace ----
        if self._csv is not None:
            clearance_csv = []
            for name in ("clearance_infeasible_pct", "clearance_weighted_margin_m", "clearance_selected_margin_m",
                         "clearance_feasible_pct", "clearance_speed_cap_mps", "clearance_speed_next_cap_mps",
                         "clearance_speed_base_target_mps", "clearance_initial_feasible_pct",
                         "clearance_retry_target_mps", "clearance_retry_roll_ms", "clearance_rollout_target_mps"):
                value = clearance_audit.get(name)
                clearance_csv.append(f"{value:.6f}" if value is not None and math.isfinite(value) else "")
            self._csv.write(
                f"{time.time():.3f},{x:.3f},{y:.3f},{yaw:.4f},{speed:.3f},{_speed:.3f},"
                f"{v_plan:.3f},{vr_eff:.3f},{steer_out:.4f},{accel_out:.3f},{lon0:.3f},"
                f"{cte_val:.3f},{float(heading_err):.4f},{cost_min:.3f},{cost_max:.3f},"
                f"{n_eff:.1f},{a0_std:.4f},{pace_m:.4f},{roll_ms:.1f},{self.I_spd:.3f},"
                f"{self.yaw_rate:.4f},{self.cur},{pd['N']},{ff_part:.4f},{mppi_part:.4f},"
                + ",".join(clearance_csv) + "\n")
            self._csv.flush()

        audit = {}
        if self.audit_enabled:
            audit = {
                **clearance_audit,
                "reference": torch.stack((self._refE[:horizon:5], self._refN[:horizon:5]), dim=-1).tolist(),
                "spline": np.column_stack((pd["rx_np"], pd["ry_np"])).tolist(),
                "rollouts": top_trajs,
                "weighted_steer_sequence": weighted[:horizon].tolist() if weighted is not None else [],
                "rollout_speed_targets": self._vtgt[:horizon].tolist(),
                "rollout_dt": 5 * self.DT_EFF,
                "rollout_first_dt": self.DT_EFF,
                "horizon": horizon,
                "roll_ms": roll_ms,
                "cost_min": cost_min,
                "cost_max": cost_max,
                "temperature": lam,
                "a0_std": a0_std,
                "I_spd": self.I_spd,
                "raw_target_speed": vr_raw,
                "corner_speed": v_corner,
                "speed_cap": speed_cap,
                "path_speed_cap": v_path_cap,
                "braking_speed_cap": v_brake_cap,
                "path_ahead": path_ahead,
                "lookahead_m": lookahead_eff,
                "v_plan": v_plan,
                "yaw_rate_seed": self.yaw_rate,
                "steering_warm_start_elapsed_steps": elapsed_steps,
                "cur": self.cur,
                "N": pd["N"],
            }
            if crawl or (weighted is not None and nom_new is not weighted):
                audit["selected_steer_sequence"] = nom_new[:horizon].tolist()
            if model_actuator:
                audit.update(wheel_angle_seed_rad=wheel_angle,
                             wheel_seed_from_feedback=measured_wheel,
                             steering_publish_delay_estimate_s=solve_s,
                             steering_prefix_steps=len(queued))

        return steer_out, accel_out, {
            "audit": audit,
            "target_speed":   vr_eff,
            "cte":            cte_val,
            "heading_err":    float(heading_err),
            "mean_traj":      mean_traj,
            "path_bad":       bool(path_is_bad),
            "clearance_blocked": blocked,
            "clearance_crawl": crawl,
            "hd_bad":         bool(hd_bad),
            "path_exhausted": exhausted,
            "ahead_eff":      float(ahead_eff),   # P14: pseudo-path trigger signal
            "top_trajs":      top_trajs,          # P14-c: 10 best rollouts, best first
            "n_eff":          n_eff,
            "actual_speed":   _speed,
            "ff_part":        ff_part,
            "mppi_part":      mppi_part,
        }

    # =====================================================================
    #  Cheap kinematic viz trajectory (no NN) — for the control_node plot
    # =====================================================================
    def _viz_kinematic(self, x0, y0, yaw0, steer_val, speed_val):
        pts = [(x0, y0)]
        L = self.L_WB_KIN
        v = max(float(speed_val), 1.0)
        for t in range(min(self.horizon, 40)):
            yr = v * math.tan(float(steer_val)) / L
            yaw_mid = float(yaw0) + 0.5 * yr * self.DT_EFF
            x0 = x0 + v * math.cos(yaw_mid) * self.DT_EFF
            y0 = y0 + v * math.sin(yaw_mid) * self.DT_EFF
            yaw0 = float(yaw0) + yr * self.DT_EFF
            if t % 8 == 0:
                pts.append((x0, y0))
        return pts
