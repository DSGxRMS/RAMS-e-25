#!/usr/bin/env python3
"""
mppi_controller.py  —  exp09 Neural ODE MPPI for RAMS-e-25 ROS2 stack.

Ports the working MATLAB implementation (fs_mppi_init.m + fs_mppi_step.m) into
Python / PyTorch so it can run in the EUFS Gazebo sim.

Key differences from the MATLAB version:
  - Path comes as a live rolling window from the PP node (not a pre-loaded 2624-pt track).
    Speed profile and feedforward curvature are derived from the live window each update.
  - State is bridged from EUFS odometry (x, y, yaw, speed) to the 6-D Neural ODE state
    [Vx, Vy, yawRate, yaw_sin, yaw_cos, slip].
  - GPU (CUDA) is used for the K=1024, T=200 rollout batch exactly as in MATLAB.

Author: Soumil (ported from MATLAB MPPI by the RMS controls team)
"""

import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Reuse helpers from the existing controls package
from controls.control_utils import (
    compute_signed_curvature,
    generate_velocity_profile,
    local_closest_index,
    cross_track_error,
    path_heading,
)

# ============================================================
# Paths
# ============================================================
_HERE = Path(__file__).parent
CKPT_PATH = _HERE / "fs_model" / "best_model_600.pt"

# ============================================================
# Neural ODE model (exact copy of export_fs_to_matlab.py / GB)
# ============================================================
VX_I, VY_I, YR_I, SIN_I, COS_I, SLIP_I = 0, 1, 2, 3, 4, 5


class _Func(nn.Module):
    """8 -> 128 -> 128 -> 128 -> 4  (SiLU activations)."""
    def __init__(self, sd=6, cd=2, hd=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sd + cd, hd), nn.SiLU(),
            nn.Linear(hd, hd),      nn.SiLU(),
            nn.Linear(hd, hd),      nn.SiLU(),
            nn.Linear(hd, 4),
        )

    def forward(self, x, u, xm, xs, um, us):
        g = self.net(torch.cat([(x - xm) / xs, (u - um) / us], dim=-1))
        yr = x[..., YR_I]
        ys = x[..., SIN_I]
        yc = x[..., COS_I]
        # grey-box: exact heading kinematics + NN residuals on [dVx,dVy,dyawRate,dslip]
        return torch.stack([
            g[..., 0] * xs[VX_I],           # dVx
            g[..., 1] * xs[VY_I],           # dVy
            g[..., 2] * xs[YR_I],           # dyawRate
            yc * yr,                         # dyaw_sin (exact)
            -ys * yr,                        # dyaw_cos (exact)
            g[..., 3] * xs[SLIP_I],         # dslip
        ], dim=-1)


def _rk4(f, x, u, h, xm, xs, um, us):
    k1 = f(x, u, xm, xs, um, us)
    k2 = f(x + 0.5 * h * k1, u, xm, xs, um, us)
    k3 = f(x + 0.5 * h * k2, u, xm, xs, um, us)
    k4 = f(x + h * k3, u, xm, xs, um, us)
    return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class _GB(nn.Module):
    """Grey-box Neural ODE wrapper — matches GB class in export_fs_to_matlab.py."""
    def __init__(self, ck, hd=128):
        super().__init__()
        self.func = _Func(6, 2, hd)
        self.nr = int(ck["num_rk4_steps"])
        for name in ("x_mean", "x_std", "u_mean", "u_std", "x_min", "x_max"):
            self.register_buffer(name, torch.as_tensor(ck[name], dtype=torch.float32))

    def step(self, x, u, dt):
        """
        x  : (..., 6)  state batch
        u  : (..., 2)  control batch
        dt : scalar    integration step (seconds)
        """
        h = dt / self.nr
        for _ in range(self.nr):
            x = _rk4(self.func, x, u, h,
                      self.x_mean, self.x_std, self.u_mean, self.u_std)
        # renormalise yaw_sin / yaw_cos
        a = x[..., SIN_I]
        b = x[..., COS_I]
        nrm = torch.sqrt(a * a + b * b).clamp_min(1e-6)
        x = x.clone()
        x[..., SIN_I] = a / nrm
        x[..., COS_I] = b / nrm
        return torch.clamp(x, self.x_min, self.x_max)


# ============================================================
# Angle utilities
# ============================================================

def _angle_diff(a, b):
    """Wrapped difference a - b in [-pi, pi].  Works on torch tensors or scalars."""
    d = a - b
    if isinstance(d, torch.Tensor):
        return torch.atan2(torch.sin(d), torch.cos(d))
    return math.atan2(math.sin(d), math.cos(d))


def _world_vel(x):
    """
    Convert body-frame velocities to world-frame using stored yaw_sin/yaw_cos.
    x : (6, K)  — state batch, columns = samples
    Returns Ve (1, K), Vn (1, K).
    Note the sign convention from MATLAB wv():
        Ve = -(Vx*yaw_sin + Vy*yaw_cos)
        Vn =  (Vx*yaw_cos - Vy*yaw_sin)
    """
    vx = x[VX_I, :]
    vy = x[VY_I, :]
    sn = x[SIN_I, :]
    cs = x[COS_I, :]
    # Standard ROS ENU rotation: yaw=0 means East, CCW positive.
    # MATLAB wv() used a North-reference convention — that must NOT be used here
    # because the EUFS SLAM yaw is standard ENU (CCW from East/x-axis).
    Ve = vx * cs - vy * sn   # = Vx*cos(yaw) - Vy*sin(yaw)
    Vn = vx * sn + vy * cs   # = Vx*sin(yaw) + Vy*cos(yaw)
    return Ve, Vn


# ============================================================
# MPPIController
# ============================================================

class MPPIController:
    """
    Steer-only MPPI with exp09 Neural ODE rollout model.

    Mirrors fs_mppi_init.m / fs_mppi_step.m exactly, adapted for the live
    rolling path window from the ROS2 PP node.
    """

    # ---- MPPI hyper-parameters (match fs_mppi_init.m) ----
    K          = 1024       # number of samples
    T          = 200        # horizon steps
    DT         = 0.005      # integration dt (matches training)
    LAMBDA     = 1.0        # temperature
    SIG_STEER  = 0.08       # steer noise std (rad)
    STEER_MAX  = 0.349      # 20 deg — model training range
    W_CT       = 1.0        # Huber crosstrack weight
    W_HEAD     = 0.3        # heading weight
    CT_SAT     = 5.0        # Huber knee (m)
    K_FF       = 0.85       # feedforward gain
    LOOKAHEAD  = 8          # path index preview offset
    KP_ROLL    = 0.40       # P-only lon gain inside rollout
    KP         = 0.35       # PI lon kp (actual output)
    KI         = 0.10       # PI lon ki
    I_CLAMP    = 5.0        # integral clamp
    V_FLOOR    = 1.0        # min speed target (m/s)  — EUFS runs at 2 m/s max
    CT0        = 4.0        # CTE threshold for speed reduction
    K_CT       = 0.7        # CTE speed reduction gain
    WIN        = 40         # window for nearest-point search
    L_WB       = 1.535      # wheelbase (m)  — same FS car
    ACCEL_MAX  = 3.0        # lon[-1,1] -> accel m/s² scaling
    V_MAX_PATH = 2.0        # cap on generated speed profile (m/s) — EUFS safe limit

    def __init__(self, ckpt_path=None):
        # ---- device ----
        if torch.cuda.is_available():
            self.dev = torch.device("cuda")
            d = torch.cuda.get_device_properties(0)
            print(f"[MPPI] GPU: {d.name}  ({d.total_memory/1e9:.1f} GB)")
        else:
            self.dev = torch.device("cpu")
            print("[MPPI] WARNING: CUDA not available, falling back to CPU. "
                  "Performance will be degraded. Consider K=128, T=50 on CPU.")

        # ---- load model ----
        cp = ckpt_path or CKPT_PATH
        ck = torch.load(str(cp), map_location=self.dev, weights_only=False)
        hd = int(ck.get("hidden_dim", 128))
        self.model = _GB(ck, hd).to(self.dev)
        self.model.eval()
        print(f"[MPPI] Loaded {Path(cp).name}  "
              f"(hidden={hd}, nr={self.model.nr}, "
              f"controls={list(ck.get('control_cols', ['steer','lon']))})")

        # ---- nominal sequence (warm-start) ----
        self.nom = torch.zeros(1, self.T, device=self.dev)   # 1xT

        # ---- longitudinal PI state ----
        self.I_spd   = 0.0
        self.vtgt0   = self.V_FLOOR

        # ---- path cache ----
        self._path_key  = None    # (len, first_pt, last_pt) tuple
        self._path_data = None    # dict of precomputed GPU tensors

        # ---- progress tracker ----
        self.cur = 0

        # ---- state bridge ----
        self.prev_yaw  = None
        self.yaw_rate  = 0.0      # low-passed finite-difference estimate

        # ---- last output (for fallback / decay) ----
        self._last_steer = 0.0

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def update_path(self, path_pts):
        """
        Call every control loop with the latest list[(x,y)] from node.get_path().
        Recomputes cached tensors only when the path actually changes.
        """
        if not path_pts or len(path_pts) < 5:
            return

        # cheap change-detection: length + endpoints
        key = (len(path_pts), path_pts[0], path_pts[-1])
        if key == self._path_key:
            return
        self._path_key = key

        pts = np.array(path_pts, dtype=np.float64)   # (N, 2)
        rx, ry = pts[:, 0], pts[:, 1]
        N = len(rx)

        # headings
        dx = np.gradient(rx)
        dy = np.gradient(ry)
        heading = np.arctan2(dy, dx)

        # left normals (nE, nN) from heading
        nE = -np.sin(heading)
        nN =  np.cos(heading)

        # feedforward curvature -> steer (C3)
        kappa = compute_signed_curvature(rx, ry)
        ff_steer = np.arctan(self.L_WB * kappa)

        # speed profile from curvature, capped at V_MAX_PATH
        try:
            vt = generate_velocity_profile(rx, ry)
            vt = np.clip(vt, self.V_FLOOR, self.V_MAX_PATH)
        except Exception:
            vt = np.full(N, self.V_FLOOR)

        # move everything to GPU
        dev = self.dev
        self._path_data = {
            "N":        N,
            "rx":       torch.tensor(rx,       dtype=torch.float32, device=dev),
            "ry":       torch.tensor(ry,       dtype=torch.float32, device=dev),
            "nE":       torch.tensor(nE,       dtype=torch.float32, device=dev),
            "nN":       torch.tensor(nN,       dtype=torch.float32, device=dev),
            "ff":       torch.tensor(ff_steer, dtype=torch.float32, device=dev),
            "vt":       torch.tensor(vt,       dtype=torch.float32, device=dev),
            "heading":  torch.tensor(heading,  dtype=torch.float32, device=dev),
            # numpy copies for CPU helpers
            "rx_np":    rx,
            "ry_np":    ry,
        }

        # Reset progress tracker when path changes
        self.cur = 0
        print(f"[MPPI] Path updated: {N} pts, "
              f"vt=[{vt.min():.1f},{vt.max():.1f}] m/s")

    def compute(self, x, y, yaw, speed, dt_ros):
        """
        Run one MPPI plan step.

        Parameters
        ----------
        x, y    : float  global position (m)
        yaw     : float  heading (rad)
        speed   : float  |velocity| (m/s)
        dt_ros  : float  real sim-clock dt since last call (s)

        Returns
        -------
        steer_out : float   steering command (rad), clamped to STEER_MAX
        accel_out : float   acceleration command (m/s²), clamped to ACCEL_MAX
        info      : dict    target_speed, cte, heading_err, mean_traj
        """
        # ---- fallback when no valid path: steer straight and brake ----
        if self._path_data is None:
            return 0.0, -self.ACCEL_MAX, {
                "target_speed": 0.0,
                "cte": 0.0, "heading_err": 0.0,
                "mean_traj": [],
            }

        pd   = self._path_data
        N    = pd["N"]
        dev  = self.dev

        # ---- state bridge: odom -> 6D Neural ODE state ----
        # yawRate via finite-difference on heading, low-pass filtered
        if self.prev_yaw is not None and dt_ros > 1e-4:
            raw_yr = _angle_diff(yaw, self.prev_yaw) / dt_ros
            alpha  = 0.3
            self.yaw_rate = alpha * raw_yr + (1 - alpha) * self.yaw_rate
        self.prev_yaw = yaw

        # yaw_sin = sin(yaw), yaw_cos = cos(yaw) — ENU convention
        # Neural ODE grey-box: yaw_sin stored at SIN_I=3, yaw_cos at COS_I=4
        # deriv: d(yaw_sin)/dt = cos(yaw)*yawRate = yaw_cos*yawRate  (SIN_I)
        #        d(yaw_cos)/dt = -sin(yaw)*yawRate = -yaw_sin*yawRate (COS_I)
        x6 = torch.tensor(
            [speed, 0.0, self.yaw_rate, math.sin(yaw), math.cos(yaw), 0.0],
            dtype=torch.float32, device=dev,
        )  # (6,)

        # ---- find current path index ----
        self.cur = local_closest_index(
            (x, y), pd["rx_np"], pd["ry_np"], self.cur, loop=False
        )
        self.cur = int(np.clip(self.cur, 0, N - 1))

        # ---- crosstrack speed scaling (same as MATLAB) ----
        cte_val, _ = cross_track_error(x, y, pd["rx_np"], pd["ry_np"], self.cur, loop=False)
        ct_abs = abs(cte_val)
        scale  = float(np.clip(1.0 - self.K_CT * min(ct_abs / self.CT0, 1.0), 0.0, 1.0))
        vr_eff = float(np.clip(pd["vt"].cpu().numpy()[self.cur] * scale, self.V_FLOOR, self.V_MAX_PATH))

        # ---- marching target indices for rollout (same as MATLAB) ----
        pace   = max(speed, 3.0) * self.DT
        lo     = self.LOOKAHEAD
        ti_np  = np.clip(
            np.round(self.cur + lo + np.arange(self.T) * pace).astype(int),
            0, N - 1
        )

        # slice reference path for the horizon
        refE  = pd["rx"][ti_np]       # (T,)
        refN  = pd["ry"][ti_np]       # (T,)
        nE_t  = pd["nE"][ti_np]       # (T,)
        nN_t  = pd["nN"][ti_np]       # (T,)
        vtgt  = pd["vt"][ti_np]       # (T,)
        ff    = pd["ff"][ti_np]       # (T,)  feedforward steer

        # ---- sample steer perturbations ----
        eps = self.SIG_STEER * torch.randn(self.K, self.T, device=dev)
        A   = (self.nom + eps).clamp(-self.STEER_MAX, self.STEER_MAX)   # (K, T)

        # ---- batch rollout ----
        xk   = x6.unsqueeze(1).expand(6, self.K).clone()    # (6, K)
        pk   = torch.tensor([[x], [y]], dtype=torch.float32, device=dev).expand(2, self.K).clone()
        cost = torch.zeros(self.K, device=dev)

        with torch.no_grad():
            for t in range(self.T):
                vx = xk[VX_I, :]                             # (K,)

                # speed-dependent steer cap (MATLAB line 47)
                smax = (self.STEER_MAX
                        - (self.STEER_MAX - 0.14) * (vx - 13.0) / 5.0
                        ).clamp(0.14, self.STEER_MAX)         # (K,)

                # C3: feedforward + sampled feedback
                steer = (self.K_FF * ff[t] + A[:, t]).clamp(-smax, smax)  # (K,)

                # P-only lon in rollout
                lon = (self.KP_ROLL * (vtgt[t] - vx)).clamp(-1.0, 1.0)   # (K,)

                # control tensor (2, K)
                u_batch = torch.stack([steer, lon], dim=0)

                # world velocities before step
                Ve1, Vn1 = _world_vel(xk)

                # Neural ODE step — model.step expects (..., 6) and (..., 2)
                # Transpose to (K, 6) and (K, 2) for batch dim
                xk_T  = xk.T.unsqueeze(0)   # (1, K, 6)  — use batch trick
                u_T   = u_batch.T.unsqueeze(0)  # (1, K, 2)
                xk_T  = self.model.step(xk_T, u_T, torch.tensor(self.DT, device=dev))
                xk    = xk_T.squeeze(0).T      # back to (6, K)

                # world velocities after step
                Ve2, Vn2 = _world_vel(xk)

                # midpoint position integration
                pk[0] += 0.5 * (Ve1 + Ve2) * self.DT
                pk[1] += 0.5 * (Vn1 + Vn2) * self.DT

                # C1: Huber crosstrack
                perp = ((pk[0] - refE[t]) * nE_t[t]
                      + (pk[1] - refN[t]) * nN_t[t])         # (K,)
                ap   = perp.abs()
                ctc  = (ap.clamp(max=self.CT_SAT) ** 2
                      + 2 * self.CT_SAT * (ap - self.CT_SAT).clamp(min=0.0))

                # C2: pure-pursuit heading — yaw_sin=sin(yaw), yaw_cos=cos(yaw), ENU
                carTh = torch.atan2(xk[SIN_I, :], xk[COS_I, :])   # (K,)
                bear  = torch.atan2(refN[t] - pk[1], refE[t] - pk[0])
                dh    = _angle_diff(carTh, bear)                      # (K,)

                cost  += self.W_CT * ctc + self.W_HEAD * dh ** 2

        cost = cost / self.T

        # ---- softmax weighting ----
        wts  = torch.exp(-(cost - cost.min()) / self.LAMBDA)
        wts  = wts / wts.sum()                                  # (K,)
        nom_new = (wts.unsqueeze(1) * A).sum(dim=0)            # (T,)  sum over K
        self.nom = nom_new.unsqueeze(0)                         # (1, T)

        # ---- apply first action ----
        smax0  = float(
            np.clip(self.STEER_MAX - (self.STEER_MAX - 0.14) * (speed - 13.0) / 5.0,
                    0.14, self.STEER_MAX)
        )
        ff0    = float(ff[0])
        steer0_raw = self.K_FF * ff0 + float(nom_new[0])
        steer_out  = float(np.clip(steer0_raw, -smax0, smax0))
        self._last_steer = steer_out

        # ---- warm-start roll ----
        # shift nominal left by 1 (next call starts from step 2 onwards)
        self.nom = torch.cat([self.nom[:, 1:], self.nom[:, -1:]], dim=1)

        # ---- PI longitudinal (actual output) ----
        vtgt0_val   = float(vr_eff)
        self.vtgt0  = vtgt0_val
        err         = vtgt0_val - speed
        self.I_spd  = float(np.clip(self.I_spd + err * dt_ros, -self.I_CLAMP, self.I_CLAMP))
        lon0        = float(np.clip(self.KP * err + self.KI * self.I_spd, -1.0, 1.0))
        accel_out   = lon0 * self.ACCEL_MAX

        # ---- telemetry: heading error + mean trajectory ----
        hd_ref      = float(path_heading(pd["rx_np"], pd["ry_np"], self.cur, loop=False))
        heading_err = _angle_diff(yaw, hd_ref)

        # mean trajectory (gather best 10% samples for viz)
        best_idx = torch.argsort(cost)[:max(1, self.K // 10)]
        # rerun a simple position rollout for those samples (cheap; no grad)
        mean_traj = self._fast_pos_rollout(x6, x, y, A[best_idx], ff, vtgt, smax0)

        info = {
            "target_speed": vtgt0_val,
            "cte":          cte_val,
            "heading_err":  float(heading_err),
            "mean_traj":    mean_traj,
        }

        return steer_out, accel_out, info

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------

    def _fast_pos_rollout(self, x6_init, px0, py0, A_sub, ff, vtgt, smax0):
        """
        Lightweight position rollout over a subset of samples for viz only.
        Returns list[(x, y)] of mean predicted positions along the horizon.
        """
        K_sub = A_sub.shape[0]
        xk    = x6_init.unsqueeze(1).expand(6, K_sub).clone()
        pk    = torch.tensor([[px0], [py0]],
                             dtype=torch.float32, device=self.dev).expand(2, K_sub).clone()
        pts   = []
        with torch.no_grad():
            for t in range(min(self.T, 50)):   # only first 50 steps for speed
                vx    = xk[VX_I, :]
                smax  = (self.STEER_MAX
                         - (self.STEER_MAX - 0.14) * (vx - 13.0) / 5.0
                         ).clamp(0.14, self.STEER_MAX)
                steer = (self.K_FF * ff[t] + A_sub[:, t]).clamp(-smax, smax)
                lon   = (self.KP_ROLL * (vtgt[t] - vx)).clamp(-1.0, 1.0)
                u_b   = torch.stack([steer, lon], dim=0)
                Ve1, Vn1 = _world_vel(xk)
                xk_T  = xk.T.unsqueeze(0)
                u_T   = u_b.T.unsqueeze(0)
                xk_T  = self.model.step(xk_T, u_T, torch.tensor(self.DT, device=self.dev))
                xk    = xk_T.squeeze(0).T
                Ve2, Vn2 = _world_vel(xk)
                pk[0] += 0.5 * (Ve1 + Ve2) * self.DT
                pk[1] += 0.5 * (Vn1 + Vn2) * self.DT
                if t % 10 == 0:
                    mx = float(pk[0].mean())
                    my = float(pk[1].mean())
                    pts.append((mx, my))
        return pts
