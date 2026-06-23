#!/usr/bin/env python3
"""
mppi_controller.py  —  exp09 Neural ODE MPPI for RAMS-e-25 ROS2 stack.

Ports the working MATLAB implementation (fs_mppi_init.m + fs_mppi_step.m) into
Python / PyTorch so it can run in the EUFS Gazebo sim.

Key differences from the MATLAB version:
  - Path comes as a live rolling window from the PP node (not a pre-loaded 2624-pt track).
    Arc-length parameterisation + GPU linear interpolation replaces index marching so the
    number/density of PP path points does not matter.
  - Speed is estimated from position derivative because SLAM odometry often omits twist.
  - T=50 (not 200) keeps each MPPI call inside the 50 ms / 20 Hz budget at 2 m/s.
    Covers the same physical arc: 2.0 m lookahead + 50*0.01 = 2.5 m total at 2 m/s.

Author: Soumil (ported from MATLAB MPPI by the RMS controls team)
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

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
# Neural ODE model  (exact copy of export_fs_to_matlab.py / GB)
# ============================================================
VX_I, VY_I, YR_I, SIN_I, COS_I, SLIP_I = 0, 1, 2, 3, 4, 5


class _Func(nn.Module):
    """8 -> 128 -> 128 -> 128 -> 4  (SiLU)."""
    def __init__(self, sd=6, cd=2, hd=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sd + cd, hd), nn.SiLU(),
            nn.Linear(hd, hd),      nn.SiLU(),
            nn.Linear(hd, hd),      nn.SiLU(),
            nn.Linear(hd, 4),
        )

    def forward(self, x, u, xm, xs, um, us):
        g  = self.net(torch.cat([(x - xm) / xs, (u - um) / us], dim=-1))
        yr = x[..., YR_I]
        ys = x[..., SIN_I]
        yc = x[..., COS_I]
        # grey-box: exact ENU heading kinematics + NN residuals [dVx,dVy,dyaw_r,dslip]
        # d(sin yaw)/dt = cos(yaw)*yawRate  ;  d(cos yaw)/dt = -sin(yaw)*yawRate
        return torch.stack([
            g[..., 0] * xs[VX_I],    # dVx
            g[..., 1] * xs[VY_I],    # dVy
            g[..., 2] * xs[YR_I],    # dyawRate
            yc * yr,                  # d(yaw_sin)/dt  — exact ENU
            -ys * yr,                 # d(yaw_cos)/dt  — exact ENU
            g[..., 3] * xs[SLIP_I],  # dslip
        ], dim=-1)


def _rk4(f, x, u, h, xm, xs, um, us):
    k1 = f(x,              u, xm, xs, um, us)
    k2 = f(x + 0.5*h*k1,  u, xm, xs, um, us)
    k3 = f(x + 0.5*h*k2,  u, xm, xs, um, us)
    k4 = f(x + h*k3,       u, xm, xs, um, us)
    return x + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


class _GB(nn.Module):
    """Grey-box Neural ODE wrapper."""
    def __init__(self, ck, hd=128):
        super().__init__()
        self.func = _Func(6, 2, hd)
        self.nr   = int(ck["num_rk4_steps"])
        for name in ("x_mean", "x_std", "u_mean", "u_std", "x_min", "x_max"):
            self.register_buffer(name, torch.as_tensor(ck[name], dtype=torch.float32))

    def step(self, x, u, dt):
        """x: (...,6)  u: (...,2)  dt: scalar tensor"""
        h = dt / self.nr
        for _ in range(self.nr):
            x = _rk4(self.func, x, u, h,
                      self.x_mean, self.x_std, self.u_mean, self.u_std)
        a   = x[..., SIN_I];  b = x[..., COS_I]
        nrm = torch.sqrt(a*a + b*b).clamp_min(1e-6)
        x   = x.clone()
        x[..., SIN_I] = a / nrm
        x[..., COS_I] = b / nrm
        return torch.clamp(x, self.x_min, self.x_max)


# ============================================================
# Utilities
# ============================================================

def _angle_diff(a, b):
    """Wrapped a-b in [-pi,pi]. Works on torch tensors or scalars."""
    d = a - b
    if isinstance(d, torch.Tensor):
        return torch.atan2(torch.sin(d), torch.cos(d))
    return math.atan2(math.sin(d), math.cos(d))


def _world_vel(x):
    """
    Body -> world velocity, standard ROS ENU convention (yaw=0 -> East).
      Ve = Vx*cos(yaw) - Vy*sin(yaw)
      Vn = Vx*sin(yaw) + Vy*cos(yaw)
    x : (6, K)  columns = samples
    """
    vx = x[VX_I, :];  vy = x[VY_I, :]
    sn = x[SIN_I, :]; cs = x[COS_I, :]
    return vx*cs - vy*sn,  vx*sn + vy*cs   # Ve, Vn


def _interp1d(s_ref, vals, s_query):
    """
    GPU 1D linear interpolation.
    s_ref  : (N,) sorted ascending arc-length knots
    vals   : (N,) values at knots
    s_query: (T,) query arc-lengths (clamped externally to [0, s_ref[-1]])
    Returns: (T,) interpolated values
    """
    idx = torch.searchsorted(s_ref.contiguous(), s_query.contiguous())
    idx = idx.clamp(1, s_ref.shape[0] - 1)
    s0  = s_ref[idx - 1];  s1 = s_ref[idx]
    v0  = vals[idx - 1];   v1 = vals[idx]
    alpha = ((s_query - s0) / (s1 - s0 + 1e-8)).clamp(0.0, 1.0)
    return v0 + alpha * (v1 - v0)


# ============================================================
# MPPIController
# ============================================================

class MPPIController:
    """
    Steer-only MPPI with exp09 Neural ODE rollout.
    Mirrors fs_mppi_init.m / fs_mppi_step.m, adapted for live PP path windows.
    """

    # ---- MPPI hyper-parameters ----
    K           = 1024      # GPU samples
    T           = 50        # horizon steps (50 * 0.005 = 0.25 s;  covers ~0.5 m at 2 m/s
                            #   + LOOKAHEAD_M = 2.5 m total — enough for dense skidpad)
    DT          = 0.005     # integration dt — must match model training step
    LAMBDA      = 1.0       # temperature
    SIG_STEER   = 0.10      # steer noise std (rad) — slightly higher than MATLAB for exploration
    STEER_MAX   = 0.349     # 20 deg — model training range (EUFS HW cap is 0.7)
    W_CT        = 2.0       # Huber crosstrack weight (raised vs MATLAB for tighter path)
    W_HEAD      = 0.5       # heading weight
    CT_SAT      = 3.0       # Huber knee (m) — smaller track = tighter saturation
    K_FF        = 0.85      # feedforward gain
    LOOKAHEAD_M = 1.5       # arc-length lookahead (metres) — replaces MATLAB index lookahead=8
    KP_ROLL     = 0.40      # P-only lon gain inside rollout
    KP          = 0.50      # PI lon kp (raised to compensate for low-speed sluggishness)
    KI          = 0.05      # PI lon ki (small to avoid windup)
    I_CLAMP     = 2.0       # integral clamp (reduced from 5.0 to limit windup)
    V_FLOOR     = 0.5       # min speed target (m/s)
    V_MAX_PATH  = 2.0       # cap on speed profile (m/s) — EUFS safe limit
    CT0         = 3.0       # CTE for speed reduction onset (m)
    K_CT        = 0.6       # CTE speed reduction gain
    ACCEL_MAX   = 3.0       # lon[-1,1] -> accel m/s² scaling
    L_WB        = 1.535     # wheelbase (m)

    def __init__(self, ckpt_path=None):
        # ---- device ----
        if torch.cuda.is_available():
            self.dev = torch.device("cuda")
            d = torch.cuda.get_device_properties(0)
            print(f"[MPPI] GPU: {d.name}  ({d.total_memory/1e9:.1f} GB)")
        else:
            self.dev = torch.device("cpu")
            print("[MPPI] WARNING: CUDA not available — CPU fallback. "
                  "Reduce K=128 and T=20 for real-time.")

        # ---- load model ----
        cp = ckpt_path or CKPT_PATH
        ck = torch.load(str(cp), map_location=self.dev, weights_only=False)
        hd = int(ck.get("hidden_dim", 128))
        self.model = _GB(ck, hd).to(self.dev)
        self.model.eval()
        # pre-cache dt as a GPU tensor so we don't create it every rollout step
        self._dt_t = torch.tensor(self.DT, dtype=torch.float32, device=self.dev)
        print(f"[MPPI] Loaded {Path(cp).name}  "
              f"(hidden={hd}, nr={self.model.nr}, K={self.K}, T={self.T})")

        # ---- nominal steer sequence (warm-start) ----
        self.nom = torch.zeros(1, self.T, device=self.dev)   # (1, T)

        # ---- longitudinal PI state ----
        self.I_spd  = 0.0
        self.vtgt0  = self.V_FLOOR

        # ---- path cache ----
        self._path_key  = None    # (len, first_pt, last_pt)
        self._path_data = None    # dict of GPU tensors

        # ---- progress tracker ----
        self.cur = 0

        # ---- state bridge ----
        self.prev_yaw   = None
        self.yaw_rate   = 0.0     # low-passed yawRate estimate (rad/s)

        # ---- speed from position derivative (SLAM odom may have zero twist) ----
        self._prev_pos  = None
        self._speed_est = 0.0     # low-passed position-derivative speed

        # ---- last output ----
        self._last_steer = 0.0

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def update_path(self, path_pts):
        """
        Call every loop with the latest list[(x,y)] from node.get_path().
        Recomputes cached tensors only when the path changes.
        """
        if not path_pts or len(path_pts) < 3:
            return

        key = (len(path_pts), path_pts[0], path_pts[-1])
        if key == self._path_key:
            return
        self._path_key = key

        pts = np.array(path_pts, dtype=np.float64)
        rx, ry = pts[:, 0], pts[:, 1]
        N = len(rx)

        # arc-length parameterisation
        ds_seg = np.hypot(np.diff(rx), np.diff(ry))
        ds_seg = np.where(ds_seg < 1e-6, 1e-6, ds_seg)   # guard zero-length segments
        s_np   = np.concatenate([[0.0], np.cumsum(ds_seg)])   # (N,)

        # headings
        dx = np.gradient(rx);  dy = np.gradient(ry)
        heading = np.arctan2(dy, dx)

        # left normals (for signed crosstrack)
        nE = -np.sin(heading)
        nN =  np.cos(heading)

        # C3 feedforward curvature -> steer
        kappa    = compute_signed_curvature(rx, ry)
        ff_steer = np.arctan(self.L_WB * kappa)

        # speed profile, capped at V_MAX_PATH
        try:
            vt = generate_velocity_profile(rx, ry)
            vt = np.clip(vt, self.V_FLOOR, self.V_MAX_PATH)
        except Exception:
            vt = np.full(N, self.V_FLOOR)

        dev = self.dev
        self._path_data = {
            "N":     N,
            "s":     torch.tensor(s_np,    dtype=torch.float32, device=dev),
            "rx":    torch.tensor(rx,      dtype=torch.float32, device=dev),
            "ry":    torch.tensor(ry,      dtype=torch.float32, device=dev),
            "nE":    torch.tensor(nE,      dtype=torch.float32, device=dev),
            "nN":    torch.tensor(nN,      dtype=torch.float32, device=dev),
            "ff":    torch.tensor(ff_steer,dtype=torch.float32, device=dev),
            "vt":    torch.tensor(vt,      dtype=torch.float32, device=dev),
            # numpy copies for CPU helpers
            "rx_np": rx,
            "ry_np": ry,
            "s_np":  s_np,
        }

        # reset progress when path changes
        self.cur    = 0
        self.I_spd  = 0.0   # also reset PI integrator to avoid windup on new path
        print(f"[MPPI] Path updated: {N} pts, "
              f"total={s_np[-1]:.1f} m, vt=[{vt.min():.1f},{vt.max():.1f}] m/s")

    def compute(self, x, y, yaw, speed, dt_ros):
        """
        One MPPI plan step.

        Parameters
        ----------
        x, y    : float  global position (m)
        yaw     : float  ENU heading (rad, CCW from East)
        speed   : float  |velocity| from odom (m/s); may be 0 if SLAM omits twist
        dt_ros  : float  sim-clock dt since last call (s)

        Returns
        -------
        steer_out : float  rad, clamped to STEER_MAX
        accel_out : float  m/s², clamped to ACCEL_MAX
        info      : dict
        """
        # ---- fallback: no valid path ----
        if self._path_data is None:
            return 0.0, -self.ACCEL_MAX, {
                "target_speed": 0.0, "cte": 0.0,
                "heading_err": 0.0, "mean_traj": [],
            }

        pd  = self._path_data
        dev = self.dev

        # ---- yawRate: finite-difference on heading, low-pass ----
        if self.prev_yaw is not None and dt_ros > 1e-4:
            raw_yr    = _angle_diff(yaw, self.prev_yaw) / dt_ros
            self.yaw_rate = 0.3 * raw_yr + 0.7 * self.yaw_rate
        self.prev_yaw = yaw

        # ---- speed: use odom if non-trivial, else position derivative ----
        _speed = float(speed)
        if _speed < 0.05:
            if self._prev_pos is not None and dt_ros > 0.005:
                dx_p = x - self._prev_pos[0]
                dy_p = y - self._prev_pos[1]
                spd_pos = math.hypot(dx_p, dy_p) / dt_ros
                self._speed_est = 0.3 * spd_pos + 0.7 * self._speed_est
                _speed = max(_speed, self._speed_est)
        self._prev_pos = (x, y)

        # ---- 6-D state bridge: odom -> Neural ODE state (ENU convention) ----
        # yaw_sin = sin(yaw),  yaw_cos = cos(yaw)
        x6 = torch.tensor(
            [_speed, 0.0, self.yaw_rate,
             math.sin(yaw), math.cos(yaw), 0.0],
            dtype=torch.float32, device=dev,
        )   # (6,)

        # ---- nearest path index ----
        self.cur = local_closest_index(
            (x, y), pd["rx_np"], pd["ry_np"], self.cur, loop=False
        )
        self.cur = int(np.clip(self.cur, 0, pd["N"] - 1))

        # ---- crosstrack error (for speed scaling + telemetry) ----
        cte_val, _ = cross_track_error(
            x, y, pd["rx_np"], pd["ry_np"], self.cur, loop=False
        )
        ct_abs = abs(cte_val)
        scale  = float(np.clip(
            1.0 - self.K_CT * min(ct_abs / self.CT0, 1.0), 0.0, 1.0
        ))
        vt_cur = float(pd["vt"][self.cur].item())
        vr_eff = float(np.clip(vt_cur * scale, self.V_FLOOR, self.V_MAX_PATH))

        # ---- arc-length marching (replaces index marching — works for any path density) ----
        s_cur  = float(pd["s"][self.cur].item())
        s_max  = float(pd["s"][-1].item())
        # advance by (speed * DT) metres per horizon step
        pace_m = max(_speed, self.V_FLOOR) * self.DT
        s_tgt  = torch.clamp(
            s_cur + self.LOOKAHEAD_M
            + torch.arange(self.T, dtype=torch.float32, device=dev) * pace_m,
            0.0, s_max,
        )   # (T,)

        refE  = _interp1d(pd["s"], pd["rx"], s_tgt)
        refN  = _interp1d(pd["s"], pd["ry"], s_tgt)
        nE_t  = _interp1d(pd["s"], pd["nE"], s_tgt)
        nN_t  = _interp1d(pd["s"], pd["nN"], s_tgt)
        vtgt  = _interp1d(pd["s"], pd["vt"], s_tgt)
        ff    = _interp1d(pd["s"], pd["ff"], s_tgt)

        # ---- sample steer perturbations (steer-only MPPI) ----
        eps  = self.SIG_STEER * torch.randn(self.K, self.T, device=dev)
        A    = (self.nom + eps).clamp(-self.STEER_MAX, self.STEER_MAX)   # (K, T)

        # ---- GPU rollout ----
        xk   = x6.unsqueeze(1).expand(6, self.K).clone()         # (6, K)
        pk   = torch.tensor([[x], [y]], dtype=torch.float32,
                             device=dev).expand(2, self.K).clone() # (2, K)
        cost = torch.zeros(self.K, device=dev)

        with torch.no_grad():
            for t in range(self.T):
                vx = xk[VX_I, :]                                    # (K,)

                # speed-dependent steer cap (MATLAB fs_mppi_step.m line 47)
                smax = (self.STEER_MAX
                        - (self.STEER_MAX - 0.14) * (vx - 13.0) / 5.0
                        ).clamp(0.14, self.STEER_MAX)

                # C3: feedforward + sampled feedback
                steer = (self.K_FF * ff[t] + A[:, t]).clamp(-smax, smax)  # (K,)

                # P-only lon in rollout
                lon   = (self.KP_ROLL * (vtgt[t] - vx)).clamp(-1.0, 1.0)  # (K,)

                # (2, K) control batch
                u_batch = torch.stack([steer, lon], dim=0)

                # world vel before step
                Ve1, Vn1 = _world_vel(xk)

                # Neural ODE step — model expects (..., 6) and (..., 2)
                xk_in  = xk.T.unsqueeze(0)       # (1, K, 6)
                u_in   = u_batch.T.unsqueeze(0)   # (1, K, 2)
                xk_out = self.model.step(xk_in, u_in, self._dt_t)
                xk     = xk_out.squeeze(0).T      # (6, K)

                # world vel after step
                Ve2, Vn2 = _world_vel(xk)

                # midpoint position integration
                pk[0] += 0.5 * (Ve1 + Ve2) * self.DT
                pk[1] += 0.5 * (Vn1 + Vn2) * self.DT

                # C1: Huber crosstrack
                perp = ((pk[0] - refE[t]) * nE_t[t]
                      + (pk[1] - refN[t]) * nN_t[t])               # (K,)
                ap   = perp.abs()
                ctc  = (ap.clamp(max=self.CT_SAT) ** 2
                      + 2 * self.CT_SAT * (ap - self.CT_SAT).clamp(min=0.0))

                # C2: pure-pursuit heading (ENU: yaw = atan2(sin, cos))
                carTh = torch.atan2(xk[SIN_I, :], xk[COS_I, :])    # (K,)
                bear  = torch.atan2(refN[t] - pk[1], refE[t] - pk[0])
                dh    = _angle_diff(carTh, bear)                     # (K,)

                cost += self.W_CT * ctc + self.W_HEAD * dh ** 2

        cost = cost / self.T

        # ---- softmax weighting -> updated nominal ----
        wts     = torch.exp(-(cost - cost.min()) / self.LAMBDA)
        wts     = wts / wts.sum()                                    # (K,)
        nom_new = (wts.unsqueeze(1) * A).sum(dim=0)                 # (T,)
        self.nom = nom_new.unsqueeze(0)                              # (1, T)

        # ---- apply first action ----
        smax0 = float(np.clip(
            self.STEER_MAX - (self.STEER_MAX - 0.14) * (_speed - 13.0) / 5.0,
            0.14, self.STEER_MAX
        ))
        steer0_raw = self.K_FF * float(ff[0]) + float(nom_new[0])
        steer_out  = float(np.clip(steer0_raw, -smax0, smax0))
        self._last_steer = steer_out

        # ---- warm-start roll ----
        self.nom = torch.cat([self.nom[:, 1:], self.nom[:, -1:]], dim=1)

        # ---- PI longitudinal ----
        err        = vr_eff - _speed
        self.I_spd = float(np.clip(
            self.I_spd + err * dt_ros, -self.I_CLAMP, self.I_CLAMP
        ))
        lon0      = float(np.clip(self.KP * err + self.KI * self.I_spd, -1.0, 1.0))
        accel_out = lon0 * self.ACCEL_MAX

        # ---- heading error for telemetry ----
        hd_ref      = path_heading(pd["rx_np"], pd["ry_np"], self.cur, loop=False)
        heading_err = _angle_diff(yaw, hd_ref)

        # ---- mean trajectory from best 10 % of samples (for viz) ----
        n_best   = max(1, self.K // 10)
        best_idx = torch.argsort(cost)[:n_best]
        mean_traj = self._fast_pos_rollout(
            x6, x, y, A[best_idx], ff, vtgt
        )

        info = {
            "target_speed": vr_eff,
            "cte":          cte_val,
            "heading_err":  float(heading_err),
            "mean_traj":    mean_traj,
            "_speed":       _speed,      # estimated speed, for debug
        }
        return steer_out, accel_out, info

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------

    def _fast_pos_rollout(self, x6_init, px0, py0, A_sub, ff, vtgt):
        """
        Cheap position rollout over a subset of samples for the viz arc.
        Returns list[(x, y)] at every 5 steps.
        """
        K_sub = A_sub.shape[0]
        xk    = x6_init.unsqueeze(1).expand(6, K_sub).clone()
        pk    = torch.tensor([[px0], [py0]], dtype=torch.float32,
                             device=self.dev).expand(2, K_sub).clone()
        pts   = []
        with torch.no_grad():
            for t in range(self.T):
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
                xk    = self.model.step(xk_T, u_T, self._dt_t).squeeze(0).T
                Ve2, Vn2 = _world_vel(xk)
                pk[0] += 0.5 * (Ve1 + Ve2) * self.DT
                pk[1] += 0.5 * (Vn1 + Vn2) * self.DT
                if t % 5 == 0:
                    pts.append((float(pk[0].mean()), float(pk[1].mean())))
        return pts
