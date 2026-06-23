#!/usr/bin/env python3
"""
mppi_controller.py  —  exp09 Neural ODE MPPI for RAMS-e-25 ROS2 stack.

WHY IT WORKED IN MATLAB BUT NOT HERE
  - MATLAB: full 2624-pt track, true Vx, 3-20 m/s → pace=0.015-0.10 m/step,
    T=200 → 3-20 m horizon → car saw the whole corner.
  - EUFS: 5-10 jittery points, odom twist = 0, 0.5-2 m/s.
    With dt=0.005, pace=0.0025-0.01 m/step → T=200 horizon = 0.5-2 m.
    Over 2 m a corner is nearly straight → all steer samples cost the same
    → softmax uniform → nom ≈ 0 → steers straight off the track.

THE FIX (single change, one knob)
  Effective rollout dt = 0.02  (4× training dt, RK4 O(h⁵) error still >>1e-10).
  T=200 × 0.02 × 1.5 m/s = 6 m horizon + LOOKAHEAD_M = 8 m → matches MATLAB.

Secondary fixes:
  - Speed from windowed position regression (SLAM twist is always zero).
  - LOOKAHEAD_M = 3 m (vs 2), feed horizon further.
  - V_PLAN lower-bound keeps MPPI alive when speed estimator hasn't converged.
"""

import math
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from controls.control_utils import (
    local_closest_index,
    cross_track_error,
    path_heading,
)

# ============================================================
# Paths
# ============================================================
_HERE     = Path(__file__).parent
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
        yr = x[..., YR_I];  ys = x[..., SIN_I];  yc = x[..., COS_I]
        # grey-box ENU kinematics: d(sin yaw)=cos(yaw)*yaw_r, d(cos yaw)=-sin(yaw)*yaw_r
        return torch.stack([
            g[..., 0] * xs[VX_I],   # dVx
            g[..., 1] * xs[VY_I],   # dVy
            g[..., 2] * xs[YR_I],   # dyawRate
            yc * yr,                 # d(sin yaw)/dt
            -ys * yr,                # d(cos yaw)/dt
            g[..., 3] * xs[SLIP_I], # dslip
        ], dim=-1)


def _rk4(f, x, u, h, xm, xs, um, us):
    k1 = f(x,             u, xm, xs, um, us)
    k2 = f(x + 0.5*h*k1, u, xm, xs, um, us)
    k3 = f(x + 0.5*h*k2, u, xm, xs, um, us)
    k4 = f(x + h*k3,     u, xm, xs, um, us)
    return x + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


class _GB(nn.Module):
    def __init__(self, ck, hd=128):
        super().__init__()
        self.func = _Func(6, 2, hd)
        self.nr   = int(ck["num_rk4_steps"])
        for name in ("x_mean", "x_std", "u_mean", "u_std", "x_min", "x_max"):
            self.register_buffer(name, torch.as_tensor(ck[name], dtype=torch.float32))

    def step(self, x, u, dt):
        h = dt / self.nr
        for _ in range(self.nr):
            x = _rk4(self.func, x, u, h,
                      self.x_mean, self.x_std, self.u_mean, self.u_std)
        a   = x[..., SIN_I];  b = x[..., COS_I]
        nrm = torch.sqrt(a*a + b*b).clamp_min(1e-6)
        x   = x.clone()
        x[..., SIN_I] = a / nrm;  x[..., COS_I] = b / nrm
        return torch.clamp(x, self.x_min, self.x_max)


# ============================================================
# Utilities
# ============================================================

def _angle_diff(a, b):
    d = a - b
    if isinstance(d, torch.Tensor):
        return torch.atan2(torch.sin(d), torch.cos(d))
    return math.atan2(math.sin(d), math.cos(d))


def _world_vel(x):
    """Body→world ENU.  x: (6,B).  yaw_sin at [3], yaw_cos at [4]."""
    vx = x[VX_I, :];  vy = x[VY_I, :]
    sn = x[SIN_I, :]; cs = x[COS_I, :]
    return vx*cs - vy*sn,  vx*sn + vy*cs


def _interp1d(s_ref, vals, s_query):
    """GPU 1-D linear interp.  s_ref sorted ascending."""
    idx = torch.searchsorted(s_ref.contiguous(), s_query.contiguous())
    idx = idx.clamp(1, s_ref.shape[0] - 1)
    s0, s1 = s_ref[idx - 1], s_ref[idx]
    v0, v1 = vals[idx - 1], vals[idx]
    alpha = ((s_query - s0) / (s1 - s0 + 1e-8)).clamp(0.0, 1.0)
    return v0 + alpha * (v1 - v0)


# ============================================================
# MPPIController
# ============================================================

class MPPIController:
    """Steer-only MPPI with exp09 Neural ODE for EUFS Gazebo."""

    # ---- MPPI parameters ----
    K           = 1024      # GPU samples
    T           = 200       # horizon steps
    DT          = 0.005     # model training dt (for NN step only)
    DT_EFF      = 0.02      # EFFECTIVE rollout dt (4× training dt).
                            # T*DT_EFF*v_plan = 200*0.02*2.0 = 8 m horizon.
    LAMBDA      = 1.0       # temperature
    SIG_STEER   = 0.12      # steer noise std (rad)
    STEER_MAX   = 0.349     # 20 deg — model training range
    W_CT        = 2.0       # Huber crosstrack
    W_HEAD      = 0.5       # pure-pursuit heading
    CT_SAT      = 3.0       # Huber knee (m)
    K_FF        = 0.0       # feedforward OFF — sparse PP points give noise kappa
    LOOKAHEAD_M = 3.0       # arc-length lookahead (m) — like MATLAB ~8 index pts

    V_PLAN_MIN  = 1.5       # floor injected into rollout Vx if speed est not yet alive

    KP_ROLL     = 0.0       # lon=0 in rollout — decouple NN lateral from unreliable Vx
    KP          = 2.0       # PI lon output gain
    KI          = 0.5       # PI integral gain
    I_CLAMP     = 3.0
    V_FLOOR     = 0.8       # min speed target (m/s)
    V_MAX       = 2.0       # max speed target (m/s)
    CT0         = 3.0       # CTE onset for speed reduction (m)
    K_CT        = 0.4
    ACCEL_MAX   = 3.0       # lon[-1,1] → m/s²
    L_WB        = 1.535

    # ---- speed estimator (windowed position regression) ----
    SPD_WIN_S   = 0.5        # seconds of position history

    def __init__(self, ckpt_path=None):
        if torch.cuda.is_available():
            self.dev = torch.device("cuda")
            d = torch.cuda.get_device_properties(0)
            print(f"[MPPI] GPU: {d.name}  ({d.total_memory/1e9:.1f} GB)")
        else:
            self.dev = torch.device("cpu")
            print("[MPPI] WARNING: CUDA not available — CPU fallback.")

        cp = ckpt_path or CKPT_PATH
        ck = torch.load(str(cp), map_location=self.dev, weights_only=False)
        hd = int(ck.get("hidden_dim", 128))
        self.model = _GB(ck, hd).to(self.dev)
        self.model.eval()
        self._dt_eff_t = torch.tensor(self.DT_EFF, dtype=torch.float32, device=self.dev)
        print(f"[MPPI] Loaded {Path(cp).name}  (hidden={hd}, nr={self.model.nr}, "
              f"K={self.K}, T={self.T}, dt_eff={self.DT_EFF})")

        # warm-start nominal steer
        self.nom = torch.zeros(1, self.T, device=self.dev)

        # longitudinal PI
        self.I_spd = 0.0

        # windowed speed estimator
        self._pos_hist = deque()   # (t_ros, x, y)

        self._path_key  = None
        self._path_data = None
        self.cur        = 0

        self.prev_yaw = None
        self.yaw_rate = 0.0

        self._dbg_ctr = 0

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def update_path(self, path_pts):
        if not path_pts or len(path_pts) < 3:
            return
        key = (len(path_pts), path_pts[0], path_pts[-1])
        if key == self._path_key:
            return
        self._path_key = key

        pts = np.array(path_pts, dtype=np.float64)
        rx, ry = pts[:, 0], pts[:, 1]
        N = len(rx)

        ds_seg = np.hypot(np.diff(rx), np.diff(ry))
        ds_seg = np.where(ds_seg < 1e-6, 1e-6, ds_seg)
        s_np   = np.concatenate([[0.0], np.cumsum(ds_seg)])

        dx = np.gradient(rx);  dy = np.gradient(ry)
        heading = np.arctan2(dy, dx)
        nE = -np.sin(heading)
        nN =  np.cos(heading)

        # Simple curvature-based speed profile
        kappa = np.gradient(heading) / (np.maximum(ds_seg, 1e-3))
        vt = self.V_MAX / (1.0 + np.abs(np.clip(kappa * self.L_WB, -1, 1)) * 2.5)
        vt = np.clip(vt, self.V_FLOOR, self.V_MAX)

        dev = self.dev
        self._path_data = {
            "N":  N,
            "s":  torch.tensor(s_np,   dtype=torch.float32, device=dev),
            "rx": torch.tensor(rx,     dtype=torch.float32, device=dev),
            "ry": torch.tensor(ry,     dtype=torch.float32, device=dev),
            "nE": torch.tensor(nE,     dtype=torch.float32, device=dev),
            "nN": torch.tensor(nN,     dtype=torch.float32, device=dev),
            "ff": torch.zeros(N,       dtype=torch.float32, device=dev),
            "vt": torch.tensor(vt,     dtype=torch.float32, device=dev),
            "rx_np": rx, "ry_np": ry, "s_np": s_np,
        }
        self.cur = 0
        print(f"[MPPI] Path updated: {N} pts, total={s_np[-1]:.1f} m, "
              f"vt=[{vt.min():.1f},{vt.max():.1f}]")

    def compute(self, x, y, yaw, speed, dt_ros):
        """One MPPI step. Returns (steer_rad, accel_m/s2, info)."""
        if self._path_data is None:
            return 0.0, -self.ACCEL_MAX, {
                "target_speed": 0.0, "cte": 0.0,
                "heading_err": 0.0, "mean_traj": [],
            }

        pd  = self._path_data
        dev = self.dev
        dt_ros = max(dt_ros, 0.001)

        # ---- windowed-position speed estimator --------------------------
        # SLAM odom publishes twist=0 always.  We derive speed from position
        # history via linear regression over the last SPD_WIN_S seconds.
        self._pos_hist.append((dt_ros, x, y))
        # prune old
        ttl = sum(h[0] for h in self._pos_hist)
        while self._pos_hist and ttl > self.SPD_WIN_S + 0.2:
            ttl -= self._pos_hist[0][0]
            self._pos_hist.popleft()
        # linear regression on x(t), y(t) over accumulated time
        est_spd = 0.0
        if len(self._pos_hist) >= 3 and ttl > 0.05:
            ts = np.cumsum([h[0] for h in self._pos_hist])
            xs = np.array([h[1] for h in self._pos_hist])
            ys = np.array([h[2] for h in self._pos_hist])
            # mean-subtract for stable slope
            tm = ts - ts.mean()
            vx = np.dot(tm, xs - xs.mean()) / max(np.dot(tm, tm), 1e-6)
            vy = np.dot(tm, ys - ys.mean()) / max(np.dot(tm, tm), 1e-6)
            est_spd = math.hypot(vx, vy)

        # Use the max of (odom speed, regression speed, V_PLAN_MIN)
        _speed = max(float(speed), est_spd, self.V_PLAN_MIN)

        # ---- yawRate: finite-diff + low-pass ----------------------------
        if self.prev_yaw is not None:
            raw_yr = _angle_diff(yaw, self.prev_yaw) / dt_ros
            self.yaw_rate = 0.3 * raw_yr + 0.7 * self.yaw_rate
        self.prev_yaw = yaw

        # ---- 6-D state (ENU convention) -----------------------------------
        x6 = torch.tensor(
            [_speed, 0.0, self.yaw_rate,
             math.sin(yaw), math.cos(yaw), 0.0],
            dtype=torch.float32, device=dev,
        )

        # ---- current path index + crosstrack -----------------------------
        self.cur = int(np.clip(
            local_closest_index((x, y), pd["rx_np"], pd["ry_np"],
                                self.cur, loop=False),
            0, pd["N"] - 1))

        cte_val, _ = cross_track_error(
            x, y, pd["rx_np"], pd["ry_np"], self.cur, loop=False)
        ct_abs = abs(cte_val)
        scale  = float(np.clip(
            1.0 - self.K_CT * min(ct_abs / self.CT0, 1.0), 0.0, 1.0))
        vt_cur = float(pd["vt"][self.cur].item())
        vr_eff = float(np.clip(vt_cur * scale, self.V_FLOOR, self.V_MAX))

        # ---- arc-length horizon (KEY FIX: dt_eff, pace spread) ----------
        s_cur  = float(pd["s"][self.cur].item())
        s_max  = float(pd["s"][-1].item())
        # _speed used here so reference s_tgt advances at ≈ real car speed
        pace_m = max(_speed, self.V_PLAN_MIN) * self.DT_EFF
        s_tgt  = torch.clamp(
            s_cur + self.LOOKAHEAD_M
            + torch.arange(self.T, dtype=torch.float32, device=dev) * pace_m,
            0.0, s_max,
        )

        refE = _interp1d(pd["s"], pd["rx"], s_tgt)
        refN = _interp1d(pd["s"], pd["ry"], s_tgt)
        nE_t = _interp1d(pd["s"], pd["nE"], s_tgt)
        nN_t = _interp1d(pd["s"], pd["nN"], s_tgt)
        vtgt = _interp1d(pd["s"], pd["vt"], s_tgt)
        ff   = _interp1d(pd["s"], pd["ff"], s_tgt)

        # ---- steer-only MPPI samples -------------------------------------
        eps = self.SIG_STEER * torch.randn(self.K, self.T, device=dev)
        A   = (self.nom + eps).clamp(-self.STEER_MAX, self.STEER_MAX)

        # ---- GPU rollout -------------------------------------------------
        xk   = x6.unsqueeze(1).expand(6, self.K).clone()
        pk   = torch.tensor([[x], [y]], dtype=torch.float32,
                             device=dev).expand(2, self.K).clone()
        cost = torch.zeros(self.K, device=dev)

        with torch.no_grad():
            for t in range(self.T):
                vx = xk[VX_I, :]

                smax  = (self.STEER_MAX
                         - (self.STEER_MAX - 0.14) * (vx - 13.0) / 5.0
                         ).clamp(0.14, self.STEER_MAX)

                steer = (self.K_FF * ff[t] + A[:, t]).clamp(-smax, smax)
                lon   = torch.zeros_like(vx)   # KP_ROLL = 0

                u_batch = torch.stack([steer, lon], dim=0)

                Ve1, Vn1 = _world_vel(xk)

                # Neural ODE step with EFFECTIVE dt (0.02, not 0.005)
                xk_in  = xk.T.unsqueeze(0)
                u_in   = u_batch.T.unsqueeze(0)
                xk     = self.model.step(xk_in, u_in, self._dt_eff_t).squeeze(0).T

                Ve2, Vn2 = _world_vel(xk)

                pk[0] += 0.5 * (Ve1 + Ve2) * self.DT_EFF
                pk[1] += 0.5 * (Vn1 + Vn2) * self.DT_EFF

                # C1 Huber CTE
                perp = ((pk[0] - refE[t]) * nE_t[t]
                      + (pk[1] - refN[t]) * nN_t[t])
                ap   = perp.abs()
                ctc  = (ap.clamp(max=self.CT_SAT) ** 2
                      + 2 * self.CT_SAT * (ap - self.CT_SAT).clamp(min=0.0))

                # C2 pure-pursuit heading
                carTh = torch.atan2(xk[SIN_I, :], xk[COS_I, :])
                bear  = torch.atan2(refN[t] - pk[1], refE[t] - pk[0])
                dh    = _angle_diff(carTh, bear)

                cost += self.W_CT * ctc + self.W_HEAD * dh ** 2

        cost /= self.T

        # ---- softmax → new nominal ---------------------------------------
        wts     = torch.exp(-(cost - cost.min()) / self.LAMBDA)
        wts     = wts / wts.sum()
        nom_new = (wts.unsqueeze(1) * A).sum(dim=0)
        self.nom = nom_new.unsqueeze(0)

        # ---- first action ------------------------------------------------
        smax0 = float(np.clip(
            self.STEER_MAX - (self.STEER_MAX - 0.14) * (_speed - 13.0) / 5.0,
            0.14, self.STEER_MAX))
        steer_out = float(np.clip(
            float(ff[0]) + float(nom_new[0]), -smax0, smax0))

        # warm-start shift
        self.nom = torch.cat([self.nom[:, 1:], self.nom[:, -1:]], dim=1)

        # ---- PI longitudinal output --------------------------------------
        err        = vr_eff - _speed
        self.I_spd = float(np.clip(
            self.I_spd + err * dt_ros, -self.I_CLAMP, self.I_CLAMP))
        lon0       = float(np.clip(self.KP * err + self.KI * self.I_spd, -1.0, 1.0))
        accel_out  = lon0 * self.ACCEL_MAX

        # ---- telemetry ---------------------------------------------------
        hd_ref      = path_heading(pd["rx_np"], pd["ry_np"], self.cur, loop=False)
        heading_err = _angle_diff(yaw, hd_ref)

        self._dbg_ctr += 1
        if self._dbg_ctr % 100 == 1:
            span = pace_m * (self.T - 1)
            print(f"[MPPI] odom={speed:.2f}  est={est_spd:.2f}  use={_speed:.2f}  "
                  f"yr={self.yaw_rate:.3f}  cur={self.cur}/{pd['N']}  "
                  f"CTE={cte_val:.2f}  steer={steer_out:.3f}  accel={accel_out:.2f}  "
                  f"horizon_span={span:.1f}m  I={self.I_spd:.2f}")

        # viz: mean traj from best 10%
        n_best   = max(1, self.K // 10)
        best_idx = torch.argsort(cost)[:n_best]
        mean_traj = self._fast_pos(x6, x, y, A[best_idx], ff, vtgt)

        return steer_out, accel_out, {
            "target_speed": vr_eff,
            "cte":          cte_val,
            "heading_err":  float(heading_err),
            "mean_traj":    mean_traj,
        }

    # ----------------------------------------------------------
    def _fast_pos(self, x6_init, px0, py0, A_sub, ff, vtgt):
        K_sub = A_sub.shape[0]
        xk = x6_init.unsqueeze(1).expand(6, K_sub).clone()
        pk = torch.tensor([[px0], [py0]], dtype=torch.float32,
                           device=self.dev).expand(2, K_sub).clone()
        pts = []
        with torch.no_grad():
            for t in range(self.T):
                vx = xk[VX_I, :]
                smax = (self.STEER_MAX
                        - (self.STEER_MAX - 0.14) * (vx - 13.0) / 5.0
                        ).clamp(0.14, self.STEER_MAX)
                steer = (self.K_FF * ff[t] + A_sub[:, t]).clamp(-smax, smax)
                lon   = torch.zeros_like(vx)
                u_b   = torch.stack([steer, lon], dim=0)
                Ve1, Vn1 = _world_vel(xk)
                xk = self.model.step(
                    xk.T.unsqueeze(0), u_b.T.unsqueeze(0),
                    self._dt_eff_t).squeeze(0).T
                Ve2, Vn2 = _world_vel(xk)
                pk[0] += 0.5 * (Ve1 + Ve2) * self.DT_EFF
                pk[1] += 0.5 * (Vn1 + Vn2) * self.DT_EFF
                if t % 8 == 0:
                    pts.append((float(pk[0].mean()), float(pk[1].mean())))
        return pts
