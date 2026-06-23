#!/usr/bin/env python3
"""
mppi_controller.py  —  exp09 Neural ODE MPPI for RAMS-e-25 (EUFS Gazebo).

Steer-only MPPI, ported from MATLAB fs_mppi_step.m / fs_mppi_init.m.
Uses the exp09 grey-box Neural ODE (best_model_600.pt) as the rollout model.

PERFORMANCE (the reason this works in WSL2 now):
  The K=1024,T=200 RK4 rollout took ~3.9 s/call on WSL2 GPU because WSL2 CUDA
  passthrough adds ~0.6 ms overhead PER kernel launch, and the per-step Python
  loop launches thousands of tiny kernels.
  Fixed with CUDA GRAPHS: the whole T-step rollout is captured once into a graph
  and replayed as a SINGLE GPU submission.  Benchmarked K=1024,T=100 -> 37.8 ms
  (bit-identical to the plain rollout).  Stays fully on GPU, full sample count.

DESIGN NOTES (EUFS-specific, validated by headless tests):
  - SLAM odom twist = 0 -> speed estimated from windowed position regression.
  - PP path is a sparse 5-11 pt rolling window -> arc-length interpolation.
  - Kinematic yaw injection (yawRate = Vx*tan(steer)/L) each step: the NN was
    trained where yawRate already tracked the kinematic value, so its dyawRate is
    a small residual, not the primary yaw driver.
  - V_PLAN_MIN injected into rollout Vx so the NN sees in-distribution speed and
    trajectories spread (gives meaningful steering cost).
  - K_FF=0: curvature feedforward off (np.gradient on ~6 pts is noise).

AUTHOR: Soumil (RMS controls team)
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import math
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from controls.control_utils import local_closest_index, cross_track_error, path_heading

# =========================================================================
# Paths
# =========================================================================
_HERE     = Path(__file__).parent
CKPT_PATH = _HERE / "fs_model" / "best_model_600.pt"

# State / control indices (matches export_fs_to_matlab.py)
VX_I, VY_I, YR_I, SIN_I, COS_I, SLIP_I = 0, 1, 2, 3, 4, 5


# =========================================================================
# Neural ODE  (exact copy of the MATLAB-side GB class)
# =========================================================================
class _Func(nn.Module):
    """8 → 128 → 128 → 128 → 4  (SiLU activations)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 4),
        )

    def forward(self, x, u, x_mean, x_std, u_mean, u_std):
        inp = torch.cat([(x - x_mean) / x_std, (u - u_mean) / u_std], dim=-1)
        g   = self.net(inp)
        yr  = x[..., YR_I]
        ys  = x[..., SIN_I]
        yc  = x[..., COS_I]
        # grey-box: exact yaw kinematics + NN residuals on 4 channels
        # ENU: d(sin)/dt = cos·ω  ,  d(cos)/dt = −sin·ω
        return torch.stack([
            g[..., 0] * x_std[VX_I],    # dVx
            g[..., 1] * x_std[VY_I],    # dVy
            g[..., 2] * x_std[YR_I],    # dyawRate
            yc * yr,                     # d(yaw_sin)  exact
            -ys * yr,                    # d(yaw_cos)  exact
            g[..., 3] * x_std[SLIP_I],  # dslip
        ], dim=-1)


def _rk4(f, x, u, h, *s):
    k1 = f(x,             u, *s)
    k2 = f(x + 0.5*h*k1, u, *s)
    k3 = f(x + 0.5*h*k2, u, *s)
    k4 = f(x + h*k3,     u, *s)
    return x + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


class _GB(nn.Module):
    """Grey-box ODE wrapper (mirrors Python export_fs_to_matlab.py)."""
    def __init__(self, ck):
        super().__init__()
        self.func = _Func()
        self.nr   = int(ck["num_rk4_steps"])
        for name in ("x_mean", "x_std", "u_mean", "u_std", "x_min", "x_max"):
            self.register_buffer(name, torch.as_tensor(ck[name], dtype=torch.float32))

    def step(self, x, u, dt, euler=False):
        """x: (...,6)  u: (...,2)  dt: scalar tensor.  if euler: single forward step."""
        if euler:
            dx = self.func(x, u, self.x_mean, self.x_std, self.u_mean, self.u_std)
            x  = x + dt * dx
        else:
            h = dt / self.nr
            for _ in range(self.nr):
                x = _rk4(self.func, x, u, h,
                          self.x_mean, self.x_std, self.u_mean, self.u_std)
        # renormalise yaw unit vector
        a   = x[..., SIN_I];  b = x[..., COS_I]
        nrm = torch.sqrt(a * a + b * b).clamp_min(1e-6)
        x   = x.clone()
        x[..., SIN_I] = a / nrm
        x[..., COS_I] = b / nrm
        return torch.clamp(x, self.x_min, self.x_max)


# =========================================================================
# angle + velocity utilities
# =========================================================================
def _angle_diff(a, b):
    d = a - b
    if isinstance(d, torch.Tensor):
        return torch.atan2(torch.sin(d), torch.cos(d))
    return math.atan2(math.sin(d), math.cos(d))


def _world_vel(x_K6):
    """
    Body → world.  ROS ENU: yaw=0 = East (+x), CCW positive.   x_K6: (K,6)
    Ve = Vx·cos(yaw) − Vy·sin(yaw)
    Vn = Vx·sin(yaw) + Vy·cos(yaw)
    """
    return (x_K6[:, VX_I] * x_K6[:, COS_I] - x_K6[:, VY_I] * x_K6[:, SIN_I],
            x_K6[:, VX_I] * x_K6[:, SIN_I] + x_K6[:, VY_I] * x_K6[:, COS_I])


def _interp_1d(s_knots, values, s_query):
    """GPU linear interpolation.  s_knots must be sorted."""
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
    """
    Steer-only MPPI, exp09 Neural ODE rollout, CUDA-graph accelerated.
    Matches the validated MATLAB fs_mppi_init.m config; EUFS deltas noted.
    """

    # ---- MPPI parameters (GPU + CUDA graph: K=1024,T=100 -> ~38 ms in WSL2) ----
    K           = 1024        # samples (full count, senior's requirement)
    T           = 100         # horizon steps (100 * DT_EFF = 2.0 s rollout)
    DT          = 0.005       # model training dt
    DT_EFF      = 0.02        # effective rollout dt (s)  -> 100*0.02*4 = 8 m @ 4 m/s
    USE_EULER   = False       # RK4 (full accuracy; CUDA graph makes it affordable)
    LAMBDA      = 1.0
    SIG_STEER   = 0.10        # steer noise (rad)
    STEER_MAX   = 0.349       # 20° — model range
    W_CT        = 2.0         # crosstrack weight (Huber)
    W_HEAD      = 0.5         # pure-pursuit heading weight
    CT_SAT      = 3.0         # Huber knee (m)
    LOOKAHEAD_M = 2.0         # arc-length preview offset (m)

    # ---- longitudinal (PI controller, applied to ACCEL output) ----
    KP          = 2.0
    KI          = 0.3
    I_CLAMP     = 3.0
    V_FLOOR     = 0.5         # min speed target (m/s)
    V_MAX       = 2.0         # max speed target (m/s)
    CT0         = 3.0         # CTE threshold for speed reduction (m)
    K_CT        = 0.5
    ACCEL_MAX   = 3.0         # m/s²  (EUFS sim clip: -3 .. 3)

    # ---- planning speed for rollout (NN needs ≥4 m/s for yaw authority) ----
    V_PLAN_MIN  = 4.0
    L_WB_KIN    = 1.535       # wheelbase for kinematic yaw injection

    # ---- windowed speed estimator ----
    SPD_WINDOW_S = 0.5

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

        # ---- load checkpoint ----
        cp = ckpt_path or CKPT_PATH
        ck = torch.load(str(cp), map_location=self.dev, weights_only=False)
        self.model = _GB(ck).to(self.dev)
        self.model.eval()
        self._dt_t = torch.tensor(self.DT_EFF, dtype=torch.float32, device=self.dev)

        # ---- static buffers for the rollout (fixed addresses for CUDA graph) ----
        dev = self.dev
        self._A      = torch.zeros(self.K, self.T, device=dev)   # steer samples
        self._refE   = torch.zeros(self.T, device=dev)
        self._refN   = torch.zeros(self.T, device=dev)
        self._nE     = torch.zeros(self.T, device=dev)
        self._nN     = torch.zeros(self.T, device=dev)
        self._x0     = torch.zeros(6, device=dev)                # initial state
        self._p0     = torch.zeros(2, device=dev)                # initial position
        self._cost   = torch.zeros(self.K, device=dev)           # output

        self._graph  = None    # captured lazily on first compute()

        print(f"[MPPI] Loaded {Path(cp).name}  K={self.K}  T={self.T}  "
              f"dt_eff={self.DT_EFF}s  RK4={not self.USE_EULER}")

        # ---- warm-start nominal steer ----
        self.nom   = torch.zeros(1, self.T, device=dev)

        # ---- longitudinal PI state ----
        self.I_spd = 0.0

        # ---- path cache ----
        self._path_key  = None
        self._path_data = None
        self.cur        = 0

        # ---- yaw-rate (finite diff + low-pass) ----
        self.prev_yaw = None
        self.yaw_rate = 0.0

        # ---- windowed speed estimator ----
        self._pose_window = deque()

        self._dbg_ctr = 0

    # =====================================================================
    #  The rollout — reads ONLY the static buffers, writes self._cost.
    #  This is what gets captured into a CUDA graph.
    # =====================================================================
    @torch.no_grad()
    def _rollout(self):
        K, T = self.K, self.T
        dev  = self.dev
        # broadcast initial state/position to K samples
        xk = self._x0.unsqueeze(0).expand(K, 6).contiguous()     # (K,6)
        pe = self._p0[0].expand(K).contiguous()                  # (K,)
        pn = self._p0[1].expand(K).contiguous()                  # (K,)
        cost = torch.zeros(K, device=dev)

        for t in range(T):
            vx = xk[:, VX_I]
            smax = (self.STEER_MAX
                    - (self.STEER_MAX - 0.14) * (vx - 13.0) / 5.0
                    ).clamp(0.14, self.STEER_MAX)
            steer = self._A[:, t].clamp(-smax, smax)             # (K,)

            # kinematic yaw injection
            yr_kin = vx * (steer / self.L_WB_KIN)
            xk = xk.clone()
            xk[:, YR_I] = yr_kin

            lon = torch.zeros(K, device=dev)
            u   = torch.stack([steer, lon], dim=1)               # (K,2)

            Ve1, Vn1 = _world_vel(xk)
            xk = self.model.step(xk, u, self._dt_t, euler=self.USE_EULER)
            Ve2, Vn2 = _world_vel(xk)

            pe = pe + 0.5 * (Ve1 + Ve2) * self.DT_EFF
            pn = pn + 0.5 * (Vn1 + Vn2) * self.DT_EFF

            # C1: Huber crosstrack
            perp = (pe - self._refE[t]) * self._nE[t] + (pn - self._refN[t]) * self._nN[t]
            ap   = perp.abs()
            ctc  = ap.clamp(max=self.CT_SAT)**2 + 2*self.CT_SAT*(ap - self.CT_SAT).clamp(min=0.0)

            # C2: pure-pursuit heading
            car_th = torch.atan2(xk[:, SIN_I], xk[:, COS_I])
            bear   = torch.atan2(self._refN[t] - pn, self._refE[t] - pe)
            dh     = torch.atan2(torch.sin(car_th - bear), torch.cos(car_th - bear))

            cost = cost + self.W_CT * ctc + self.W_HEAD * (dh * dh)

        self._cost.copy_(cost / T)

    def _capture_graph(self):
        """Capture self._rollout() into a CUDA graph (call once)."""
        # warmup in a side stream (required before capture)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._rollout()
        torch.cuda.current_stream().wait_stream(s)

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._rollout()
        torch.cuda.synchronize()
        print("[MPPI] CUDA graph captured")

    # =====================================================================
    #  update_path
    # =====================================================================
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

        seg  = np.maximum(np.hypot(np.diff(rx), np.diff(ry)), 1e-6)
        s_np = np.concatenate([[0.0], np.cumsum(seg)])

        dx, dy  = np.gradient(rx), np.gradient(ry)
        heading = np.arctan2(dy, dx)
        nE      = -np.sin(heading)
        nN      =  np.cos(heading)
        vt      = np.full(N, self.V_MAX, dtype=np.float64)

        dev = self.dev
        self._path_data = {
            "N":  N,
            "s":  torch.tensor(s_np, dtype=torch.float32, device=dev),
            "rx": torch.tensor(rx,   dtype=torch.float32, device=dev),
            "ry": torch.tensor(ry,   dtype=torch.float32, device=dev),
            "nE": torch.tensor(nE,   dtype=torch.float32, device=dev),
            "nN": torch.tensor(nN,   dtype=torch.float32, device=dev),
            "vt": torch.tensor(vt,   dtype=torch.float32, device=dev),
            "rx_np": rx, "ry_np": ry, "s_np": s_np,
        }
        self.cur = 0
        print(f"[MPPI] path  {N} pts  {s_np[-1]:.1f} m  vt={self.V_MAX:.1f}")

    # =====================================================================
    #  compute  — one MPPI planning step
    # =====================================================================
    def compute(self, x, y, yaw, speed, dt_ros):
        if self._path_data is None:
            return 0.0, -self.ACCEL_MAX, {
                "target_speed": 0.0, "cte": 0.0,
                "heading_err": 0.0, "mean_traj": [],
            }

        pd     = self._path_data
        dev    = self.dev
        dt_ros = max(dt_ros, 1e-3)

        # ---- speed estimate (windowed position regression) ----
        self._pose_window.append((dt_ros, x, y))
        window_total = sum(p[0] for p in self._pose_window)
        while self._pose_window and window_total > self.SPD_WINDOW_S + 0.2:
            window_total -= self._pose_window[0][0]
            self._pose_window.popleft()

        reg_spd = 0.0
        if len(self._pose_window) >= 3 and window_total > 0.05:
            ts = np.cumsum([p[0] for p in self._pose_window])
            xs = np.array([p[1] for p in self._pose_window])
            ys = np.array([p[2] for p in self._pose_window])
            tm = ts - ts.mean()
            denom = max(np.dot(tm, tm), 1e-6)
            vx_reg = np.dot(tm, xs - xs.mean()) / denom
            vy_reg = np.dot(tm, ys - ys.mean()) / denom
            reg_spd = math.hypot(vx_reg, vy_reg)

        _speed = max(float(speed), reg_spd)
        v_plan = max(_speed, self.V_PLAN_MIN)

        # ---- yawRate estimate ----
        if self.prev_yaw is not None:
            raw_yr = _angle_diff(yaw, self.prev_yaw) / dt_ros
            self.yaw_rate += 0.3 * (raw_yr - self.yaw_rate)
        self.prev_yaw = yaw

        # ---- closest path point + crosstrack ----
        self.cur = int(np.clip(
            local_closest_index((x, y), pd["rx_np"], pd["ry_np"], self.cur, loop=False),
            0, pd["N"] - 1))
        cte_val, _ = cross_track_error(x, y, pd["rx_np"], pd["ry_np"], self.cur, loop=False)
        ct_abs = abs(cte_val)
        scale  = float(np.clip(1.0 - self.K_CT * min(ct_abs / self.CT0, 1.0), 0.0, 1.0))
        vr_eff = float(np.clip(self.V_MAX * scale, self.V_FLOOR, self.V_MAX))

        # ---- arc-length reference horizon ----
        s_cur  = float(pd["s"][self.cur].item())
        s_max  = float(pd["s"][-1].item())
        pace_m = max(_speed, self.V_PLAN_MIN) * self.DT_EFF
        s_tgt  = torch.clamp(
            s_cur + self.LOOKAHEAD_M
            + torch.arange(self.T, dtype=torch.float32, device=dev) * pace_m,
            0.0, s_max)
        refE = _interp_1d(pd["s"], pd["rx"], s_tgt)
        refN = _interp_1d(pd["s"], pd["ry"], s_tgt)
        nE_t = _interp_1d(pd["s"], pd["nE"], s_tgt)
        nN_t = _interp_1d(pd["s"], pd["nN"], s_tgt)

        # ---- steer samples ----
        eps = self.SIG_STEER * torch.randn(self.K, self.T, device=dev)
        A   = (self.nom + eps).clamp(-self.STEER_MAX, self.STEER_MAX)

        # ---- fill static buffers ----
        self._A.copy_(A)
        self._refE.copy_(refE)
        self._refN.copy_(refN)
        self._nE.copy_(nE_t)
        self._nN.copy_(nN_t)
        self._x0.copy_(torch.tensor(
            [v_plan, 0.0, self.yaw_rate, math.sin(yaw), math.cos(yaw), 0.0],
            dtype=torch.float32, device=dev))
        self._p0.copy_(torch.tensor([x, y], dtype=torch.float32, device=dev))

        # ---- run rollout (CUDA graph replay on GPU, plain call on CPU) ----
        if self.use_graph:
            if self._graph is None:
                self._capture_graph()
            self._graph.replay()
        else:
            self._rollout()
        cost = self._cost

        # ---- softmax → updated nominal ----
        wts     = torch.exp(-(cost - cost.min()) / self.LAMBDA)
        wts     = wts / wts.sum()
        nom_new = (wts.unsqueeze(1) * A).sum(dim=0)
        self.nom = nom_new.unsqueeze(0)

        # ---- first action ----
        smax0 = float(np.clip(
            self.STEER_MAX - (self.STEER_MAX - 0.14) * (_speed - 13.0) / 5.0,
            0.14, self.STEER_MAX))
        steer_out = float(nom_new[0].clamp(-smax0, smax0))

        # warm-start shift
        self.nom = torch.cat([self.nom[:, 1:], self.nom[:, -1:]], dim=1)

        # ---- PI longitudinal ----
        err        = vr_eff - _speed
        self.I_spd = float(np.clip(self.I_spd + err * dt_ros, -self.I_CLAMP, self.I_CLAMP))
        lon0       = float(np.clip(self.KP * err + self.KI * self.I_spd, -1.0, 1.0))
        accel_out  = lon0 * self.ACCEL_MAX

        # ---- telemetry ----
        hd_ref      = path_heading(pd["rx_np"], pd["ry_np"], self.cur, loop=False)
        heading_err = _angle_diff(yaw, hd_ref)
        mean_traj   = self._viz_kinematic(x, y, yaw, steer_out, _speed)

        self._dbg_ctr += 1
        if self._dbg_ctr % 100 == 1:
            span_m = pace_m * (self.T - 1)
            print(f"[MPPI] odom={speed:.2f} reg={reg_spd:.2f} use={_speed:.2f} "
                  f"yr={self.yaw_rate:.3f} cur={self.cur}/{pd['N']} CTE={cte_val:.2f} "
                  f"steer={steer_out:+.3f} accel={accel_out:+.2f} horiz={span_m:.1f}m "
                  f"I={self.I_spd:.2f}")

        return steer_out, accel_out, {
            "target_speed": vr_eff,
            "cte":          cte_val,
            "heading_err":  float(heading_err),
            "mean_traj":    mean_traj,
        }

    # =====================================================================
    #  cheap kinematic viz (no NN)
    # =====================================================================
    def _viz_kinematic(self, x0, y0, yaw0, steer_val, speed_val):
        pts = [(x0, y0)]
        L = self.L_WB_KIN
        v = max(float(speed_val), 1.0)
        for t in range(min(self.T, 40)):
            yr = v * math.tan(float(steer_val)) / L
            yaw_mid = float(yaw0) + 0.5 * yr * self.DT_EFF
            x0 = x0 + v * math.cos(yaw_mid) * self.DT_EFF
            y0 = y0 + v * math.sin(yaw_mid) * self.DT_EFF
            yaw0 = float(yaw0) + yr * self.DT_EFF
            if t % 8 == 0:
                pts.append((x0, y0))
        return pts
