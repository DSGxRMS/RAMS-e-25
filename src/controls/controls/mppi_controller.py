#!/usr/bin/env python3
"""
mppi_controller.py  —  exp09 Neural ODE MPPI for RAMS-e-25 (EUFS Gazebo).

Steer-only MPPI, ported from MATLAB fs_mppi_step.m / fs_mppi_init.m.
Uses the exp09 grey-box Neural ODE (best_model_600.pt) as the rollout model.

BUGS FIXED (catalog from all attempts that failed):
  B1 - shape mismatch crash: np.gradient(heading) (N,) / ds_seg (N-1,).  Removed
       curvature computation entirely — K_FF=0 because 5-11 PP points are too sparse
       for a meaningful curvature estimate.
  B2 - horizon collapse: with DT=0.005, v=2 m/s, T=200, the arc-advance per step
       is 0.01 m → 200 steps span 2 m.  A corner looks straight over 2 m → all K
       steer samples have equal cost → softmax uniform → nom ≈ 0 → car drives straight.
       FIX: DT_EFF=0.02 (4× training dt) → 200×0.02×1.5 = 6 m horizon + LOOKAHEAD = 8 m.
       Local RK4 error O(h⁵) stays negligible (≈5e-12 at h=0.02 vs 5e-15 at h=0.005).
  B3 - SLAM odom twist = 0 always → PI saw no error or integrator saturated blindly.
       FIX: windowed linear regression on (x,y) over last 0.5 s gives real speed.
       Falls back to V_PLAN_MIN if the regression hasn't converged yet (first ~10 calls).
  B4 - Vx=0 blindness: if rollout state Vx=0, all K trajectories stay at start →
       identical cost → nom = weighted noise = 0 → car ignores corners.
       FIX: Vx in rollout initial state is clamped to V_PLAN_MIN so trajectories
       always diverge in position and generate meaningful steering cost.
  B5 - path key too volatile: the PP node slides its 5-10pt window forward every
       frame → (len, first_pt, last_pt) key changes every 2-3 frames → recompute
       cache → PI reset (old bug, already removed).  Tolerated — arc-length
       parameterisation means recomputation is cheap (5-10 pts, numpy-only).
  B6 - speed profile from curvature on 5 pts: generate_velocity_profile() computes
       np.gradient(route_x) three times on 5 points → output near-constant ≈ V_FLOOR.
       FIX: flat V_MAX target.  PI + CTE-scale is enough for safe speed at 2 m/s.

AUTHOR: Soumil (RMS controls team)
"""

import math
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ---- reuse existing controls helpers ----
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


def _world_vel(x_6K):
    """
    Body → world.  ROS ENU: yaw=0 = East (+x), CCW positive.
    Ve = Vx·cos(yaw) − Vy·sin(yaw)
    Vn = Vx·sin(yaw) + Vy·cos(yaw)
    """
    return (x_6K[VX_I] * x_6K[COS_I] - x_6K[VY_I] * x_6K[SIN_I],
            x_6K[VX_I] * x_6K[SIN_I] + x_6K[VY_I] * x_6K[COS_I])


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
    Steer-only MPPI, exp09 Neural ODE rollout.
    Configuration matches the validated MATLAB fs_mppi_init.m except where
    EUFS constraints force differences (noted individually).
    """

    # ---- MPPI parameters (CPU — WSL2 CUDA kernel launch overhead ~0.6ms/op) ----
    K           = 96         # samples (CPU: scales linearly, GPU was O(1) per launch)
    T           = 16         # horizon steps (16*0.03*4=1.9m + LOOKAHEAD ≈ 4m preview)
    DT          = 0.005      # model training dt
    DT_EFF      = 0.03       # effective rollout dt (s)
    USE_EULER   = True       # single MLP call per step (not 4 via RK4)
    LAMBDA      = 1.0
    SIG_STEER   = 0.10        # steer noise (rad)
    STEER_MAX   = 0.349       # 20° — model range
    W_CT        = 2.0         # crosstrack weight (Huber)
    W_HEAD      = 0.5         # pure-pursuit heading weight
    CT_SAT      = 3.0         # Huber knee (m)
    LOOKAHEAD_M = 2.0         # arc-length preview offset (m)

    # ---- longitudinal (PI controller, applied to ACCEL output) ----
    KP          = 2.0         # P gain
    KI          = 0.3         # I gain
    I_CLAMP     = 3.0         # integral windup cap
    V_FLOOR     = 0.5         # min speed target (m/s)
    V_MAX       = 2.0         # max speed target (m/s)
    CT0         = 3.0         # CTE threshold for speed reduction (m)
    K_CT        = 0.5         # speed-reduction gain per metre of CTE
    ACCEL_MAX   = 3.0         # m/s²  (EUFS sim clip: -3 .. 3)

    # ---- planning speed for rollout (the NN needs >= 4 m/s for good yaw authority) ----
    V_PLAN_MIN  = 4.0         # m/s  — floor for rollout state Vx

    # ---- wheelbase for kinematic yaw injection ----
    L_WB_KIN    = 1.535       # m  — used to inject tan(steer)*Vx/L into yawRate

    # ---- windowed speed estimator (B3) ----
    SPD_WINDOW_S = 0.5        # seconds of pose history for linear regression

    def __init__(self, ckpt_path=None):
        # ---- device: CPU ONLY ----
        # WSL2 CUDA passthrough adds ~0.6ms overhead per kernel launch.
        # For this tiny MLP (49k params) with small batch (K≈100),
        # CPU is faster for the iterative rollout because there is zero
        # kernel-launch overhead.
        self.dev = torch.device("cpu")
        print("[MPPI] CPU mode (WSL2 GPU overhead > CPU for tiny MLP+batch)")

        # ---- load checkpoint ----
        cp = ckpt_path or CKPT_PATH
        ck = torch.load(str(cp), map_location=self.dev, weights_only=False)
        self.model = _GB(ck).to(self.dev)
        self.model.eval()
        self._dt_t = torch.tensor(self.DT_EFF, dtype=torch.float32, device=self.dev)
        print(f"[MPPI] Loaded {Path(cp).name}  K={self.K}  T={self.T}  "
              f"dt_eff={self.DT_EFF}s  horizon ~{self.T*self.DT_EFF*1.5:.0f}m (at 1.5 m/s)")

        # ---- warm-start nominal steer ----
        self.nom   = torch.zeros(1, self.T, device=self.dev)

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
        self._pose_window = deque()   # (dt_ros, x, y)

        # ---- debug counter ----
        self._dbg_ctr = 0

    # =====================================================================
    #  update_path  — call every control loop, cheap no-op if unchanged
    # =====================================================================
    def update_path(self, path_pts):
        if not path_pts or len(path_pts) < 3:
            return

        # cheap change detection
        key = (len(path_pts), path_pts[0], path_pts[-1])
        if key == self._path_key:
            return
        self._path_key = key

        pts  = np.array(path_pts, dtype=np.float64)
        rx   = pts[:, 0]
        ry   = pts[:, 1]
        N    = len(rx)

        # ---- arc-length parameterisation ----
        seg  = np.hypot(np.diff(rx), np.diff(ry))
        seg  = np.maximum(seg, 1e-6)          # zero-length guard
        s_np = np.concatenate([[0.0], np.cumsum(seg)])

        # ---- heading + left normal (for signed crosstrack) ----
        dx      = np.gradient(rx)
        dy      = np.gradient(ry)
        heading = np.arctan2(dy, dx)
        nE      = -np.sin(heading)
        nN      =  np.cos(heading)

        # ---- flat speed target (B6: curvature on 5 pts is junk) ----
        vt = np.full(N, self.V_MAX, dtype=np.float64)

        # ---- move to GPU ----
        dev = self.dev
        self._path_data = {
            "N":  N,
            "s":  torch.tensor(s_np, dtype=torch.float32, device=dev),
            "rx": torch.tensor(rx,   dtype=torch.float32, device=dev),
            "ry": torch.tensor(ry,   dtype=torch.float32, device=dev),
            "nE": torch.tensor(nE,   dtype=torch.float32, device=dev),
            "nN": torch.tensor(nN,   dtype=torch.float32, device=dev),
            "vt": torch.tensor(vt,   dtype=torch.float32, device=dev),
            # CPU copies for local_closest_index / cross_track_error
            "rx_np": rx,  "ry_np": ry,  "s_np": s_np,
        }
        self.cur = 0          # reset progress to start of new path
        print(f"[MPPI] path  {N} pts  {s_np[-1]:.1f} m  vt={self.V_MAX:.1f}")

    # =====================================================================
    #  compute  — one MPPI planning step
    # =====================================================================
    def compute(self, x, y, yaw, speed, dt_ros):
        """
        Returns  (steer_rad, accel_m_s2, info_dict)
        """
        # ---- no path yet → brake ----
        if self._path_data is None:
            return 0.0, -self.ACCEL_MAX, {
                "target_speed": 0.0, "cte": 0.0,
                "heading_err": 0.0, "mean_traj": [],
            }

        pd     = self._path_data
        dev    = self.dev
        dt_ros = max(dt_ros, 1e-3)

        # ==================================================================
        #  SPEED ESTIMATE  (B3)
        # ==================================================================
        # SLAM /slam/odom publishes twist = 0 always.  We derive speed from
        # the (x,y) pose history by linear regression over SPD_WINDOW_S sec.
        # This is robust to slow SLAM updates because the regression line
        # averages out the per-frame jitter.
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

        # use the best available speed estimate
        _speed = max(float(speed), reg_spd)
        # planning speed: floor for the rollout state Vx (gives NN yaw authority)
        v_plan = max(_speed, self.V_PLAN_MIN)

        # ==================================================================
        #  STATE BRIDGE  odom → 6-D Neural ODE state  (ENU convention)
        # ==================================================================
        # yawRate: finite-difference + low-pass
        if self.prev_yaw is not None:
            raw_yr = _angle_diff(yaw, self.prev_yaw) / dt_ros
            self.yaw_rate += 0.3 * (raw_yr - self.yaw_rate)   # low-pass
        self.prev_yaw = yaw

        # yaw_sin = sin(yaw),  yaw_cos = cos(yaw)
        x6 = torch.tensor(
            [v_plan, 0.0, self.yaw_rate,
             math.sin(yaw), math.cos(yaw), 0.0],
            dtype=torch.float32, device=dev,
        )

        # ==================================================================
        #  CLOSEST PATH POINT  +  CROSSTRACK
        # ==================================================================
        self.cur = int(np.clip(
            local_closest_index((x, y), pd["rx_np"], pd["ry_np"],
                                self.cur, loop=False),
            0, pd["N"] - 1,
        ))

        cte_val, _ = cross_track_error(
            x, y, pd["rx_np"], pd["ry_np"], self.cur, loop=False,
        )
        ct_abs = abs(cte_val)
        scale  = float(np.clip(
            1.0 - self.K_CT * min(ct_abs / self.CT0, 1.0), 0.0, 1.0,
        ))
        vr_eff = float(np.clip(self.V_MAX * scale, self.V_FLOOR, self.V_MAX))

        # ==================================================================
        #  ARC-LENGTH REFERENCE HORIZON  (B2: dt_eff → 8m preview)
        # ==================================================================
        s_cur  = float(pd["s"][self.cur].item())
        s_max  = float(pd["s"][-1].item())
        pace_m = _speed * self.DT_EFF
        s_tgt  = torch.clamp(
            s_cur + self.LOOKAHEAD_M
            + torch.arange(self.T, dtype=torch.float32, device=dev) * pace_m,
            0.0, s_max,
        )

        refE = _interp_1d(pd["s"], pd["rx"], s_tgt)
        refN = _interp_1d(pd["s"], pd["ry"], s_tgt)
        nE_t = _interp_1d(pd["s"], pd["nE"], s_tgt)
        nN_t = _interp_1d(pd["s"], pd["nN"], s_tgt)
        vtgt = _interp_1d(pd["s"], pd["vt"], s_tgt)

        # ==================================================================
        #  STEER SAMPLING
        # ==================================================================
        eps = self.SIG_STEER * torch.randn(self.K, self.T, device=dev)
        A   = (self.nom + eps).clamp(-self.STEER_MAX, self.STEER_MAX)

        # ==================================================================
        #  GPU ROLLOUT
        # ==================================================================
        xk   = x6.unsqueeze(1).expand(6, self.K).clone()
        pk   = torch.tensor([[x], [y]], dtype=torch.float32,
                             device=dev).expand(2, self.K).clone()
        cost = torch.zeros(self.K, device=dev)

        with torch.no_grad():
            for t in range(self.T):
                vx = xk[VX_I]

                # speed-dependent steer cap  (MATLAB fs_mppi_step.m line 47)
                smax = (self.STEER_MAX
                        - (self.STEER_MAX - 0.14) * (vx - 13.0) / 5.0
                        ).clamp(0.14, self.STEER_MAX)

                steer = A[:, t].clamp(-smax, smax)           # (K,)   no feedforward
                lon   = torch.zeros(self.K, device=dev)       # (K,)   KP_ROLL = 0

                # --- kinematic yaw-rate injection (critical fix) ---
                # The NN was trained on steady-state cornering data where yawRate
                # was already ~ Vx*tan(steer)/L.  The NN's dyawRate residual is a
                # small correction, not the primary yaw driver.  Without injecting
                # the kinematic rate, the NN cannot steer the car from zero yawRate.
                yr_kin = vx * (steer / self.L_WB_KIN).clamp(-self.STEER_MAX, self.STEER_MAX)  # tan(θ)≈θ at small angles → OK for clamps
                xk[YR_I] = yr_kin                               # overwrite yawRate in state

                u_batch = torch.stack([steer, lon], dim=0)     # (2, K)

                Ve1, Vn1 = _world_vel(xk)

                # Neural ODE step — pass (K,6) and (K,2) directly, no reshape
                xk     = self.model.step(xk.T, u_batch.T, self._dt_t,
                                         euler=self.USE_EULER).T

                Ve2, Vn2 = _world_vel(xk)

                # midpoint position integration
                pk[0] += 0.5 * (Ve1 + Ve2) * self.DT_EFF
                pk[1] += 0.5 * (Vn1 + Vn2) * self.DT_EFF

                # ---- C1: Huber crosstrack ----
                perp = ((pk[0] - refE[t]) * nE_t[t]
                      + (pk[1] - refN[t]) * nN_t[t])
                ap   = perp.abs()
                ctc  = (ap.clamp(max=self.CT_SAT) ** 2
                      + 2 * self.CT_SAT * (ap - self.CT_SAT).clamp(min=0.0))

                # ---- C2: pure-pursuit heading ----
                car_th = torch.atan2(xk[SIN_I], xk[COS_I])
                bear   = torch.atan2(refN[t] - pk[1], refE[t] - pk[0])
                dh     = _angle_diff(car_th, bear)

                cost += self.W_CT * ctc + self.W_HEAD * (dh * dh)

        cost = cost / self.T

        # ==================================================================
        #  SOFTMAX → UPDATED NOMINAL
        # ==================================================================
        wts     = torch.exp(-(cost - cost.min()) / self.LAMBDA)
        wts     = wts / wts.sum()
        nom_new = (wts.unsqueeze(1) * A).sum(dim=0)          # (T,)
        self.nom = nom_new.unsqueeze(0)

        # ==================================================================
        #  FIRST ACTION
        # ==================================================================
        smax0 = float(np.clip(
            self.STEER_MAX - (self.STEER_MAX - 0.14) * (_speed - 13.0) / 5.0,
            0.14, self.STEER_MAX,
        ))
        steer_out = float(nom_new[0].clamp(-smax0, smax0))

        # warm-start: shift nominal left by 1
        self.nom = torch.cat([self.nom[:, 1:], self.nom[:, -1:]], dim=1)

        # ==================================================================
        #  PI LONGITUDINAL
        # ==================================================================
        err        = vr_eff - _speed
        self.I_spd = float(np.clip(
            self.I_spd + err * dt_ros, -self.I_CLAMP, self.I_CLAMP,
        ))
        lon0      = float(np.clip(self.KP * err + self.KI * self.I_spd, -1.0, 1.0))
        accel_out = lon0 * self.ACCEL_MAX

        # ==================================================================
        #  TELEMETRY
        # ==================================================================
        hd_ref      = path_heading(pd["rx_np"], pd["ry_np"], self.cur, loop=False)
        heading_err = _angle_diff(yaw, hd_ref)

        # viz: simple kinematic forward projection (no NN — too expensive every frame)
        mean_traj = self._viz_kinematic(x, y, yaw, steer_out, _speed)

        # debug every ~100 calls
        self._dbg_ctr += 1
        if self._dbg_ctr % 100 == 1:
            span_m = pace_m * (self.T - 1)
            print(f"[MPPI] odom={speed:.2f}  reg={reg_spd:.2f}  "
                  f"use={_speed:.2f}  yr={self.yaw_rate:.3f}  "
                  f"cur={self.cur}/{pd['N']}  CTE={cte_val:.2f}  "
                  f"steer={steer_out:+.3f}  accel={accel_out:+.2f}  "
                  f"horiz={span_m:.1f}m  I={self.I_spd:.2f}")

        return steer_out, accel_out, {
            "target_speed": vr_eff,
            "cte":          cte_val,
            "heading_err":  float(heading_err),
            "mean_traj":    mean_traj,
        }

    # =====================================================================
    #  cheap kinematic viz (no NN — compute budget is tight)
    # =====================================================================
    def _viz_kinematic(self, x0, y0, yaw0, steer_val, speed_val):
        pts = [(x0, y0)]
        L = self.L_WB_KIN
        v = max(float(speed_val), 1.0)
        for t in range(min(self.T, 20)):
            yr = v * math.tan(float(steer_val)) / L
            yaw_mid = float(yaw0) + 0.5 * yr * self.DT_EFF
            x0 = x0 + v * math.cos(yaw_mid) * self.DT_EFF
            y0 = y0 + v * math.sin(yaw_mid) * self.DT_EFF
            yaw0 = float(yaw0) + yr * self.DT_EFF
            if t % 4 == 0:
                pts.append((x0, y0))
        return pts
