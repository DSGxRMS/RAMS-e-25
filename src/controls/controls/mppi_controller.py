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
  and replayed as a SINGLE GPU submission (K=1024,T=200 -> ~80 ms in WSL2,
  bit-identical to the plain rollout).  Stays fully on GPU, full sample count.

DESIGN NOTES (EUFS-specific):
  - Speed comes from /ground_truth/odom twist (SLAM odom twist is 0) and is fed
    straight into the rollout state Vx (exp10 model is valid 0-13.9 m/s; no floor).
  - PP path is a sparse 5-11 pt rolling window -> B-spline smoothed + densely
    resampled, giving a clean arc-length reference and analytic curvature.
  - Steering = K_FF * spline-curvature feedforward + MPPI sampled correction.
  - The rollout steps the grey-box model with [steer, longitudinal] only; yaw rate
    evolves from the learned dynamics (no per-step kinematic yaw injection). The
    measured yaw rate seeds only the INITIAL rollout state.

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

from controls.control_utils import local_closest_index, cross_track_error, path_heading

# =========================================================================
# Paths — find the model wherever it actually exists. colcon does NOT copy
# the .pt into install/, so the install-tree copy of this file would otherwise
# look in a dir that has no model. Search a few known locations and use the
# first that exists (the source tree always has it).
# =========================================================================
_HERE = Path(__file__).parent

def _find_ckpt():
    cands = [
        _HERE / "fs_model" / "best_model_600.pt",                       # alongside this file
        Path("/mnt/d/RAMS-e-25/src/controls/controls/fs_model/best_model_600.pt"),
        Path(__file__).resolve().parents[3] / "src" / "controls" / "controls" / "fs_model" / "best_model_600.pt",
    ]
    for c in cands:
        if c.exists():
            return c
    # last resort: return the first (will raise a clear FileNotFoundError)
    return cands[0]

CKPT_PATH = _find_ckpt()

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
    Body → world, EXACTLY matching MATLAB fs_mppi_step.m `wv` (lines 81-84).
    The TRAINING convention stores the heading slots as:
        yaw_sin_slot (SIN_I) = -cos(theta)
        yaw_cos_slot (COS_I) =  sin(theta)
    (verified: training data row 1 has yaw_sin=-1, yaw_cos=0 at theta=0.)
    MATLAB wv:  Ve = -(vx*S + vy*C),  Vn = (vx*C - vy*S)
    Substituting S=-cos, C=sin gives standard ENU (Ve=vx*cos-vy*sin, etc.).
    """
    S = x_K6[:, SIN_I]; C = x_K6[:, COS_I]
    vx = x_K6[:, VX_I]; vy = x_K6[:, VY_I]
    return (-(vx * S + vy * C), (vx * C - vy * S))


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
    # Values restored to the VALIDATED fs_mppi_mitl.py reference (drift fixed).
    K           = 1024        # samples (full count)
    # MATLAB-EXACT rollout granularity: the model has num_rk4_steps=1, so each rollout
    # step is ONE RK4 eval of size DT_EFF. The dataset transitions are 0.005 s apart, so
    # DT_EFF MUST be 0.005 to integrate at training granularity (was 0.02 = 4x coarse ->
    # inaccurate rolled-out trajectories -> MPPI scored samples against a wrong prediction
    # -> turned in late / ran wide). T=200 keeps a 1.0 s horizon, identical to MATLAB
    # fs_mppi (o.T=200, o.dt=0.005). ~2x the compute of T=100 (~80 ms in WSL2), still fine.
    T           = 200         # horizon steps (200 * 0.005 = 1.0 s rollout)  [MATLAB o.T]
    DT_EFF      = 0.005       # rollout dt = training dt (num_rk4_steps=1)    [MATLAB o.dt]
    USE_EULER   = False       # RK4 (full accuracy; CUDA graph makes it affordable)
    LAMBDA_FRAC = 0.15        # adaptive softmax temp = LAMBDA_FRAC * (cost_max-cost_min)
    LAMBDA_MIN  = 1e-3        # floor so lam never hits 0 when all costs identical
    SIG_STEER   = 0.08        # per-step steer JITTER (rad)  [MITL] (fine exploration)
    # ---- sustained per-sample bias (this is what makes MPPI actually work) -------
    # Pure per-step white noise (the old code) averages out over T=200 steps: a
    # sample's NET sustained lean is only SIG_STEER/sqrt(T) ~= 0.006 rad, so all 1024
    # rollouts drove to ~the same place -> identical cost -> nEff~1024 -> MPPI dead.
    # SIG_BIAS gives each sample ONE steady steer offset held across the whole horizon
    # (some lean left, some right), so the rollouts fan out into genuinely different
    # trajectories -> the cost gets a real gradient -> the softmax can pick the plan
    # that hugs the path. This is the lever that makes MPPI carry the steering instead
    # of the feedforward. Tune: too small -> MPPI stays weak (nEff near 1024); too big
    # -> jittery / over-exploring. Watch nEff DROP well below 1024 once this is on.
    SIG_BIAS    = 0.05        # sustained per-sample steer bias (rad)
                              # 0.12 woke MPPI but TOO aggressively: it made bold
                              # full-lock commits and, when the path reference was even
                              # slightly off, committed boldly the WRONG way -> violent
                              # left/right oscillation, off-track in seconds. 0.05 keeps
                              # MPPI discriminating (nEff still drops below 1024) but its
                              # corrections are gentle. Ramp UP only if stable & too weak.
    STEER_MAX   = 0.349       # 20° — model range
    W_CT        = 1.0         # crosstrack weight (Huber)    [MITL]
    W_HEAD      = 0.3         # pure-pursuit heading weight  [MITL]
    CT_SAT      = 5.0         # Huber knee (m)               [MITL]
    LOOKAHEAD_M = 3.0         # FIXED arc-length preview offset (m). Bigger = the feedforward
                              # reads the curve sooner and turns in EARLY (cuts the inside);
                              # smaller = turns late (runs wide). At V_MAX=2.5 the rollout only
                              # reaches ~2.5 m, so lookahead must be near that, not beyond it.
                              # History: 5 m & 4 m both cut inside ~3 m at speed; 1.5-2.5 ran
                              # wide BUT only because the speed collapsed (now fixed via the
                              # softer CT0/K_CT). 3 m ~= the planning reach. DO NOT make this
                              # speed-adaptive again; tune the fixed value if it cuts/runs wide.
    K_FF        = 0.85        # curvature feedforward gain (MATLAB o.k_ff)
    SPLINE_NPTS   = 40        # dense points resampled from the B-spline
    SPLINE_SMOOTH = 0.12      # spline smoothing per point. Raised 0.05->0.12: the rolling-
                              # window accumulator keeps slightly-offset centreline points from
                              # different PP frames, so a near-interpolating spline wiggled and
                              # the curvature spiked -> feedforward slammed the wheel. More
                              # smoothing irons out sub-~0.35 m point noise while keeping real
                              # corner shape (corners are bigger features than the noise).

    # ---- longitudinal (PI controller, applied to ACCEL output) ----
    KP          = 0.50        # raised 0.35->0.50: tighter P tracking so actual hugs target
                              # (on GT there's no sensor noise to amplify, so a firmer P is safe)
    KI          = 0.10
    I_CLAMP     = 1.5         # was 5.0. The integral could reach KI*5=0.5 of full throttle —
                              # a HUGE positive bias that wound up during launch / straights and
                              # then RESISTED braking when the corner target dropped -> actual
                              # overshot target by up to 1.9 m/s. 1.5 -> max KI*I=0.15 trim only.
    I_BAND      = 1.0         # only integrate when |target-actual| < this. The integral's job is
                              # steady-state trim near the setpoint; during big transients (launch,
                              # corner-exit accel) P handles it and the integral must NOT wind up.
    KP_ROLL     = 0.40        # P-only longitudinal INSIDE the rollout (track target speed)
    V_CREEP     = 1.0         # tiny hard floor so the car never fully stops on a live path
    HORIZON_CAP_FRAC = 0.5    # cap speed so 50% of the horizon fits in visible path
    STOP_PATH_M = 1.5         # if remaining path < this, command a FULL STOP (don't drive blind)
    BRAKE_A     = 2.5         # m/s^2 braking decel used for the stop-at-path-end ramp
    V_MAX       = 3.0         # max speed (m/s). Moderate test speed now that the model is
                              # actually loaded (was running on random weights — see
                              # model-weights-never-loaded). 3 m/s gives a clean read of the
                              # now-working MPPI without the "outrun the ~10 m PP path" risk
                              # that 5 m/s had. Bump toward 5 for the senior's high-speed test
                              # once low/mid speed is confirmed good. exp10 model valid 0-12 m/s.
    # CTE -> speed reduction. SOFTENED from MITL (CT0=4, K_CT=0.7) to break a death
    # spiral: at low V_MAX the old values crushed speed to ~0.75 m/s when off-line, and
    # at that speed MPPI can only plan ~1 m ahead, so it cut the corner MORE -> more CTE
    # -> even slower -> a 1-2 m cut snowballed to 6 m through the cones. CT0=8/K_CT=0.4
    # keeps ~2 m/s even at 4 m off-line, so MPPI can still see far enough to steer back.
    CT0         = 8.0         # CTE (m) at which the speed reduction reaches its max
    K_CT        = 0.4         # how hard to slow per unit CTE (gentler than MITL 0.7)
    ACCEL_MAX   = 3.0         # m/s²  (EUFS sim clip: -3 .. 3)
    HD_BAD_RAD  = 1.4         # ~80°: if path heading at car is more off than this,
                              # the path is one-sided-cone garbage -> stop, don't chase
    A_LAT_MAX   = 2.5         # m/s² lateral budget for the per-point corner SPEED PROFILE
                              # v_prof = min(V_MAX, sqrt(A_LAT_MAX/|kappa|)) + backward
                              # braking pass (ports MATLAB fs_mppi_init.m vt mechanism).

    # ---- planning speed for rollout ----
    # exp10 model is valid 0-13.9 m/s (x_min[Vx]=0); feed the REAL measured speed into
    # the rollout state Vx, only capping the TOP at the model's training ceiling.
    V_PLAN_MAX  = 13.0        # exp10 training ceiling (~13.9) — never feed Vx above this
    SPEED_SANE_MAX = 16.0     # reject absurd odom readings
    L_WB_KIN    = 1.535       # wheelbase for kinematic yaw injection

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
        self.model = _GB(ck)
        # CRITICAL: load the TRAINED weights. _GB.__init__ only sets up the
        # normalisation buffers + a RANDOM _Func; WITHOUT this load the Neural ODE runs
        # on random weights, so the rollout ignores steering completely (every MPPI
        # sample rolls out the same -> nEff=1024 "MPPI dead", and whenever MPPI did act
        # it was trusting a random world-model -> steered the wrong way). model_state_dict
        # holds both the norm buffers and the func.net.* weights; its keys line up exactly
        # with _GB.state_dict(), so this one call restores the real model.
        self.model.load_state_dict(ck["model_state_dict"])
        self.model = self.model.to(self.dev)
        self.model.eval()
        # sanity: trained net should NOT look like default init (std ~0.2)
        _w_std = float(self.model.func.net[0].weight.std())
        print(f"[MPPI] model weights loaded (net[0].weight std={_w_std:.3f})")
        self._dt_t = torch.tensor(self.DT_EFF, dtype=torch.float32, device=self.dev)

        # ---- static buffers for the rollout (fixed addresses for CUDA graph) ----
        dev = self.dev
        self._A      = torch.zeros(self.K, self.T, device=dev)   # steer samples
        self._refE   = torch.zeros(self.T, device=dev)
        self._refN   = torch.zeros(self.T, device=dev)
        self._nE     = torch.zeros(self.T, device=dev)
        self._nN     = torch.zeros(self.T, device=dev)
        self._vtgt   = torch.zeros(self.T, device=dev)           # target speed over horizon
        self._ff     = torch.zeros(self.T, device=dev)           # curvature feedforward steer
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
        self._ahead_lp = 8.0          # smoothed path-ahead (m), seeded mid-range
        self._vr_lp    = self.V_CREEP # smoothed target speed (rate-limited down)

        # ---- path cache ----
        self._path_key  = None
        self._path_data = None
        self.cur        = 0

        # ---- yaw-rate (finite diff + low-pass) ----
        self.prev_yaw = None
        self.yaw_rate = 0.0

        self._dbg_ctr = 0

        # ---- optional CSV trace ----
        self._csv = None
        csv_path = os.environ.get("MPPI_CSV", "")
        if csv_path:
            self._csv = open(csv_path, "w")
            self._csv.write(
                "time,x,y,yaw,odom_spd,use_spd,v_plan,vr_eff,steer,accel,lon0,"
                "cte,heading_err,cost_min,cost_max,n_eff,a0_std,pace_m,roll_ms,"
                "I_spd,yaw_rate,cur,N\n")
            print(f"[MPPI] CSV trace -> {csv_path}", flush=True)

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
            # C3: steer = curvature feedforward + sampled correction (MATLAB fs_mppi_step.m:48)
            steer = (self.K_FF * self._ff[t] + self._A[:, t]).clamp(-smax, smax)   # (K,)

            # P-only longitudinal INSIDE the rollout (matches MATLAB fs_mppi_step.m:46)
            lon = (self.KP_ROLL * (self._vtgt[t] - vx)).clamp(-1.0, 1.0)   # (K,)
            u   = torch.stack([steer, lon], dim=1)               # (K,2)
            # NOTE: NO kinematic yaw injection — MATLAB just steps the grey-box model
            # with [steer; lon]. Yaw rate evolves from the learned dynamics alone.

            Ve1, Vn1 = _world_vel(xk)
            xk = self.model.step(xk, u, self._dt_t, euler=self.USE_EULER)
            Ve2, Vn2 = _world_vel(xk)

            pe = pe + 0.5 * (Ve1 + Ve2) * self.DT_EFF
            pn = pn + 0.5 * (Vn1 + Vn2) * self.DT_EFF

            # C1: Huber crosstrack
            perp = (pe - self._refE[t]) * self._nE[t] + (pn - self._refN[t]) * self._nN[t]
            ap   = perp.abs()
            ctc  = ap.clamp(max=self.CT_SAT)**2 + 2*self.CT_SAT*(ap - self.CT_SAT).clamp(min=0.0)

            # C2: pure-pursuit heading.  carTh from the TRAINING heading slots:
            # MATLAB carTh = atan2(yaw_cos_slot, -yaw_sin_slot) = atan2(sin th, cos th) = th.
            car_th = torch.atan2(xk[:, COS_I], -xk[:, SIN_I])
            dE = self._refE[t] - pe
            dN = self._refN[t] - pn
            bear   = torch.atan2(dN, dE)
            dh     = torch.atan2(torch.sin(car_th - bear), torch.cos(car_th - bear))
            # Gate the heading term when the preview point is ~on the car: the bearing
            # is meaningless there and would inject huge spurious heading error -> spin.
            dist = torch.sqrt(dE * dE + dN * dN)
            head_gate = (dist > 1.0).float()
            cost = cost + self.W_CT * ctc + self.W_HEAD * head_gate * (dh * dh)

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

        raw = np.array(path_pts, dtype=np.float64)
        rx0, ry0 = raw[:, 0], raw[:, 1]
        N0 = len(rx0)

        # ---- B-SPLINE smoothing (senior's fix) --------------------------------
        # PP gives only ~5-6 sparse points. Using them raw makes the reference
        # jagged and the curvature (-> feedforward) noisy. Fit a smooth B-spline
        # through the points, then RESAMPLE densely so the MPPI sees a continuous
        # path and we can read CLEAN analytic curvature for the feedforward.
        try:
            # de-duplicate near-identical points (splprep fails on repeats)
            keep = np.concatenate([[True], np.hypot(np.diff(rx0), np.diff(ry0)) > 1e-3])
            rxk, ryk = rx0[keep], ry0[keep]
            k = min(3, len(rxk) - 1)                  # cubic if >=4 pts, else lower
            if k >= 1 and len(rxk) >= 2:
                # s>0 = smoothing spline (tolerates PP jitter); scale with #points
                tck, _ = splprep([rxk, ryk], s=self.SPLINE_SMOOTH * len(rxk), k=k)
                u = np.linspace(0.0, 1.0, self.SPLINE_NPTS)
                rx, ry = splev(u, tck)
                dx1, dy1 = splev(u, tck, der=1)       # 1st derivative
                dx2, dy2 = splev(u, tck, der=2)       # 2nd derivative
                rx, ry = np.asarray(rx), np.asarray(ry)
                # analytic signed curvature kappa = (x'y'' - y'x'') / (x'^2+y'^2)^1.5
                denom = np.power(dx1*dx1 + dy1*dy1, 1.5) + 1e-9
                kap = (dx1*dy2 - dy1*dx2) / denom
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

        N = len(rx)
        seg  = np.maximum(np.hypot(np.diff(rx), np.diff(ry)), 1e-6)
        s_np = np.concatenate([[0.0], np.cumsum(seg)])

        nE   = -np.sin(heading)
        nN   =  np.cos(heading)

        # ---- C3 curvature feedforward from the SMOOTH spline curvature ----
        # Clip tightened 0.5->0.3: the track's tightest corner is ~R>3 m (|kap|<0.3),
        # so 0.5 (R=2 m) was pure headroom that only let curvature NOISE spike the
        # feedforward to near full lock (ff=atan(1.535*0.5)=0.65). 0.3 caps ff at 0.43
        # without clipping any real corner, killing the spurious hard-steer spikes.
        kap = np.clip(kap, -0.3, 0.3)
        ff_steer = np.arctan(self.L_WB_KIN * kap)

        # ---- CORNER SPEED PROFILE (ports MATLAB fs_mppi_init.m:16-25) ----------
        # Per-point target speed: lateral-accel limit v=sqrt(a_lat/|kappa|), capped at
        # V_MAX, then a BACKWARD braking pass v_i <= sqrt(v_{i+1}^2 + 2*a_brake*ds) so
        # the car slows DOWN BEFORE the corner instead of arriving too fast and running
        # wide. This is the real mechanism the previous reactive caps were faking.
        vt = np.minimum(self.V_MAX,
                        np.sqrt(self.A_LAT_MAX / np.maximum(np.abs(kap), 1e-3)))
        for _ in range(2):                                   # iterate to converge
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
            "kap_np": np.asarray(kap, dtype=np.float64),   # signed curvature per pt
            "vt_np":  np.asarray(vt,  dtype=np.float64),   # corner speed profile per pt
        }
        self.cur = 0
        print(f"[MPPI] path  {N0}->{N} pts (spline)  {s_np[-1]:.1f} m  vt={self.V_MAX:.1f}")

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

        # ---- speed estimate ----
        # Use the GT twist speed DIRECTLY (passed in as `speed` from ros_connect's
        # gt_speed). The gt_vs_slam diagnostic proved it is clean and accurate
        # (Vx climbs 0->32 smoothly, Vy=0). The old windowed position-regression
        # spiked to 25 m/s on position jumps and poisoned the rollout — removed.
        _speed = float(np.clip(float(speed), 0.0, self.SPEED_SANE_MAX))
        # Feed the REAL measured speed into the rollout state Vx (exp10 is valid down
        # to 0 m/s, so NO vplan "lie" needed). Only clamp the very top to the model's
        # training ceiling so we never extrapolate above it.
        v_plan = float(np.clip(_speed, 0.0, self.V_PLAN_MAX))

        # ---- yawRate estimate ----
        if self.prev_yaw is not None:
            raw_yr = _angle_diff(yaw, self.prev_yaw) / dt_ros
            self.yaw_rate += 0.3 * (raw_yr - self.yaw_rate)
        self.prev_yaw = yaw

        # ---- closest path point + crosstrack ----
        # ROLLING-WINDOW mode: PP gives a fresh ~6-pt window each frame whose
        # point-0 is the car. We do NOT march a global `cur`; we just find the
        # closest point in the current window (loop=False, full search).
        self.cur = int(np.clip(
            local_closest_index((x, y), pd["rx_np"], pd["ry_np"], 0, loop=False),
            0, pd["N"] - 1))
        cte_val, _ = cross_track_error(x, y, pd["rx_np"], pd["ry_np"], self.cur, loop=False)
        ct_abs = abs(cte_val)

        # ---- BAD-PATH GUARD (the real one-sided-cone failure) -----------------
        # The stop-on-EMPTY-path rule never fires when one-sided cones make PP emit
        # a full-length but WRONG path (heading ~100 deg off). The car then chases
        # that garbage reference and slides metres off track before geometry trips
        # the path-end brake. So: if the path heading at the car is wildly off the
        # car's heading, treat the path as UNUSABLE -> command a stop (coast down)
        # instead of following it. |hd| > HD_BAD_RAD means "this path points the
        # wrong way; do not chase it."
        _hd_ref_now = path_heading(pd["rx_np"], pd["ry_np"], self.cur, loop=False)
        _hd_off     = abs(_angle_diff(yaw, _hd_ref_now))
        path_is_bad = (_hd_off > self.HD_BAD_RAD)

        # ---- CLEAN HAND-OFF: bail out early on a bad path ----------------------
        # If the path is garbage (points the wrong way), the recovery in control_node
        # takes over and ignores whatever we return here. So DON'T run the heavy rollout
        # and DON'T touch any of MPPI's carried-over "memory":
        #     self.nom    -> the warm-started steering plan
        #     self._vr_lp -> the smoothed target speed
        #     self.I_spd  -> the speed integral
        # All three are updated only LATER in this function, so returning now FREEZES them
        # exactly as they were on the last GOOD frame. That way, when the path becomes good
        # again and control hands the wheel back to MPPI, MPPI resumes from its last good
        # plan instead of one quietly corrupted by the garbage path -> no confused transition.
        if path_is_bad:
            return 0.0, 0.0, {
                "target_speed": 0.0,
                "cte":          cte_val,
                "heading_err":  float(_angle_diff(yaw, _hd_ref_now)),
                "mean_traj":    [],
                "path_bad":     True,
            }

        # ---- horizon capped to AVAILABLE path length (anti-outrun) ----
        # The car must never plan/drive beyond the ~12 m it can actually see, else
        # it aims at nothing, locks steering, and runs off in circles.
        s_cur     = float(pd["s"][self.cur].item())
        s_max     = float(pd["s"][-1].item())
        # FIXED preview (reverted from speed-adaptive). The speed-adaptive version shrank
        # the lookahead to its floor exactly when the car SLOWED into a corner -> it lost
        # anticipation -> turned in too LATE -> ran WIDE and diverged (CTE +10). That was
        # the same failure short lookahead always caused. A fixed preview is stable and
        # never collapses mid-corner; only clamp it to half the visible path so a short PP
        # window can't push the reference past the path end.
        lookahead_eff = min(self.LOOKAHEAD_M, 0.5 * max(s_max - s_cur, 0.0))
        path_ahead = max(s_max - s_cur - lookahead_eff, 0.5)   # metres of path left
        # speed that keeps the *first part* of the horizon within the visible path.
        # Use a fraction of the horizon (not full T) so the cap isn't overly strict
        # but still slows the car hard when little path remains.
        # SMOOTH path_ahead: PP path length jitters frame-to-frame (5<->9 pts) as
        # cones enter/leave camera FOV. Without smoothing, one short frame slams the
        # speed cap to ~0 and the car stop-starts. Use a running max-ish low-pass so
        # a transient dip is ignored but a SUSTAINED short path still slows the car.
        self._ahead_lp += 0.25 * (path_ahead - self._ahead_lp)   # ~4-frame smoothing
        ahead_eff = self._ahead_lp
        v_path_cap = ahead_eff / (self.HORIZON_CAP_FRAC * self.T * self.DT_EFF)
        # CTE-based reduction (slow when off-line).
        scale  = float(np.clip(1.0 - self.K_CT * min(ct_abs / self.CT0, 1.0), 0.0, 1.0))
        # BASE speed = the CORNER SPEED PROFILE at the current point (already accounts for
        # the curvature ahead via the backward braking pass, so the car is slow BEFORE the
        # corner). Then CTE scaling, path-visibility cap, and stop-at-end on top.
        vt_np   = pd.get("vt_np")
        v_corner = float(vt_np[self.cur]) if (vt_np is not None and len(vt_np)) else self.V_MAX
        vr_uncapped = v_corner * scale
        v_brake_cap = math.sqrt(2.0 * self.BRAKE_A * max(path_ahead, 0.0))
        vr_raw = min(vr_uncapped, v_path_cap, v_brake_cap)
        # below STOP_PATH_M of remaining path, command a full stop (don't creep blind)
        if path_ahead <= self.STOP_PATH_M:
            vr_raw = 0.0
        # OR if the path points the wrong way (one-sided-cone garbage), stop too —
        # chasing a ~100-deg-off reference is what slid the car 9.7 m off track.
        if path_is_bad:
            vr_raw = 0.0
        vr_raw = float(np.clip(vr_raw, 0.0, self.V_MAX))
        # Rate-limit DOWN moves so a brief cap dip can't stop the car mid-turn; allow
        # fast UP moves (safe). This kills the stop-start stutter.
        if vr_raw < self._vr_lp:
            self._vr_lp += 0.15 * (vr_raw - self._vr_lp)   # slow to slow down
        else:
            self._vr_lp += 0.6 * (vr_raw - self._vr_lp)    # quick to speed up
        vr_eff = float(self._vr_lp)

        # ---- arc-length reference horizon ----
        # pace marches reference points along the path per rollout step at the planning
        # speed (so the rollout reference matches the speed we intend), but never beyond
        # the visible path (clamped to s_max).
        pace_m = max(vr_eff, 0.5) * self.DT_EFF
        s_tgt  = torch.clamp(
            s_cur + lookahead_eff
            + torch.arange(self.T, dtype=torch.float32, device=dev) * pace_m,
            0.0, s_max)
        refE = _interp_1d(pd["s"], pd["rx"], s_tgt)
        refN = _interp_1d(pd["s"], pd["ry"], s_tgt)
        nE_t = _interp_1d(pd["s"], pd["nE"], s_tgt)
        nN_t = _interp_1d(pd["s"], pd["nN"], s_tgt)
        ff_t = _interp_1d(pd["s"], pd["ff"], s_tgt)
        # per-horizon target speed = corner profile along the reference (so the rollout's
        # P-longitudinal anticipates the corner too, matching MATLAB vtgt=Vr_eff(ti))
        vtgt_t = _interp_1d(pd["s"], pd["vt"], s_tgt)

        # ---- steer samples: sustained per-sample bias + fine per-step jitter ----
        # bias: ONE steady offset per sample, held across the whole horizon -> the
        #       1024 rollouts fan out into truly different trajectories (the thing that
        #       makes the cost discriminate and MPPI actually steer). See SIG_BIAS note.
        # jit:  small per-step exploration on top, for fine variation.
        bias = self.SIG_BIAS  * torch.randn(self.K, 1, device=dev)
        jit  = self.SIG_STEER * torch.randn(self.K, self.T, device=dev)
        eps  = bias + jit
        A    = (self.nom + eps).clamp(-self.STEER_MAX, self.STEER_MAX)

        # ---- fill static buffers ----
        self._A.copy_(A)
        self._refE.copy_(refE)
        self._refN.copy_(refN)
        self._nE.copy_(nE_t)
        self._nN.copy_(nN_t)
        self._ff.copy_(ff_t)
        # target speed over the horizon = corner profile, but never above the speed we'll
        # actually command this cycle (vr_eff), so the rollout's P-longitudinal (KP_ROLL)
        # predicts a speed consistent with reality. Matches MATLAB vtgt=Vr_eff(ti).
        self._vtgt.copy_(torch.clamp(vtgt_t, max=vr_eff))
        # Heading slots use the TRAINING convention (matches MATLAB fs_mppi_lap.m:17
        # `x = [vt;0;0; -cos(th0); sin(th0); 0]` and the Simulink init yaw_sin=-1,
        # yaw_cos=0 at theta=0): yaw_sin_slot = -cos(yaw), yaw_cos_slot = sin(yaw).
        self._x0.copy_(torch.tensor(
            [v_plan, 0.0, self.yaw_rate, -math.cos(yaw), math.sin(yaw), 0.0],
            dtype=torch.float32, device=dev))
        self._p0.copy_(torch.tensor([x, y], dtype=torch.float32, device=dev))

        # ---- run rollout (CUDA graph replay on GPU, plain call on CPU) ----
        _t_roll = time.perf_counter()
        if self.use_graph:
            if self._graph is None:
                self._capture_graph()
            self._graph.replay()
            if self.dev.type == "cuda":
                torch.cuda.synchronize()
        else:
            self._rollout()
        roll_ms = (time.perf_counter() - _t_roll) * 1000.0
        cost = self._cost

        # ---- softmax → updated nominal (ADAPTIVE temperature) ----
        # Fixed LAMBDA=1.0 made the softmax degenerate: when the cost spread is tiny
        # (e.g. 0.01 at low speed), exp(-0.01/1.0)~1 for every sample -> uniform
        # weights -> nEff=K -> MPPI is just averaging noise (does nothing).
        # Scale the temperature to the ACTUAL per-cycle cost spread so the softmax
        # always discriminates good from bad regardless of absolute cost magnitude.
        c_lo = cost.min()
        c_spread = float(cost.max() - c_lo)
        lam = max(self.LAMBDA_FRAC * c_spread, self.LAMBDA_MIN)   # adaptive temperature
        wts     = torch.exp(-(cost - c_lo) / lam)
        wts     = wts / wts.sum()
        nom_new = (wts.unsqueeze(1) * A).sum(dim=0)
        self.nom = nom_new.unsqueeze(0)

        # ---- DEBUG: cost + sample diversity diagnostics ----
        cost_min  = float(cost.min()); cost_max = float(cost.max())
        # effective sample size: ~K if all equal-weighted (degenerate), ~1 if one dominates
        n_eff     = float(1.0 / (wts * wts).sum())
        # steer sample diversity at t=0 (are the 1024 samples actually different?)
        a0_std    = float(A[:, 0].std())

        # ---- first action: feedforward + MPPI correction (MATLAB fs_mppi_step.m:70) ----
        smax0 = float(np.clip(
            self.STEER_MAX - (self.STEER_MAX - 0.14) * (_speed - 13.0) / 5.0,
            0.14, self.STEER_MAX))
        ff0 = float(ff_t[0])
        steer_out = float(np.clip(self.K_FF * ff0 + float(nom_new[0]), -smax0, smax0))

        if os.environ.get("MPPI_FORCE_STEER0") == "1":
            steer_out = 0.0

        # warm-start shift
        self.nom = torch.cat([self.nom[:, 1:], self.nom[:, -1:]], dim=1)

        # ---- PI longitudinal (with anti-windup) ----
        err          = vr_eff - _speed
        raw          = self.KP * err + self.KI * self.I_spd
        lon0         = float(np.clip(raw, -1.0, 1.0))
        # Anti-windup: integrate ONLY when (a) the command isn't saturated AND (b) we're
        # close to the target (|err| < I_BAND). Integrating during big transients wound the
        # integral up into a large positive bias that then RESISTED braking when the corner
        # target dropped -> the car overshot target by up to ~1.9 m/s and entered corners hot.
        # Near the setpoint the integral just trims steady-state error; P handles transients.
        saturated    = (raw >= 1.0 and err > 0.0) or (raw <= -1.0 and err < 0.0)
        if (not saturated) and abs(err) < self.I_BAND:
            self.I_spd = float(np.clip(self.I_spd + err * dt_ros, -self.I_CLAMP, self.I_CLAMP))
        accel_out    = lon0 * self.ACCEL_MAX

        # ---- telemetry ----
        hd_ref      = path_heading(pd["rx_np"], pd["ry_np"], self.cur, loop=False)
        heading_err = _angle_diff(yaw, hd_ref)
        mean_traj   = self._viz_kinematic(x, y, yaw, steer_out, _speed)

        self._dbg_ctr += 1
        if self._dbg_ctr % 20 == 1:
            print(f"[MPPI] t={roll_ms:4.0f}ms odom={speed:5.2f} use={_speed:5.2f} "
                  f"vplan={v_plan:4.1f} vr={vr_eff:4.1f} vcap={v_path_cap:4.1f} vcrnr={v_corner:4.1f} look={lookahead_eff:3.1f} | "
                  f"steer={steer_out:+.3f} ff0={ff0:+.3f} accel={accel_out:+.2f} lon0={lon0:+.2f} | "
                  f"cur={self.cur}/{pd['N']} ahead={path_ahead:4.1f}m CTE={cte_val:+.2f} "
                  f"hd_err={float(heading_err):+.2f} | "
                  f"cost[{cost_min:.2f},{cost_max:.2f}] nEff={n_eff:.0f}/{self.K} "
                  f"lam={lam:.3f} a0std={a0_std:.3f} I={self.I_spd:.2f}",
                  flush=True)

        # ---- optional CSV trace (set MPPI_CSV=/path to enable) ----
        if self._csv is not None:
            self._csv.write(
                f"{time.time():.3f},{x:.3f},{y:.3f},{yaw:.4f},{speed:.3f},{_speed:.3f},"
                f"{v_plan:.3f},{vr_eff:.3f},{steer_out:.4f},{accel_out:.3f},{lon0:.3f},"
                f"{cte_val:.3f},{float(heading_err):.4f},{cost_min:.3f},{cost_max:.3f},"
                f"{n_eff:.1f},{a0_std:.4f},{pace_m:.4f},{roll_ms:.1f},{self.I_spd:.3f},"
                f"{self.yaw_rate:.4f},{self.cur},{pd['N']}\n")
            self._csv.flush()

        return steer_out, accel_out, {
            "target_speed": vr_eff,
            "cte":          cte_val,
            "heading_err":  float(heading_err),
            "mean_traj":    mean_traj,
            "path_bad":     bool(path_is_bad),   # one-sided-cone heading flip -> recovery
            "n_eff":        n_eff,               # MPPI sample diversity (K=degenerate, 1=peaked)
            "actual_speed": _speed,              # measured speed (for the speed-tracking plot)
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
