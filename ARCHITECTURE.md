# RAMS-e-25 — Autonomous Stack Architecture 

This document explains our **whole self-driving stack** end to end: every node, what it does, what
data it consumes and produces, and — most importantly — **why each piece needs the one before it**.


The car runs inside the **EUFS Gazebo simulator** (a Formula Student Driverless simulator). The
pipeline is **track-agnostic** — it drives any course marked out by cones. The same stack is expected to run on every event
track (acceleration, skidpad, autocross, trackdrive). One historical note: the path-planner node is
*named* `pp_node_skidpad` for legacy reasons, but it is a general cone-track planner, not
skidpad-specific.

> If you just want to build and run, see [`README.md`](README.md). This file is the "how and why".

---

## 1. The big picture

The stack is a classic autonomy pipeline. Data flows top to bottom; each stage refines the last and
the final command loops back into the simulator.

```
            ┌──────────────────────────────────────────────────────────────┐
            │                    EUFS Gazebo Simulator                      │
            │   camera · LiDAR · /ground_truth/odom · /ground_truth/cones   │
            └──────────────────────────────────────────────────────────────┘
                 │ images       │ points      │ true pose     │ true cones
                 ▼              ▼              │               ▼
            ┌──────────────────────────────┐  │      ┌───────────────────────┐
            │          PERCEPTION          │  │      │   (cones can come from│
            │  camera_node · lidar_node    │  │      │    ground truth while │
            │  fusion_node                 │◄─┼──────┤    perception matures)│
            └──────────────────────────────┘  │      └───────────────────────┘
                 │ /perception/cones_fused     │
                 ▼                             ▼
            ┌──────────────────────────────┐  ┌───────────────────────────────┐
            │             SLAM             │  │        prediction_node         │
            │  slam_node:                  │◄─┤  /ground_truth/odom            │
            │  stable cone MAP + pose      │  │       → /slam/odom_raw (live)  │
            └──────────────────────────────┘  └───────────────────────────────┘
                 │ /slam/map_cones             │ /slam/odom_raw (live pose)
                 ▼                             │
            ┌──────────────────────────────┐  │
            │         PATH PLANNING        │◄─┤
            │  pp_node → /path_points      │  │
            └──────────────────────────────┘  │
                 │ /path_points                │  (controls also reads
                 ▼                             │   /ground_truth/odom for speed
            ┌──────────────────────────────┐  │   and /slam/map_cones for recovery)
            │           CONTROLS           │◄─┘
            │  control_node + MPPI → /cmd  │
            └──────────────────────────────┘
                 │ /cmd  (AckermannDriveStamped)
                 ▼
            the simulator applies /cmd → the car moves → new sensor data → loop
```

Each ROS 2 package is one stage:

| Package | Job | Produces |
|---|---|---|
| `perception` | Turn raw sensors into a list of cones | `/perception/cones_fused` |
| `slam` | Turn noisy cone sightings + odometry into a stable **map** and **car pose** | `/slam/map_cones`, `/slam/odom`, `/slam/odom_raw` |
| `pathplanning` | Turn the cone map into a **centreline path** | `/path_points` |
| `controls` | Turn the path into **steering + throttle** | `/cmd` |

---

## 2. The simulator — what EUFS gives us

EUFS publishes everything a real car's sensors and a "god's-eye" reference would provide:

- **Camera** — `/zed/left/image_rect_color`, `/zed/right/image_rect_color` (stereo pair) + `camera_info`.
- **LiDAR** — `/velodyne_points`, a raw 3-D point cloud.
- **Ground-truth odometry** — `/ground_truth/odom` (`nav_msgs/Odometry`): the car's *exact* pose and
  velocity. A real car never has this; we use it as a perfect reference and (today) as a live-pose
  source while SLAM matures.
- **Ground-truth cones** — `/ground_truth/cones` (`eufs_msgs/ConeArrayWithCovariance`): exact cone
  positions and colours.

**Cone colour convention (matters later):** in Formula Student, **blue = left edge**, **yellow =
right edge** of the track (orange = start/finish). The car drives so blue is on its left, yellow on
its right.

---

## 3. Perception — from pixels and points to cones

Answers one question: *"where are the cones around me, and what colour are they?"* Entry points
(`perception/setup.py`): `camera_node`, `lidar_node`, `fusion_node`.

```
  /zed left+right + camera_info ─► camera_node ─► /perception/cones_stereo ┐
  /velodyne_points ─────────────► lidar_node  ─► /perception/cones ────────┼─► fusion_node ─► /perception/cones_fused
  /ground_truth/cones ───────────────────────────────────────────────────-┘   (adapter: GT cones → fused format)
```

### 3.1 `camera_node.py` — cones from the stereo camera
- **In:** left/right rectified images + both `camera_info` (time-synchronised).
- **Out:** `/perception/cones_stereo` (`sensor_msgs/PointCloud2`, one point per cone + colour class).
- **How:** a **YOLO** detector finds and colour-classifies cone boxes in the left image; a **stereo
  disparity** between left/right gives depth; each box is **back-projected** to a 3-D position using
  the camera intrinsics.
- **Why:** the camera is the only sensor that knows **colour** — and colour tells us which side of
  the track a cone marks.

### 3.2 `lidar_node.py` — cones from the LiDAR
- **In:** `/velodyne_points`.
- **Out:** `/perception/cones` (`sensor_msgs/PointCloud2`, one point per cone).
- **How:** crop to a box in front of the car → remove the ground with **RANSAC** plane fitting →
  **DBSCAN** cluster the leftover points; each cluster centroid is a cone.
- **Why:** LiDAR gives far more accurate **distance** than a camera. Camera = colour, LiDAR =
  position; neither alone is enough.

### 3.3 `fusion_node.py` — one clean cone list
- **Out:** `/perception/cones_fused` (`sensor_msgs/PointCloud2`) — the canonical cone list everyone
  downstream uses.
- **Two variants:**
  - **Real fusion** (`perception/backup/fusion_backup.py`): for each LiDAR cone, borrow the colour of
    the nearest camera cone — trust LiDAR for *where*, camera for *what colour*.
  - **Active adapter** (`fusion_node.py`, `GroundTruthConeAdapter`): republish the simulator's perfect
    `/ground_truth/cones` in the `/perception/cones_fused` format.
- **Why this design:** downstream only cares about the *format* of `/perception/cones_fused`. Making
  fusion an adapter lets us build and validate SLAM → planning → control on **perfect cones first**,
  then swap in real camera+LiDAR fusion later without touching anything downstream.

**Hand-off:** perception's output is a *flickering, body-frame snapshot* with no memory. SLAM fixes
that.

---

## 4. SLAM — a stable map and a corrected pose

To plan a path we need a **stable map in world coordinates** and a reliable **car pose** in it.
Entry points (`slam/setup.py`): `pred_node`, `slam_node`.

```
  /ground_truth/odom ─► prediction_node ─► /slam/odom_raw ──┐
                                                            ├─► slam_node ─► /slam/odom        (corrected pose, cone-gated ~5 Hz)
  /perception/cones_fused ──────────────────────────────────┘              └► /slam/map_cones  (stable map, map frame)
```

### 4.1 `prediction_node.py` — the live pose relay
- **In:** `/ground_truth/odom`. **Out:** `/slam/odom_raw` (`nav_msgs/Odometry`, `map` frame).
- Despite the name, it does **no prediction** — it copies the ground-truth pose straight through at
  full rate (~190 Hz). Think of it as a clean, always-on "live position" tap.
- **Why:** the main SLAM node only updates its pose **when it sees cones** (slow, bursty). The planner
  and controller need a smooth, continuous position; `/slam/odom_raw` is that signal.

### 4.2 `slam_node.py` — the particle-filter SLAM
- **In:** `/slam/odom_raw` (motion) + `/perception/cones_fused` (measurements).
- **Out:** `/slam/map_cones` (`eufs_msgs/ConeArrayWithCovariance`, `map` frame — the de-flickered map
  the planner uses) and `/slam/odom` (the SLAM-corrected pose).
- **How:** a ~80-particle **particle filter** — each particle is a guess of the car pose carrying its
  own landmark map. Odometry moves all particles (+ noise); each cone frame scores particles on how
  well cones match their map (Mahalanobis gating + Hungarian matching from `slam_utils/association.py`),
  refines landmarks with an EKF, and resamples. A rolling local map and **snap-back loop closure**
  (RANSAC re-alignment) keep the map crisp. Core engine: `slam_utils/pf_slam.py`.
- **Why:** planning needs the **whole, stable picture** of the cones, not one flickering frame.

`slam_final/slam_skidpad_full.py` is a simpler baseline (no loop closure); the launch uses
`slam_node.py`.

---

## 5. The odometry story — `/slam/odom` vs `/slam/odom_raw`

This caused real, hard-to-find bugs, and it explains a key choice in both planning and control.

**What we originally did:** both the planner and the controller took the car pose from **`/slam/odom`**
(the SLAM-corrected pose) — it seemed the "proper" choice (map frame, aligned with the cone map).

**Why it broke:** in `slam_node.py`, **`/slam/odom` is published only inside the cone callback** — it
updates *only when a cone measurement arrives*. So it runs at ~5 Hz, is **frozen at the spawn point**
at start-up (before SLAM sees cones), and **freezes** whenever cones briefly drop out. Both planner
and controller were reading a pose that **lagged or froze** while the real car kept moving. We
measured it: `/slam/odom` ≈ 5 Hz and bursty vs `/slam/odom_raw` ≈ 190 Hz and rock-steady.

**The fix:** switch the parts that need a **live, continuous position** to **`/slam/odom_raw`**, while
the **cone map still comes from SLAM** (`/slam/map_cones`):
- **Controls** reads **position/heading from `/slam/odom_raw`** and **speed from `/ground_truth/odom`**
  (the SLAM pose carries no usable velocity).
- **Path planning** reads its car pose from **`/slam/odom_raw`** too (§6).

This is a pragmatic decision: it leans on the live, ground-truth-derived pose to get a timely signal.
Real SLAM has an inherent timing limitation today (it can only correct when it sees cones); the
long-term answer is on the SLAM side (a fast live updater paired with a slower correction filter). For
now, the live relay lets the rest of the stack run smoothly so planning and control can be finished
and validated.

---

## 6. Path planning — a centreline through the cones

Turns the stable cone map + live pose into **a path to drive**. Active node: `pp_node_skidpad.py`
(general planner despite the name; entry point `pp_node_skidpad`, plus a `pp_debug` visualiser).

```
  /slam/odom_raw (car pose) ──┐
                              ├─► pp_node_skidpad ─► /path_points  (nav_msgs/Path, map frame)
  /slam/map_cones (cones) ────┘     FOV → Delaunay → colour-filtered edges → midpoints → greedy path
```

### 6.1 How the path is built (every tick, ~20 Hz)
1. **Sector FOV:** keep only cones in a wedge in front of the car — tip ~5 m behind, reaching ~30 m
   out, ±45° around heading.
2. **Build a graph:** connect nearby cones with **Delaunay triangulation** (k-NN fallback) → candidate
   edges.
3. **Colour-filter edges:** drop blue–blue and yellow–yellow edges (they run *along* a wall). A
   **blue–yellow edge spans the track**, and its **midpoint is a centreline point**.
4. **Candidates:** midpoints of kept edges (+ special handling for big-orange start/finish cones).
5. **Thin out:** enforce minimum spacing and minimum distance from the car.
6. **Greedy path:** from the car, hop to the nearest remaining candidate (bounded hop) → an ordered
   centreline.
7. **Publish** `/path_points`, with the **car's own position prepended** as the first pose.

### 6.2 Why the planner also needed `/slam/odom_raw`
The whole FOV is built **around the car's current pose** (`_compute_fov_points(car_x, car_y,
car_yaw)`). A stale pose keeps the wedge pointed where the car *used to be*, so the selected cones and
the path are wrong — at start-up the FOV stayed glued to the spawn point, and in corners a lagging
heading pointed the wedge slightly off and missed the inside wall. Pointing the planner at
`/slam/odom_raw` keeps the FOV centred on where the car *actually is*, every tick.

**Hand-off:** `/path_points` is a short, rolling **window** of the track ahead (the ~6–15 cones
currently in view), refreshed every tick — not the whole lap. The controller follows this moving
window while it is continuously redrawn.

---

## 7. Controls — driving the path with MPPI

The controller's job: given the path window and the car's live state, decide **how much to steer and
how much to accelerate**, ~10 times a second. Entry point (`controls/setup.py`): `control_node`.

```
  /slam/odom_raw (pose) ───┐
  /ground_truth/odom (spd) ┤
  /path_points (path) ─────┼─► control_node ─► MPPIController.compute() ─► /cmd  (steer + accel)
  /slam/map_cones (recov.) ┘            └── rolls out the exp10 car model 1024× on the GPU
```

We use **MPPI** (Model Predictive Path Integral control). Each cycle it imagines **K = 1024** steering
plans, simulates each one **T = 200** steps into the future with a **learned car model**, scores them
on how well they hug the path (based on a cost function), and blends the good ones into one steering command. A separate
controller handles throttle.

### 7.1 The car model (the "brain" MPPI rolls out)
MPPI can only imagine futures with a model of *how the car responds to controls*. The model used here is based on  a **Grey Box Neural ODE**.

- **What it is:** a **grey-box Neural ODE** — we hard-code the physics we know exactly (how heading
  changes as the car yaws) and let a small NN learn only the hard parts (how velocity, yaw-rate,
  and slip evolve). NN Architecture : `8 → 128 → 128 → 128 → 4`, SiLU activations.
- **Trained on:** a vehicle model with our car's parameters driven through Simulink across a practical
  **0.1–12 m/s** range, producing (state, control, next-state) transitions. 
- **Ported into EUFS as** `controls/controls/fs_model/best_model_600.pt`. It holds the network weights
  **plus** the normalisation constants (`x_mean`, `x_std`, `u_mean`, `u_std`), the valid state range
  (`x_min`, `x_max` — Vx valid 0–13.9 m/s), and the integration setting (`num_rk4_steps = 1`). The
  controller loads it directly — no retraining, no conversion. The exact same grey-box equations exist
  on the training side and here, so the ported model behaves identically to what was validated offline.

### 7.2 State, control, and the heading convention
The model's **state** is a 6-D vector and the **control** is a 2-D vector:

```
x = [ Vx, Vy, yawRate, s1, s2, slip ]          u = [ steer, longitudinal ]
```

`Vx, Vy` are car-frame velocities, `yawRate` is the turn rate, `slip` is the side-slip state. The two
heading slots `s1, s2` use a **rotated convention** carried over from the training data:

```
s1 = -cos(θ)      s2 = sin(θ)       (θ = car heading in the world)
```

so the heading is recovered as `θ = atan2(s2, -s1)`. (This exact convention is what the model was
trained on; getting it wrong silently rotates everything 90°, which we hit and fixed early on.)

### 7.3 The grey-box model equations (one rollout step)
Given state `x` and control `u`, the network computes a state derivative `dx/dt`:

```
1.  Normalise:     x̂ = (x − x_mean) / x_std,     û = (u − u_mean) / u_std
2.  Network:       g = MLP([x̂ ; û])              (4 outputs)
3.  Derivatives:
        dVx      = g₀ · x_std[Vx]
        dVy      = g₁ · x_std[Vy]
        dyawRate = g₂ · x_std[yawRate]
        ds1      = s2 · yawRate          ← exact heading kinematics (not learned)
        ds2      = −s1 · yawRate         ← exact heading kinematics (not learned)
        dslip    = g₃ · x_std[slip]
```

Integration is **RK4** with `num_rk4_steps = 1`, step `h = DT_EFF = 0.005 s`:

```
k1 = f(x),  k2 = f(x + ½h·k1),  k3 = f(x + ½h·k2),  k4 = f(x + h·k3)
x ← x + (h/6)(k1 + 2k2 + 2k3 + k4)
```

then the heading pair `(s1, s2)` is renormalised to the unit circle and the state is clamped to
`[x_min, x_max]`. **Why DT_EFF = 0.005:** the training data transitions are 0.005 s apart and
`num_rk4_steps = 1`, so the rollout must step at 0.005 s to integrate at the granularity the model
learned. A coarser step makes the imagined trajectories inaccurate and the car turns in late.

The whole T-step rollout for all K samples is captured once as a **CUDA graph** and replayed each
cycle (~75 ms for 1024 × 200 steps on the laptop GPU), which is what makes real-time MPPI possible
here.

### 7.4 The MPPI algorithm (one `compute()` cycle)
**Build the reference.** The raw ~6 path points are smoothed with a **B-spline** and resampled into
**40** evenly-spaced points, each with a clean curvature `κ`. A reference is then *marched* along the
path: starting `LOOKAHEAD_M = 5 m` ahead of the car and advancing by `pace = max(vr, 0.5)·DT_EFF` per
horizon step (so the reference moves at the speed we intend to drive). At each of the T steps we have a
reference point `(refE, refN)`, its left-normal `(nE, nN)`, and a feed-forward steer `ff`.

**Sample K steering sequences.** Around the previous best plan `nom` (length T), add Gaussian noise:

```
A = clamp( nom + ε ,  −STEER_MAX, +STEER_MAX ),     ε ~ N(0, SIG_STEER²),
SIG_STEER = 0.08 rad,   STEER_MAX = 0.349 rad (≈20°),   A is K×T = 1024×200
```

**Roll out + score.** For each sample, step the model T times. At horizon step `t`, with body speed
`vx`:

```
speed-dependent steer cap:   smax = clamp( 0.349 − (0.349−0.14)·(vx−13)/5 ,  0.14, 0.349 )
applied steer:               steer = clamp( K_FF·ff[t] + A[:,t] ,  −smax, +smax )   (K_FF = 0.85)
rollout throttle (P-only):   lon   = clamp( KP_ROLL·(vtgt[t] − vx) , −1, +1 )       (KP_ROLL = 0.40)
```

step the model, then integrate the world position with the trapezoid rule
`p ← p + ½(V_before + V_after)·DT_EFF`, and accumulate cost (next section). Finally `cost ← cost / T`.

### 7.5 The cost function
Two terms, summed over the horizon. **(C1) Cross-track**, a Huber penalty on the signed perpendicular
distance from the reference:

```
perp = (pₓ − refE)·nE + (p_y − refN)·nN
ctc  = min(|perp|, CT_SAT)²  +  2·CT_SAT·max(|perp| − CT_SAT, 0)        (CT_SAT = 5.0 m)
```

(quadratic up to 5 m, then linear — so a far-off sample is still pulled back but doesn't blow the cost
up). **(C2) Heading**, a *pure-pursuit* term that points the car at the upcoming reference point:

```
carθ = atan2(s2, −s1)
bear = atan2(refN − p_y, refE − pₓ)          (bearing from car to the preview point)
dh   = wrap(carθ − bear)
```

gated off when the preview point is within 1 m (its bearing is meaningless there). The per-step cost:

```
cost += W_CT·ctc  +  W_HEAD·gate·dh²          (W_CT = 1.0,  W_HEAD = 0.3)
```

### 7.6 Blending — softmax with adaptive temperature
Turn the 1024 costs into weights and blend the steering sequences:

```
λ   = max( LAMBDA_FRAC · (cost_max − cost_min),  LAMBDA_MIN )      (LAMBDA_FRAC = 0.15, LAMBDA_MIN = 1e-3)
w_k = exp( −(cost_k − cost_min) / λ ),   then normalise  Σw = 1
nom = Σ_k  w_k · A_k                       (the new best steering sequence)
```

**Why adaptive λ:** with a *fixed* temperature, when all samples cost about the same (e.g. on a
straight, spread ≈ 0.01) every weight ≈ 1 → the softmax is uniform → MPPI just averages noise and
does nothing. Scaling λ to the *actual* cost spread each cycle keeps the softmax discriminating
regardless of the absolute cost size. (This is the same reasoning as scaled-dot-product attention:
normalise by the spread so the softmax neither saturates nor flattens.)

### 7.7 The applied command, the feed-forward, and how MPPI relates to it
The first action actually sent to the car:

```
smax0     = clamp( 0.349 − (0.349−0.14)·(v−13)/5 , 0.14, 0.349 )
steer_out = clamp( K_FF · ff[0]  +  nom[0] ,  −smax0, +smax0 )
```

So **`steer = K_FF·ff + MPPI`** — a **feed-forward + feedback** pair :

- **Feed-forward `K_FF·ff`** is the **open-loop curvature** term. `ff = atan(L·κ)` is the Ackermann
  steering for the path's curvature `κ` (`L = L_WB_KIN = 1.535 m`, `κ` from the spline, clipped to
  ±0.5). When the path bends, this supplies the *anticipated* turn without waiting for error to build.
- **MPPI `nom[0]`** is the **closed-loop correction** — it reacts to where the car actually is versus
  the path and steers to rejoin.

**Which one dominates? :** 
Across 37 commands of a completed
lap, the feed-forward was the larger term in **22/37** and MPPI in **15/37**; mean magnitudes were
`|K·ff| = 0.095` vs `|MPPI| = 0.078`. So they are **comparable partners, not one-dominates-the-other**:
the feed-forward carries most of the steering on smooth, curving sections (where `κ` is large), while
**MPPI dominates the corrections** — whenever the path is locally straight (`ff ≈ 0`) but the car is
off-line, MPPI is the term that brings it back. MPPI is genuinely doing work (it is the feedback that
holds the line); it is *not* a small correction bolted onto the feed-forward, nor is the reverse true.

### 7.8 Longitudinal control (throttle)
Steering is MPPI; **speed is rule-based** . A target speed `vr` is
formed, then a PI controller turns the speed error into throttle.

**Target speed `vr`** is the minimum of several limits:

```
corner profile:   vt[i] = min( V_MAX, sqrt(A_LAT_MAX / |κ[i]|) ),   then a backward braking pass
                  vt[i] = min( vt[i], sqrt(vt[i+1]² + 2·BRAKE_A·ds) )   so the car slows BEFORE a corner
cross-track scale: scale = clip( 1 − K_CT·min(|cte|/CT0, 1), 0, 1 )      (slow when off-line)
visibility cap:   v_path = ahead / (HORIZON_CAP_FRAC · T · DT_EFF)        (never outrun the visible path)
braking cap:      v_brake = sqrt( 2 · BRAKE_A · path_ahead )              (stop by the path's end)

vr_raw = min( vt[cur]·scale,  v_path,  v_brake )
```

with **full stops** if very little path remains (`path_ahead ≤ STOP_PATH_M = 1.5 m`) or the path
points the wrong way (`|heading offset| > HD_BAD_RAD = 1.4 rad` — the one-sided-cone case, §7.11).
`vr_raw` is then rate-limited (slow to slow down, quick to speed up) into `vr`.

**The PI throttle:**

```
err   = vr − speed
I    += err·dt        (clamped to ±I_CLAMP, with anti-windup)
lon0  = clamp( KP·err + KI·I , −1, +1 )
accel = lon0 · ACCEL_MAX
```

`KP = 0.35, KI = 0.10, I_CLAMP = 5.0, ACCEL_MAX = 3.0 m/s²`.

Note: at the low speeds we run (`V_MAX = 2.5`), the corner profile rarely binds — a 2.5 m/s car isn't
grip-limited even on tight corners — so the profile is "there and correct for higher speeds" rather
than active right now.

### 7.9 Every constant, in one place

| Symbol | Value | Meaning |
|---|---|---|
| `K` | 1024 | MPPI samples per cycle |
| `T` | 200 | horizon steps |
| `DT_EFF` | 0.005 s | rollout step (= training dt; horizon = 1.0 s) |
| `num_rk4_steps` | 1 | RK4 sub-steps inside the model |
| `SIG_STEER` | 0.08 rad | steering sample noise |
| `STEER_MAX` | 0.349 rad | steering limit (≈20°, model range) |
| `W_CT` | 1.0 | cross-track cost weight |
| `W_HEAD` | 0.3 | pure-pursuit heading cost weight |
| `CT_SAT` | 5.0 m | Huber knee |
| `LOOKAHEAD_M` | 5.0 m | reference preview offset |
| `K_FF` | 0.85 | curvature feed-forward gain |
| `LAMBDA_FRAC` | 0.15 | adaptive softmax temperature fraction |
| `LAMBDA_MIN` | 1e-3 | temperature floor |
| `KP, KI` | 0.35, 0.10 | longitudinal PI gains |
| `I_CLAMP` | 5.0 | PI integral clamp |
| `KP_ROLL` | 0.40 | in-rollout throttle P-gain |
| `V_MAX` | 2.5 m/s | speed cap (kept low for now) |
| `A_LAT_MAX` | 2.5 m/s² | corner-profile lateral budget |
| `BRAKE_A` | 2.5 m/s² | braking decel for speed caps |
| `CT0, K_CT` | 4.0 m, 0.7 | cross-track speed-reduction |
| `HORIZON_CAP_FRAC` | 0.5 | fraction of horizon that must fit visible path |
| `STOP_PATH_M` | 1.5 m | stop if less path remains |
| `HD_BAD_RAD` | 1.4 rad | path-heading-flip threshold (recovery trigger) |
| `V_PLAN_MAX` | 13.0 m/s | model's valid speed ceiling |
| `L_WB_KIN` | 1.535 m | wheelbase used in `ff = atan(L·κ)` |
| `ACCEL_MAX` | 3.0 m/s² | simulator accel limit |
| `SPLINE_NPTS` | 40 | resampled path points |
| `RECOVER_STEER` | 0.20 rad | one-sided-cone recovery steer |
| `RECOVER_SPEED` | 1.0 m/s | one-sided-cone recovery creep speed |

### 7.10 The supporting files
- **`ros_connect.py`** — all ROS I/O, isolated from the control maths. Subscribes `/slam/odom_raw`
  (pose), `/ground_truth/odom` (speed), `/path_points` (path), `/slam/map_cones` (cones, for §7.11).
  Publishes `/cmd`. Maintains a small **rolling buffer** of path points so one bad frame can't wipe the
  path, and exposes `get_state()`, `get_path()`, `get_local_cone_counts()`.
- **`control_node.py`** — the entry point and main loop: fetch state + path → `MPPIController.compute()`
  → clamp → publish `/cmd`; runs the recovery (§7.11) when the path is unusable; records a trajectory
  plot saved on exit (the main debugging artefact).
- **`control_utils.py`** — pure helpers: closest-point search, signed cross-track error, path heading,
  curvature, spline resampling, a kinematic bicycle model for visualisation, PID/velocity utilities.
  No ROS, no state.

### 7.11 The one-sided-cone recovery
There are spots where the car briefly sees only **one wall** of cones; the planner then can't build a
centreline and emits a path that points sideways. Following it would steer into the cones, so the
controller detects it (path heading vs car heading off by more than `HD_BAD_RAD`) and switches to a
**recovery** instead of following the bad path:

- only **blue** (left) cones nearby → steer gently **right** (`−RECOVER_STEER`) at `RECOVER_SPEED`,
- only **yellow** (right) cones nearby → steer gently **left** (`+RECOVER_STEER`),
- the moment the planner gives a usable path again → hand straight back to MPPI.

It counts colours from `/slam/map_cones` within ~12 m ahead. This is a safety fallback to carry the
car through brief one-wall sections; the fuller fix (synthesising a temporary centreline from the
single visible wall) belongs in the planner.

---

## 8. End-to-end — the life of one control decision

1. The **camera** sees a blue cone and labels it; the **LiDAR** measures it's 7 m away; **fusion**
   merges them (or hands over the true cones) → `/perception/cones_fused`.
2. **SLAM** adds it to the persistent map and confirms the car pose → `/slam/map_cones`, while
   **prediction_node** streams the live pose → `/slam/odom_raw`.
3. The **planner** pairs blue with yellow and draws a centreline → `/path_points`.
4. The **controller** imagines 1024 ways to follow that line over the next second with the exp10 model,
   blends the best, adds the curvature feed-forward → `/cmd`.
5. The **simulator** applies `/cmd`; the car moves; new data arrives; the loop repeats ~10×/s.

---

## 9. Topic reference

| Topic | Type | Produced by | Consumed by |
|---|---|---|---|
| `/zed/*` images, info | `sensor_msgs/Image`, `CameraInfo` | simulator | `camera_node` |
| `/velodyne_points` | `sensor_msgs/PointCloud2` | simulator | `lidar_node` |
| `/ground_truth/odom` | `nav_msgs/Odometry` | simulator | `prediction_node`, controls (speed) |
| `/ground_truth/cones` | `eufs_msgs/ConeArrayWithCovariance` | simulator | `fusion_node` (adapter) |
| `/perception/cones_stereo` | `sensor_msgs/PointCloud2` | `camera_node` | fusion (real) |
| `/perception/cones` | `sensor_msgs/PointCloud2` | `lidar_node` | fusion (real) |
| `/perception/cones_fused` | `sensor_msgs/PointCloud2` | `fusion_node` | `slam_node` |
| `/slam/odom_raw` | `nav_msgs/Odometry` | `prediction_node` | `slam_node`, planner, controls |
| `/slam/odom` | `nav_msgs/Odometry` | `slam_node` (cone-gated, ~5 Hz) | debug / legacy |
| `/slam/map_cones` | `eufs_msgs/ConeArrayWithCovariance` | `slam_node` | planner, controls (recovery) |
| `/path_points` | `nav_msgs/Path` | `pp_node_skidpad` | controls |
| `/cmd` | `ackermann_msgs/AckermannDriveStamped` | `control_node` | simulator |

---

## 10. Build & run

See [`README.md`](README.md) for the canonical commands. In short:

```bash
cd RAMS-e-25
colcon build --symlink-install
source install/setup.bash

# upstream stack (perception + slam + planning):
ros2 launch slam testslam.launch.py

# the controller:
MPPI_CSV=/tmp/mppi_trace.csv ros2 run controls control_node
```

Run the EUFS simulator (we test on the **small_track** option) with the driving mode set to
**manual/“go”** so the car accepts `/cmd`, and make sure the EUFS workspace is sourced (the
cone-recovery uses the `eufs_msgs` message type).

---

## 11. Known limitations & where we're heading

- **Tracking quality:** the car cuts slightly to the inside of corners (a pure-pursuit look-ahead
  effect) and slows itself when off-line. Curvature-/speed-adaptive look-ahead is the next step.
- **One-sided cones:** handled today by the recovery fallback; the fuller fix (synthesise a centreline
  from a single wall) belongs in the planner.
- **SLAM timing:** the stack currently leans on the live `/slam/odom_raw`. Removing that dependence is
  a SLAM-side roadmap item (a fast live updater plus a slower correction filter).
- **Lap end:** behaviour at the end of the known path (clean stop vs. loop closure) is still being
  defined.
- **Speed:** capped low (`V_MAX = 2.5`) while we harden tracking; the corner speed profile is already
  in place for when we raise it.

The current milestone: the car **completes a full lap** of the test track end-to-end through this
pipeline.
