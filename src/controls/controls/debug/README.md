# Controls Debug & Diagnostic Tools

This folder contains standalone scripts for testing, benchmarking, and diagnosing the MPPI controller
and the car's dynamics. None of these are imported by the live stack — they are run directly when you
need to investigate something.

---

## Simulation / ROS diagnostics

### `gt_vs_slam.py`
**What:** Prints ground-truth vs SLAM odometry side-by-side at 2 Hz while the sim is running.  
**When to use:** Diagnosing whether SLAM pose is frozen / lagging / diverging from reality.  
**How to run:**
```bash
python3 gt_vs_slam.py         # run while sim + stack are up (controller optional)
```
**Output:** Continuously prints `GT: (x,y) heading Vx Vy` vs `SLAM: ...` so you can spot drift.

---

### `diag_listen.py`
**What:** Subscribes to `/slam/odom_raw` and `/path_points` with the same QoS the controller uses
(BEST_EFFORT, spinning node) and reports whether they actually arrive + their values.  
**When to use:** When the controller claims "no path" but you think PP is publishing; bypasses the
`ros2 topic echo` discovery quirk by living inside a real ROS node.  
**How to run:**
```bash
python3 diag_listen.py        # while sim + stack are up
```

---

### `drive_fwd.py`
**What:** Simplest possible Ackermann command publisher — drives the car straight forward at a fixed
acceleration, no path, no MPPI.  
**When to use:** Sanity-checking that the sim responds to `/cmd`, or collecting a clean straight-line
run for calibration.  
**How to run:**
```bash
python3 drive_fwd.py          # defaults: accel 2.0, steer 0
python3 drive_fwd.py 3.0 0.0  # accel 3.0, steer 0.0
```
Publishes `/cmd` at 50 Hz until you Ctrl+C.

---

## MPPI internals

### `plot_mppi_trace.py`
**What:** Visualizes an MPPI CSV trace (saved when you run the controller with `MPPI_CSV=/path`).
Plots speed, CTE, steering, costs, etc. over time.  
**When to use:** After a run, to see what the controller actually did.  
**How to run:**
```bash
# (after running: MPPI_CSV=/tmp/mppi_trace.csv ros2 run controls control_node)
python3 plot_mppi_trace.py /tmp/mppi_trace.csv
```
**Output:** Saves `<csv>.png` (and shows it if a display is available).

---

### `bench_cudagraph.py`
**What:** Benchmarks three rollout strategies (plain Python loop, batched kernels, CUDA graph) for the
K=1024, T=200 Neural ODE rollout. Proves CUDA graph fixes the WSL2 per-kernel-launch overhead that
made the rollout ~3.9 s without it.  
**When to use:** Verifying latency optimizations or porting to a new machine.  
**How to run:**
```bash
python3 bench_cudagraph.py
```
**Output:** Timing for each strategy (plain / batched / graph) and a bit-identical check.

---

### `test_graph_ctrl.py`
**What:** Instantiates `MPPIController`, feeds it a dummy curving path, and verifies the CUDA-graph
rollout produces non-zero, sane steering commands.  
**When to use:** Smoke-testing the controller after a big change to `mppi_controller.py`.  
**How to run:**
```bash
python3 test_graph_ctrl.py
```
**Output:** "Graph controller test PASSED" if steer and cost are reasonable.

---

## Model / vehicle dynamics checks

### `fit_tire.py`
**What:** Fits EUFS tire parameters (B, C, D, E) to reproduce the IITRMS dataset's linear tire model
(`Cf = Cr = 61012 N/rad, μ = 1.7`). Documents the parameter values that went into `configDry.yaml`.  
**When to use:** If you re-tune the simulator tire or want to understand the EUFS vs IITRMS mapping.  
**How to run:**
```bash
python3 fit_tire.py
```
**Output:** Prints the fitted `(B, C, D, E)` and error vs the target linear model.

---

## How to add a new diagnostic

1. Write a standalone `.py` script (no imports from the live controller if possible; if needed,
   `sys.path.insert` like the existing ones do).
2. Add a docstring at the top explaining **what, when, how**.
3. Drop it in this folder.
4. Update this README with a new section.

The goal: anyone on the team can run these without wading through code or guessing command-line args.
