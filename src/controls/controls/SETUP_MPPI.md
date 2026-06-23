# RAMS-e-25 MPPI — Setup & Status (branch: controls_rl_Soumil_mppi)

Neural ODE (exp09) MPPI controller for the EUFS Gazebo stack. Runs entirely in WSL2.

## What works now
- **exp09 grey-box Neural ODE** as the MPPI rollout model (best_model_600.pt).
- **GPU, K=1024, T=100, RK4** — full sample count, on GPU, in WSL2.
- **38 ms per planning step** (was 3.9 s). Fixed with CUDA graphs — see below.
- **EUFS car = IITRMS car** — the Gazebo plant now uses our mass/inertia/wheelbase/
  tire/aero, so the model's predictions match the plant.
- Steer direction verified correct (off-left→steer right, off-right→steer left).

## The three problems and how each was solved
1. **Compute speed (3.9 s/call in WSL2).** WSL2 CUDA passthrough adds ~0.6 ms per kernel
   launch; the 100-step rollout launched thousands of kernels.
   FIX: (a) `@torch.no_grad()` on the rollout, (b) **CUDA graph** capture/replay — the whole
   rollout is captured once and replayed as ONE GPU submission. Bit-identical results, 38 ms.
   → No native-Windows ROS2 build needed. Nothing installed on C:.
2. **Car mismatch.** exp09 trained on IITRMS car; EUFS shipped a generic car.
   FIX: edited `eufs_ws/.../robots/eufs/configDry.yaml` to IITRMS params. Tire B,C,D,E fitted
   to reproduce the DATASET's linear tire (Cf=Cr=61012 N/rad, μ=1.7) — NOT the raw .tir,
   because the dataset only ever used the linearized tire. Fit script: `fit_tire.py`.
3. **SLAM/PP realities.** odom twist=0, sparse 5-11pt path.
   FIX: windowed position-regression speed estimate; arc-length path interpolation;
   kinematic yaw injection (model's dyawRate is a residual, not the yaw driver).

## How to run (4 WSL terminals, each: `wsl -d Ubuntu-2004-eufs`)
```bash
# T1 — simulator
~/run_eufsim.sh
#   In launcher: Track=small_track (or skidpad), Vehicle=DynamicBicycle,
#   Preset=DryTrack, robot=eufs, Command Mode=acceleration -> Launch

# T2 — stack (fusion + SLAM + path planning)
ros2 launch bringup stack.launch.py

# T3 — MPPI controller (prints GPU + "CUDA graph captured")
ros2 run controls control_node

# T4 — enable driving
ros2 service call /ros_can/reset std_srvs/srv/Trigger
ros2 service call /ros_can/set_mission eufs_msgs/srv/SetCanState "{ami_state: 21}"
```

## Files changed (all on branch controls_rl_Soumil_mppi)
| File | What |
|------|------|
| `src/controls/controls/mppi_controller.py` | CUDA-graph MPPI, exp09 NN, GPU K=1024 T=100 |
| `src/controls/controls/control_node.py` | calls mppi.compute(); ROS plumbing unchanged |
| `src/controls/controls/fs_model/best_model_600.pt` | exp09 checkpoint (resaved for numpy 1.23) |
| `eufs_ws/.../robots/eufs/configDry.yaml` | IITRMS car params (backup: configDry.yaml.orig) |

## Diagnostics / helper scripts (can delete later)
- `bench_cudagraph.py` — proves the CUDA-graph speedup (BENCH_K / BENCH_T env vars)
- `test_graph_ctrl.py` — verifies 38 ms + correct steer direction headless
- `fit_tire.py` — derives the tire B,C,D,E from Cf/μ

## To revert the EUFS car to stock
```bash
cp ~/eufs_ws/src/eufs_sim/eufs_racecar/robots/eufs/configDry.yaml.orig \
   ~/eufs_ws/src/eufs_sim/eufs_racecar/robots/eufs/configDry.yaml
cd ~/eufs_ws && colcon build --packages-select eufs_racecar
```

## Later (2-PC testing)
The native-Windows + DDS/firewall path is NOT needed for single-PC. When you go 2-PC
(one sim, one stack), set matching ROS_DOMAIN_ID + RMW_IMPLEMENTATION=rmw_cyclonedds_cpp on
both, keep `.wslconfig` networkingMode=mirrored, and add Windows Firewall inbound UDP rules
(7400-7600). That work was scoped but shelved since CUDA graphs removed the need.
