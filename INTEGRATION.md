# Shared planner and controls baseline

This branch pairs the teammate's skidpad planner with Soumil's mirror-symmetric Exp13 Neural ODE and MPPI controller. Gazebo supplies odometry and cone observations. The Exp13 checkpoint and per-run recorder are included.

## Prerequisites

Use Ubuntu 20.04 or WSL with ROS 2 Galactic, Gazebo Classic, a built EUFS workspace and colcon. In the Python environment used by ROS, install NumPy, SciPy, Matplotlib and a compatible PyTorch build. An NVIDIA GPU and CUDA-enabled PyTorch are needed for comparable controller timing. CPU fallback is available for diagnostics. A desktop display and Tk are needed for the planner's Matplotlib window.

The setup script defaults to Galactic in `/opt/ros/galactic` and EUFS at `~/eufs_ws`. If your setup differs, set `ROS_SETUP` to the ROS setup script and `EUFS_WS` to your built EUFS workspace before sourcing. Set `EUFS_MASTER` if the simulator source is outside `$EUFS_WS/src/eufs_sim`. Use the same ROS domain and middleware in every terminal.

## Clone and build

```bash
git clone --branch abh_soum_integration https://github.com/DSGxRMS/RAMS-e-25.git
cd RAMS-e-25
source scripts/setup_integration.sh
colcon build --base-paths bringup src --packages-select bringup perception slam pathplanning controls
source install/setup.bash
python3 -c 'import numpy, scipy, matplotlib, torch; print(torch.__version__, torch.cuda.is_available())'
```

From the repository root, source both scripts in every new terminal:

```bash
source scripts/setup_integration.sh
source install/setup.bash
```

## Run the simulation

**Terminal 1**

```bash
ros2 launch eufs_launcher eufs_launcher.launch.py
```

Use Small Track and the same spawn for comparisons. Enable ground truth, select **Acceleration** command mode and use **Manual Drive**.

**Terminal 2**

```bash
MPLBACKEND=TkAgg ros2 launch bringup stack.launch.py
```

Use `MPLBACKEND=Agg` for headless planning.

**Terminal 3**

```bash
ros2 run controls control_node
```

Exp13 is the default. Press Ctrl+C in this terminal and wait for the report path. Each start writes to a fresh `runs/` subdirectory with logs, raw events, an HTML report, and a trajectory plot when controller samples are available. Set `MPPI_RUNS_DIR` to override the directory.

## Repeat a Small Track test

1. Stop the controller and wait for its report. Relaunch Gazebo and bringup for each measured trial so the cones, vehicle, planner, and controls all start fresh.
2. Keep the track, spawn, launch settings, PP build, controller build and recording settings fixed. Arm Manual Drive at the same point in startup. Start scoring at the same forward gate crossing.
3. Start the controller with `MPPI_SEED=42 ros2 run controls control_node`. The default is also 42 and the chosen seed is saved as `mppi_seed` in `run.json`.
4. Repeat several times with the same seed to measure variation from simulator timing and live ROS inputs. Then reuse the same set of seeds for each controller variant. Compare complete laps, cone-contact coverage, recovery and average speed across the group.

The seed repeats the controller's sampling sequence when the device and sequence of solves are the same. The GUI simulator, scheduling and message arrival can still vary.

Exp13 models the Small Track EUFS steering actuator by default. Its rollouts start from the newest valid wheel sample at or before the pose stamp, replay commands already in flight over the 0.20 s simulator delay plus estimated solve latency, and limit wheel travel to 1.04 rad/s. Recent command history supplies an estimate when wheel feedback is unavailable. These constants match the dry DynamicBicycle vehicle. Exp13 reuses the last scored steering plan on the new simulation timeline, accounting for the first command hold, horizon changes and elapsed time.

For a timing A/B comparison, leave the actuator enabled and restart the simulation between trials. Use `MPPI_STEERING_SYNC=0 MPPI_ELAPSED_WARM_START=0` with the controller command to reproduce the earlier actuator-enabled timing. Use `MPPI_STEERING_SYNC=1 MPPI_ELAPSED_WARM_START=0` to isolate timestamp alignment. Both timing fixes are enabled for a normal Exp13 run. To reproduce the original controller before actuator compensation, set `MPPI_STEERING_ACTUATOR=0 MPPI_ELAPSED_WARM_START=0`. The options are recorded in `run.json`. The report plots requested and applied steering, and `applied_steering.csv` contains full-rate stamped wheel samples when Gazebo publishes them.

To see actual planner paths and MPPI rollouts, run `python3 src/controls/controls/debug/pseudo_debug.py` from a sourced terminal.

## Run Exp13 on an RTX 50-series GPU with Galactic

Galactic's Ubuntu packages use Python 3.8. A Blackwell-capable PyTorch wheel can run in a separate Python environment. The ROS controller, Gazebo and the planner continue using their existing environments. The optional worker runs the same Exp13 checkpoint and MPPI solver over a local socket. It requires usable CUDA and keeps the usual ROS topics and run recorder.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed. In WSL, create the GPU environment once:

```bash
uv venv --python 3.11 "$HOME/.venvs/rams-mppi-gpu"
uv pip install --python "$HOME/.venvs/rams-mppi-gpu/bin/python" \
  --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1
uv pip install --python "$HOME/.venvs/rams-mppi-gpu/bin/python" numpy scipy
"$HOME/.venvs/rams-mppi-gpu/bin/python" -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0)); print(torch.ones(8, device="cuda").sum().item())'
```

Source and build the controls package in the usual Galactic workspace. In the controller terminal with the existing Galactic virtual environment active:

```bash
MPPI_WORKER_PYTHON="$HOME/.venvs/rams-mppi-gpu/bin/python" MPPI_SEED=42 \
  ros2 run --prefix "$VIRTUAL_ENV/bin/python3" controls control_node
```

The controller log should show the GPU and both CUDA graph captures before it sends driving commands. The worker interpreter is saved in `run.json`. Run this only for the controller terminal. The planner and Gazebo use their usual launch commands.

## Shared interface

The planner uses `/slam/odom_raw` and `/slam/map_cones`. It publishes `/path_points` in `map`, with ego position first followed by ordered forward points. Controls subscribes using best-effort QoS and publishes `/cmd`. Before enabling controls, verify that `/ground_truth/odom`, `/ground_truth/cones` and `/path_points` are publishing.

Keep planner geometry and controller tracking changes in separate commits. Coordinate the handoff when the planner starts publishing a finished spline. Exp13 uses the planner waypoint order and currently fits its own spline to that ordered input.
