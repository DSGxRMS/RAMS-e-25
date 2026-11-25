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

To see actual planner paths and MPPI rollouts, run `python3 src/controls/controls/debug/pseudo_debug.py` from a sourced terminal.

## Shared interface

The planner uses `/slam/odom_raw` and `/slam/map_cones`. It publishes `/path_points` in `map`, with ego position first followed by ordered forward points. Controls subscribes using best-effort QoS and publishes `/cmd`. Before enabling controls, verify that `/ground_truth/odom`, `/ground_truth/cones` and `/path_points` are publishing.

Keep planner geometry and controller tracking changes in separate commits. Coordinate the handoff when the planner starts publishing a finished spline. Exp13 uses the planner waypoint order and currently fits its own spline to that ordered input.
