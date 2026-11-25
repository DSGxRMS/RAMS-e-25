# Run the RAMS-e-25 stack in Gazebo

Use branch `abh_soum_integration` for the ground-truth EUFS Small Track stack, with Abhi's path planner and Soumil's mirror-symmetric Exp13 Neural ODE MPPI. This guide uses the Gazebo GUI. All commands below are for Bash in Ubuntu 20.04, including Ubuntu 20.04 under WSL 2. For controller changes, verification and upcoming objectives, see the [controls changelog](changes-controls.md).

## 1. Set up ROS and EUFS once

If ROS 2 Galactic and a working EUFS Gazebo Classic workspace are already installed, skip to [step 2](#2-get-and-build-the-team-stack). For a new environment, install Galactic Desktop on Ubuntu 20.04 first. Galactic is end of life. If the older ROS packages are unavailable on a new machine, use the team's working Ubuntu 20.04 environment or arrange to copy it before building the stack. On Windows, check that WSL 2 can open GUI applications and that the NVIDIA Windows driver exposes the GPU in WSL. [ROS Galactic Ubuntu installation](https://github.com/ros2/ros2_documentation/blob/galactic/source/Installation/Ubuntu-Install-Debians.rst) · [Microsoft WSL GUI setup](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps) · [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/)

Run the following only for a new EUFS workspace. The revisions match the reference environment used for Small Track.

```bash
sudo apt update
sudo apt install git python3-colcon-common-extensions python3-rosdep \
  python3-pip python3-venv python3-tk python3-numpy python3-scipy python3-matplotlib

source /opt/ros/galactic/setup.bash
mkdir -p "$HOME/eufs_ws/src"
cd "$HOME/eufs_ws/src"

git clone https://gitlab.com/eufs/public/eufs_sim.git
git clone https://gitlab.com/eufs/public/eufs_msgs.git
git clone https://gitlab.com/eufs/public/eufs_rviz_plugins.git
git -C eufs_sim switch --detach 0390cdd1580b7fa40fdcb7947dd93cf1ed42033b
git -C eufs_msgs switch --detach 9e918686c9e9292c613f321e6fd85e3a5d87cd87
git -C eufs_rviz_plugins switch --detach f1f236d1918996572c7502632cf48734049cb396

cd "$HOME/eufs_ws"
# Run sudo rosdep init once per machine if rosdep has never been initialized.
rosdep update --include-eol-distros
rosdep install --from-paths src --ignore-src --rosdistro galactic -r -y
colcon build
source install/setup.bash
ros2 pkg prefix eufs_launcher
```

The last command should print the installed EUFS launcher path. The simulator sources and dependencies follow the original Classic simulator package layout. [EUFS simulator repository](https://gitlab.com/eufs/public/eufs_sim)

## 2. Get and build the team stack

For a new checkout:

```bash
git clone --branch abh_soum_integration https://github.com/DSGxRMS/RAMS-e-25.git "$HOME/RAMS-e-25-integration"
cd "$HOME/RAMS-e-25-integration"
```

For an existing checkout, change into its actual directory. For example, it may be `~/RAMS-e-25-shared` or `/mnt/d/RAMSe/RAMS-e-25-integration`. On the shared branch:

```bash
git branch --show-current
# Expect abh_soum_integration. Keep any local changes before switching branches.
git pull --ff-only origin abh_soum_integration
```

Build from the repository root:

```bash
source scripts/setup_integration.sh
colcon build --base-paths bringup src --packages-select bringup perception slam pathplanning controls
source install/setup.bash
ros2 pkg prefix eufs_launcher
ros2 pkg prefix bringup
ros2 pkg executables controls
/usr/bin/python3 -c 'import rclpy, eufs_msgs, numpy, scipy, matplotlib, tkinter; print("Galactic Python imports OK")'
```

If you reused an EUFS installation and the Python check reports a missing numerical or plotting package, install it with `sudo apt install python3-numpy python3-scipy python3-matplotlib python3-tk` and repeat the check.

If your ROS or EUFS installation is elsewhere, set `ROS_SETUP` to Galactic's `setup.bash` and `EUFS_WS` to the root of the built EUFS workspace before sourcing. Set `EUFS_MASTER` only if the simulator source is outside `$EUFS_WS/src/eufs_sim`. Use the same values in every terminal. When a later update touches only controls, rebuild from the repository root with `colcon build --base-paths src --packages-select controls` after sourcing `scripts/setup_integration.sh`.

## 3. Choose the controller's GPU setup

Run `nvidia-smi` in Ubuntu or WSL. Then use the same interpreter for the PyTorch check that you will use to launch the controller. The test must print the GPU name and **8.0**, confirming that a CUDA operation actually ran.

### A. Direct CUDA on Galactic Python

Use this if your existing Galactic Python environment already runs CUDA operations. The reference RTX 4050 uses PyTorch `2.4.1+cu121` on Python 3.8. If you need to create an environment for that configuration:

```bash
/usr/bin/python3 -m venv --system-site-packages "$HOME/.venvs/rams-galactic"
"$HOME/.venvs/rams-galactic/bin/python3" -m pip install \
  --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1
"$HOME/.venvs/rams-galactic/bin/python3" -m pip install numpy==1.23.5 scipy==1.10.1
```

Set `ros_python` to your working interpreter. For the venv above:

```bash
ros_python="$HOME/.venvs/rams-galactic/bin/python3"
"$ros_python" -c 'import rclpy, torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0)); print(torch.ones(8, device="cuda").sum().item())'
```

If CUDA PyTorch is already installed for the system interpreter, set `ros_python=/usr/bin/python3`. For an existing Galactic virtual environment, set `ros_python="$VIRTUAL_ENV/bin/python3"` while it is active. [Official PyTorch version and wheel list](https://pytorch.org/get-started/previous-versions/)

### B. RTX 50-series or a separate CUDA Python environment

Galactic's Ubuntu ROS extensions use Python 3.8. The RTX 5050 can run a Blackwell-capable PyTorch wheel in a separate Python 3.11 environment. The ROS node, planner and recorder still use Galactic Python. This branch includes an optional worker that runs the same Exp13 solver and checkpoint. [PyTorch Blackwell support](https://pytorch.org/blog/pytorch-2-7/) · [Galactic Python compatibility](https://github.com/ros2/ros2_documentation/blob/galactic/source/How-To-Guides/Using-Python-Packages.rst)

Install `uv` if needed, then run this once in Bash. [uv installation](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv venv --python 3.11 "$HOME/.venvs/rams-mppi-gpu"
gpu_python="$HOME/.venvs/rams-mppi-gpu/bin/python"
uv pip install --python "$gpu_python" \
  --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1
uv pip install --python "$gpu_python" numpy scipy
"$gpu_python" -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0)); print(torch.ones(8, device="cuda").sum().item())'
```

The ROS interpreter needs `rclpy`, the EUFS messages, NumPy, SciPy and Matplotlib. This launch option loads PyTorch in the GPU worker. Keep your existing Galactic venv if it passes the ROS import check. If it is unnecessary, use `/usr/bin/python3` on the ROS side.

The full 1,024-rollout CPU fallback is useful for offline diagnostics. In a recorded CPU Gazebo run the 3.0 m/s target was overshot to 7.20 m/s and most steering solves had no remaining time to affect their rollouts. Use a working CUDA configuration for driving comparisons.

## 4. Launch Small Track in three terminals

At the start of **each** terminal, change into the repository. Substitute your actual checkout path if it differs. Source in this order:

```bash
cd "$HOME/RAMS-e-25-integration"
source scripts/setup_integration.sh
source install/setup.bash
export ROS_DOMAIN_ID=0
```

Use the same ROS domain and middleware for Gazebo, bringup, controls and any debug terminal. If your setup requires a particular `RMW_IMPLEMENTATION`, use that setting in all terminals.

### Terminal 1 · EUFS launcher

```bash
ros2 launch eufs_launcher eufs_launcher.launch.py
```

In the launcher, choose **Small Track** (`small_track.launch`), robot **eufs**, **DynamicBicycle**, **DryTrack**, and command mode **acceleration**. Enable **Gazebo GUI**, **Publish Ground Truth** and **Use Simulated Perception**. Use the default spawn and keep cone noise settings fixed across comparisons. Click **Launch** and wait for the car and cones to appear. If Gazebo is paused, resume physics. The controller issues acceleration commands in m/s². The speed field records its target.

### Terminal 2 · Team bringup and the PP window

```bash
MPLBACKEND=TkAgg ros2 launch bringup stack.launch.py
```

Wait for the planner's window and for an ordered path to appear. Bringup starts fusion, the ground-truth odometry relay, SLAM in GT passthrough and the planner. Controls are started separately. If the plot fails to open, check that Tk is available and the terminal can display windows. `MPLBACKEND=Agg` starts this same bringup with its plot window disabled.

Before starting controls, check the simulator and planner streams from Terminal 3 after sourcing the environment. Run each line individually and press Ctrl+C after seeing messages. At spawn, `/path_points` should contain the car pose followed by several forward waypoints.

```bash
ros2 topic echo /ground_truth/odom --field pose.pose.position
ros2 topic echo /ground_truth/cones
ros2 topic echo /path_points --field poses
```

### Terminal 3 · Exp13 and automatic recording

Pick **one** of the commands below. Set `ros_python` in this terminal to the Galactic interpreter chosen in step 3. Both options use the team controller entry point and record automatically.

Direct CUDA:

```bash
# Example interpreter. Use /usr/bin/python3 if that passed the CUDA check.
ros_python="$HOME/.venvs/rams-galactic/bin/python3"
MPPI_MODEL=exp13 MPPI_SEED=42 ros2 run --prefix "$ros_python" controls control_node
```

Separate GPU worker:

```bash
# Use "$VIRTUAL_ENV/bin/python3" here if using an activated Galactic venv.
ros_python=/usr/bin/python3
MPPI_WORKER_PYTHON="$HOME/.venvs/rams-mppi-gpu/bin/python" MPPI_MODEL=exp13 MPPI_SEED=42 \
  ros2 run --prefix "$ros_python" controls control_node
```

Wait for **`[MPPI] GPU ... + CUDA graph`**, the **200- and 400-step CUDA graph captures**, and startup warm-up completion. In EUFS Mission Control click **Manual Drive** once. Leave the steering controls alone. Use the same arm point for comparable runs. If a second terminal may be controlling the car, check `ros2 topic info /cmd -v` and confirm that the running controller is the only command publisher.

### Optional terminal · MPPI and PP overlay

Use the same checkout and sources as the other terminals. This opens the separate Matplotlib window with raw PP, the controller reference, the recovery path and the cheapest MPPI rollouts.

```bash
/usr/bin/python3 src/controls/controls/debug/pseudo_debug.py
```

## 5. Finish, share and repeat a run

1. Press **Ctrl+C in the controller terminal** and wait for **`[run] Report:`** before closing the terminal. The controller writes a fresh `runs/<id>/` folder. `MPPI_RUNS_DIR` overrides that location.
2. Open `report.html` and `trajectory.png` in that folder. Share the whole folder if another teammate needs to diagnose the run. It includes the model, seed and worker setting in `run.json`, `controller.log`, the raw `events.jsonl`, `timeline.csv`, `ground_truth.csv` and `applied_steering.csv` when wheel feedback is available. A contact count marked unknown means the recorder received no contact payload.
3. For a new comparable run, stop bringup and the EUFS launcher after the report is written and relaunch from step 4. This restores the starting cone poses, car pose and wheel state. Keep the track, launch settings, planner build and seed fixed. Seed 42 repeats the optimizer's random sequence under the same device and sequence of solves. Simulator and ROS timing can still vary.

For a timing A/B test, leave `MPPI_STEERING_ACTUATOR=1` and `MPPI_STEERING_SYNC=1` in both trials and toggle `MPPI_ELAPSED_WARM_START=0` versus `1` on the controller command. Exp13 defaults to all three on. The settings are saved in `run.json`.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `Package 'eufs_launcher' not found` | Check `EUFS_WS` and confirm the simulator built. Source `scripts/setup_integration.sh` in that terminal. |
| `Package 'bringup' not found` | Build from both `bringup` and `src`, then source this checkout's `install/setup.bash`. |
| Nothing on `/ground_truth/odom` | Check Publish Ground Truth, car spawn, pause state, ROS domain and middleware. |
| Ground truth arrives but `/path_points` has no useful waypoints | Check Terminal 2 for errors and verify `/ground_truth/cones`, `/slam/odom_raw` and `/slam/map_cones` have data. |
| No PP window | Use `MPLBACKEND=TkAgg`, check `python3-tk` and GUI access, and read the planner's live-plot warning if present. |
| `CPU mode`, severe surging, or no CUDA graph captures | End the controller and check the launcher interpreter or `MPPI_WORKER_PYTHON`. `torch.version.cuda is None` identifies a CPU-only PyTorch build. |
| Gazebo has the car and the path is live but the car stays still | Check Manual Drive, acceleration mode, the paused state and that `/cmd` has one active publisher. |

For comparisons, the shared planner interface is `/slam/odom_raw` and `/slam/map_cones` in, `/path_points` in `map` out. The first path pose is the ego pose. Controls publish `/cmd`, use the planner's waypoint order and fit a spline to it. Keep planner and controls changes in separate commits.
