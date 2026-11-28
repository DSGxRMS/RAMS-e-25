# Run the RAMS-e-25 stack in Gazebo

The `srijan` branch runs the ground-truth EUFS stack with the TEL planner and Exp13 MPPI. This guide uses the Gazebo GUI. Run the commands in Bash in Ubuntu 20.04, including Ubuntu 20.04 under WSL 2. For controller changes, see the [controls changelog](changes-controls.md).

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

In WSL, enter the Windows checkout you edit. Adjust the path if your checkout is elsewhere.

```bash
cd /mnt/c/Code/RMS/RAMS-e-25
git branch --show-current
bash scripts/run_integration.sh build
```

The launcher builds directly from this checkout into `~/rams-wsl/build` and `~/rams-wsl/install`. Build logs go to `~/rams-wsl/log`. If you have not created the Galactic venv yet, follow step 3 or set `RAMS_ROS_PYTHON=/usr/bin/python3` if the system interpreter has your ROS and numerical dependencies.

After editing only controls, rebuild and then restart the controls terminal. Use `build` without a package name after switching branches or changing multiple packages.

```bash
bash /mnt/c/Code/RMS/RAMS-e-25/scripts/run_integration.sh build controls
```

To run ROS diagnostics directly from the checkout, source the EUFS underlay and the WSL install.

```bash
source scripts/setup_integration.sh
source "${RAMS_WSL_WORKSPACE:-$HOME/rams-wsl}/install/local_setup.bash"
ros2 pkg prefix eufs_launcher
ros2 pkg prefix bringup
ros2 pkg executables controls
```

If your ROS or EUFS installation is elsewhere, set `ROS_SETUP` to Galactic's `setup.bash` and `EUFS_WS` to the root of the built EUFS workspace. Set `EUFS_MASTER` if the simulator source is outside `$EUFS_WS/src/eufs_sim`. If you set `RAMS_WSL_WORKSPACE` to a different Ubuntu directory, use that same path for the build and all launch terminals.

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

## 4. Launch in three WSL terminals

Each command sources ROS, EUFS, and the WSL build automatically. Use the same ROS domain and middleware for Gazebo, bringup, controls, and any debug terminal. The launcher defaults to ROS domain 0.

### Terminal 1 · EUFS launcher

```bash
bash /mnt/c/Code/RMS/RAMS-e-25/scripts/run_integration.sh sim
```

In the launcher, choose your track, robot **eufs**, **DynamicBicycle**, **DryTrack**, and command mode **acceleration**. Enable **Gazebo GUI**, **Publish Ground Truth** and **Use Simulated Perception**. Use the default spawn and keep cone noise settings fixed across comparisons. Click **Launch** and wait for the car and cones to appear. If Gazebo is paused, resume physics. The controller issues acceleration commands in m/s². The speed field records its target.

### Terminal 2 · Team bringup and the PP window

```bash
bash /mnt/c/Code/RMS/RAMS-e-25/scripts/run_integration.sh planner
```

Bringup starts fusion, the ground-truth odometry relay, SLAM in GT passthrough, and the TEL planner. The planner runs headless by default. To view the path and steering, open the optional debug tab.

To check the simulator and planner streams, source the environment as in step 2 and run these commands separately. At spawn, `/path_points` should contain the car pose followed by several forward waypoints.

```bash
ros2 topic echo /ground_truth/odom --field pose.pose.position
ros2 topic echo /ground_truth/cones
ros2 topic echo /path_points --field poses
```

### Terminal 3 · Exp13 and automatic recording

```bash
bash /mnt/c/Code/RMS/RAMS-e-25/scripts/run_integration.sh controls
```

The launcher uses the separate CUDA worker by default. If your Galactic interpreter runs CUDA directly, set `MPPI_WORKER_PYTHON=` on the launch command to use that interpreter for MPPI as well.

Wait for **`[MPPI] GPU ... + CUDA graph`**, the **200- and 400-step CUDA graph captures**, and startup warm-up completion. In EUFS Mission Control click **Manual Drive** once. Leave the steering controls alone. Use the same arm point for comparable runs. If a second terminal may be controlling the car, check `ros2 topic info /cmd -v` and confirm that the running controller is the only command publisher.

### Optional terminal · MPPI and PP overlay

Use the same checkout and sources as the other terminals. This opens the separate Matplotlib window with raw PP, the controller reference, the recovery path and the cheapest MPPI rollouts.

```bash
bash /mnt/c/Code/RMS/RAMS-e-25/scripts/run_integration.sh debug
```

## 5. Finish, share and repeat a run

1. Press **Ctrl+C in the controller terminal** and wait for **`[run] Report:`** before closing the terminal. The launcher writes each recording to `~/rams-wsl/runs/<id>/` by default. `MPPI_RUNS_DIR` overrides that location.
2. Open `report.html` and `trajectory.png` in that folder. Share the whole folder if another teammate needs to diagnose the run. It includes the model, seed and worker setting in `run.json`, `controller.log`, the raw `events.jsonl`, `timeline.csv`, `ground_truth.csv` and `applied_steering.csv` when wheel feedback is available. A contact count marked unknown means the recorder received no contact payload.
3. For a new comparable run, stop bringup and the EUFS launcher after the report is written and relaunch from step 4. This restores the starting cone poses, car pose and wheel state. Keep the track, launch settings, planner build and seed fixed. Seed 42 repeats the optimizer's random sequence under the same device and sequence of solves. Simulator and ROS timing can still vary.

For a timing A/B test, leave `MPPI_STEERING_ACTUATOR=1` and `MPPI_STEERING_SYNC=1` in both trials and toggle `MPPI_ELAPSED_WARM_START=0` versus `1` on the controller command. Exp13 defaults to all three on. The settings are saved in `run.json`.

Exp13 enables cone clearance and clearance-based speed limiting by default. For a baseline run, set `MPPI_CONE_CLEARANCE=0`. To keep the clearance steering checks and disable only the speed limit add-on, set `MPPI_CLEARANCE_SPEED=0`. Restart from a fresh world for each comparison. In the report, check feasible rollouts, the applied speed cap, the rollout target, and the selected margin. Use the default simulator car and its normal ground-truth cone stream.

Clearance uses a conservative eight-sided outline of the default EUFS chassis, the four tyre boxes, and a 0.10 m cone buffer. With clearance speed limiting enabled, Exp13 can retry at 0.5 m/s when the current batch has no fully clear path. It prefers crawl paths that improve the gap to each cone already inside the buffer and keep the full buffer around the other cones. These paths are recorded as the `escape` decision within `clearance_crawl`. If there are no such paths, the existing crawl fallback can still accept a cone contact. The normal speed cap recovers as fully clear paths return.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `Package 'eufs_launcher' not found` | Check `EUFS_WS` and confirm the simulator built. Source `scripts/setup_integration.sh` in that terminal. |
| `Package 'bringup' not found` | Run the launcher's `build` command. For manual ROS commands, source the WSL install as in step 2. |
| Nothing on `/ground_truth/odom` | Check Publish Ground Truth, car spawn, pause state, ROS domain and middleware. |
| Ground truth arrives but `/path_points` has no useful waypoints | Check Terminal 2 for errors and verify `/ground_truth/cones`, `/slam/odom_raw` and `/slam/map_cones` have data. |
| No PP window | Use `MPLBACKEND=TkAgg`, check `python3-tk` and GUI access, and read the planner's live-plot warning if present. |
| `CPU mode`, severe surging, or no CUDA graph captures | End the controller and check the launcher interpreter or `MPPI_WORKER_PYTHON`. `torch.version.cuda is None` identifies a CPU-only PyTorch build. |
| Gazebo has the car and the path is live but the car stays still | Check Manual Drive, acceleration mode, the paused state and that `/cmd` has one active publisher. |

For comparisons, the shared planner interface is `/slam/odom_raw` and `/slam/map_cones` in, `/path_points` in `map` out. The first path pose is the ego pose. Controls publish `/cmd`, use the planner's waypoint order and fit a spline to it. Keep planner and controls changes in separate commits.
