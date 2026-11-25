# RAMS-e-25 autonomous vehicle stack

RAMS-e-25 is a ROS 2 stack for running the EUFS vehicle in Gazebo Classic. The default simulation pipeline takes vehicle-frame cone observations directly from Gazebo, estimates the vehicle pose and cone map with SLAM, and generates a skidpad path. Controls runs as a separate process.

For controller integration, see the [path planning handoff](README_CONTROLLER_INTEGRATION.md).

## Prerequisites

- Ubuntu 20.04 or WSL Ubuntu 20.04 with ROS 2 Galactic, Gazebo Classic 11, and `colcon`.
- A built EUFS workspace with `eufs_launcher`, `eufs_msgs`, and the Gazebo vehicle and sensor plugins. The examples assume `~/eufs_ws`.
- The stack's Python dependencies available in the same ROS environment. SLAM and planning use NumPy. SciPy enables Hungarian association and Delaunay triangulation. Live plots use Matplotlib and a working GUI backend.
- Gazebo and the stack on the same ROS domain. In WSL, Gazebo's GUI and the plots require a working display such as WSLg.

## Build

From a ROS 2 Bash terminal, adjust the workspace paths if your checkout is elsewhere.

```bash
source /opt/ros/galactic/setup.bash
source "$HOME/eufs_ws/install/setup.bash"
cd /mnt/d/RAMSe/RAMS-e-25
colcon build
source install/setup.bash
```

Source the ROS, EUFS, and RAMS-e-25 setup files in each new terminal used below. After editing only SLAM and bringup, use `colcon build --packages-select slam bringup` from the RAMS-e-25 root and source the resulting setup file again.

## Run in Gazebo

### 1. Start EUFS

With the EUFS workspace sourced, start a single simulator session.

```bash
export EUFS_MASTER="$HOME/eufs_ws/src/eufs_sim"
ros2 launch eufs_launcher simulation.launch.py \
  track:=skidpad vehicleModel:=DynamicBicycle commandMode:=acceleration \
  vehicleModelConfig:=configDry.yaml robot_name:=eufs \
  launch_group:=default pub_ground_truth:=true gazebo_gui:=true rviz:=false
```

Set `track:=small_track` to use that circuit. If launching through the EUFS GUI, select the track there and enable ground-truth publishing. The default SLAM pipeline uses Gazebo's `/ground_truth/cones` and `/imu/data`.

### 2. Start SLAM and the planner

In a separate terminal with the RAMS-e-25 overlay sourced:

```bash
ros2 launch bringup stack.launch.py
```

The default `cone_source:=ground_truth` starts `/pred`, `/slam`, and `/pp`. The planner window appears when a GUI backend is available and draws once corrected odometry arrives.

### 3. Start controls when ready to drive

After confirming that SLAM and the planner publish messages:

```bash
ros2 run controls control_node
```

The controller starts as `/ros_interface`, consumes the path and corrected odometry, and sends Ackermann commands to the simulated vehicle. Select **Manual Drive** in EUFS Mission Control for the simulation to accept driving commands. The controller needs an initial path with at least five poses.

## ROS 2 connections

| Topic | Publisher → subscriber | Message type |
| --- | --- | --- |
| `/clock` | Gazebo → `/pred`, `/slam`, `/pp`, and controls when started | `rosgraph_msgs/msg/Clock` |
| `/imu/data` | Gazebo IMU → `/pred` | `sensor_msgs/msg/Imu` |
| `/ground_truth/cones` | Gazebo ground-truth plugin → `/slam` in ground-truth mode | `eufs_msgs/msg/ConeArrayWithCovariance` |
| `/slam/odom_raw` | `/pred` → `/slam` | `nav_msgs/msg/Odometry` |
| `/slam/odom` | `/slam` → `/pp` and `/ros_interface` | `nav_msgs/msg/Odometry` |
| `/slam/map_cones` | `/slam` → `/pp` | `eufs_msgs/msg/ConeArrayWithCovariance` |
| `/path_points` | `/pp` → `/ros_interface` and optional path viewers | `nav_msgs/msg/Path` |
| `/cmd` | `/ros_interface` → Gazebo vehicle | `ackermann_msgs/msg/AckermannDriveStamped` |

- [Prediction](src/slam/slam/prediction_node.py) publishes IMU-based odometry for SLAM.
- [SLAM](src/slam/slam/slam_node.py) combines odometry and cone observations. In ground-truth mode it expects cones in `base_footprint` or `base_link`, and publishes the estimated pose and cone map in `map`. Gazebo's `/ground_truth/track` is a full-track map for visualization.
- [Skidpad planning](src/pathplanning/pathplanning/pp_node_skidpad.py) is the planner started by [bringup](bringup/launch/stack.launch.py). It filters cones by the car's field of view, generates candidate waypoints, and publishes a greedy path in `map` with the car pose first. Its path publisher uses best-effort QoS.
- [Controls](src/controls/controls/control_node.py) uses the path and corrected odometry to compute steering and longitudinal commands.

### Optional perception mode

```bash
ros2 launch bringup stack.launch.py cone_source:=perception
```

This mode starts `/lidar`, `/camera`, and `/fusion`. The camera and LiDAR nodes publish cone detections, and fusion sends `sensor_msgs/msg/PointCloud2` on `/perception/cones_fused` to SLAM. The SLAM and planner output topics are shared by both modes.

The camera loads `perception/yolov11/weights/best.onnx` relative to its installed Python module. The model is present in the source tree, and the current copy-install packaging omits it. Perception mode needs that resource available in the installed package.

## Plotting and troubleshooting

- The planner's built-in plot belongs to `/pp`. A blank window can mean it is waiting for corrected odometry. On a headless machine, the planner can still publish paths.
- `ros2 run slam slam_debug` opens the SLAM comparison plot. It uses SLAM outputs, raw odometry, and Gazebo ground-truth odometry.
- `ros2 run controls sim_viz` opens a standalone viewer for Gazebo ground truth and `/path_points`. The separate `pathplanning pp_debug` and `pathplanning pp_viz` commands require the `pathplanning.pp_debug` subpackage to be included in the install.
- **Autonomous Control Telemetry** opens from `control_node` after it receives corrected odometry and accepts a path with at least five poses.

To find the first silent topic, check the graph and then check for actual messages. Run the echo commands separately and stop each with Ctrl-C.

```bash
ros2 node list
ros2 topic list -t
ros2 node info /slam
ros2 topic info -v /slam/map_cones
ros2 topic echo /ground_truth/cones --qos-reliability best_effort
ros2 topic echo /slam/odom --qos-reliability best_effort
ros2 topic echo /slam/map_cones --qos-reliability best_effort
ros2 topic echo /path_points --qos-reliability best_effort
```

| Symptom | Check |
| --- | --- |
| Gazebo shows a different track from the one selected | Check whether a previous Gazebo server is still running on the same master port. |
| `/slam` is missing and the launch log reports `No module named 'slam.slam_utils'` | Rebuild `slam` from this checkout and source its overlay. |
| `/slam/map_cones` is listed but silent | Check that `/slam` is a publisher, both `/ground_truth/cones` and `/slam/odom_raw` carry messages, and `/clock` is advancing. |
| Controls prints “Waiting for first odometry message” | Check `/slam/odom` and the SLAM launch log. |
| Perception build reports a syntax error in `tempCodeRunnerFile.py` | This is an editor scratch file copied by the perception package. Check the build summary and the runtime log for the node you are diagnosing. |
