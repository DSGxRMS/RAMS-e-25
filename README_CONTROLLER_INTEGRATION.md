# Path planning and controller integration handoff

The active planner is [`pp_node_skidpad.py`](src/pathplanning/pathplanning/pp_node_skidpad.py). The [stack launch](bringup/launch/stack.launch.py) runs it as `/pp` on ROS 2 Galactic. It generates a short local path from the latest cone map and corrected vehicle pose. A replacement controller can integrate through `/path_points` and `/slam/odom` without importing the planner or SLAM code.

## Controller contract

| Topic | Publisher → consumer | ROS 2 type | Meaning |
| --- | --- | --- | --- |
| `/path_points` | `/pp` → controller | `nav_msgs/msg/Path` | Ordered planar waypoints in the SLAM `map` frame. Best-effort, volatile, keep-last depth 50 by default. |
| `/slam/odom` | `/slam` → `/pp` and controller | `nav_msgs/msg/Odometry` | Corrected vehicle pose in `map`, with `base_link` as the child frame. Reliable, keep-last depth 5. |
| `/cmd` | Controller → EUFS Gazebo vehicle | `ackermann_msgs/msg/AckermannDriveStamped` | Steering and longitudinal command. Use the speed or acceleration field appropriate to the simulator's selected command mode. |
| `/clock` | Gazebo → stack and controller clocks | `rosgraph_msgs/msg/Clock` | Enable `use_sim_time` in the controller to compare timestamps with the stack. |

- Subscribe to `/path_points` with **best-effort** reliability, such as `rclpy.qos.qos_profile_sensor_data`. A reliable subscriber is incompatible with this publisher. The same sensor profile can also receive the reliable odometry publisher.
- **Pose 0 is the car position at planning time.** Poses 1 onward are the targets, ordered from nearest in the greedy traversal. Use the latest odometry for the current vehicle state. Each position is in metres with `z=0`. Every path quaternion is identity, including pose 0. Use odometry orientation for car yaw and adjacent path positions to derive path heading or curvature.
- The planner nominally runs at **20 Hz of ROS time**. Each tick gets a new timestamp and can replace the whole path. Waypoint indices have no persistent identity across messages. The timestamp is the planner's clock at publication, so use odometry freshness and usable waypoint count as separate checks.
- A path with only pose 0 means the tick found no usable forward chain. A path can shrink, jump, or end when the observed cones change. The planner generates positions only. Speed, steering, lookahead, handling of short or stale paths, lap counting, and stop behavior belong to the controller or mission logic.
- The bundled [controller interface](src/controls/controls/ros_connect.py) accepts paths with at least **five total poses** and caches its last accepted path. That threshold and caching behavior are controller choices. If your controller reuses the last good path, give it an explicit expiry. Run a single controller that publishes to `/cmd`. Bringup leaves the controller to be started separately.
- Use `/slam/odom` with `/path_points`. Gazebo's `/ground_truth/odom` and `/ground_truth/track` are useful for visualization, but their origin can differ from the SLAM origin even when both streams say `map`. Align the frames before mixing coordinates.
- The predictor's body-frame `twist.twist.linear.x` is the estimated forward speed in m/s and is derived from a speed magnitude. Its lateral component is zero. `twist.twist.angular.z` is the yaw-rate estimate in rad/s. A controller that needs signed reverse speed or measured lateral velocity needs a suitable source.

## What the planner does

The planner reads `point.x` and `point.y` from the blue, yellow, orange, and big-orange arrays on `/slam/map_cones` (`eufs_msgs/msg/ConeArrayWithCovariance`). It extracts vehicle yaw from `/slam/odom` and runs the following steps in [`_tick`](src/pathplanning/pathplanning/pp_node_skidpad.py#L3052):

1. **Select visible cones.** Subtract the vehicle position and rotate by negative yaw into a temporary vehicle frame with +X forward and +Y left. Test a sector aimed forward, with its apex **5 m behind** the car, **30 m** radius measured from the apex, and **60°** total opening. Keep the positions of accepted cones in `map`.
2. **Normal mode** applies when at least two blue and two yellow cones are inside the sector. Build an undirected Delaunay graph where possible. If SciPy is unavailable, there are too few points, or the geometry is degenerate, use up to three nearest neighbours per point. Keep edges shorter than 6 m. Reject blue–blue and yellow–yellow edges and edges joining a blue or yellow cone to an orange-like cone. Use midpoints of the remaining edges as candidates, except for big-orange to big-orange edges. When at least two big-orange cones are visible, add their midpoint or, for three or more, their centroid as one candidate.
3. **Single-colour fallback** applies when either blue or yellow has at most one visible cone. Select the more numerous blue/yellow side if it contains at least two cones. Add one virtual cone **1 m ahead of the car**, computed as `car_xy + distance * (cos(yaw), sin(yaw))`. Build a graph from the selected real cones plus that virtual cone using Delaunay or nearest neighbours. Use midpoints of all graph edges shorter than 6 m, including edges between selected real cones. This branch skips the normal mode's colour filters and big-orange centroid. A tie or fewer than two cones on the selected side produces no fallback candidates.
4. **Filter candidate midpoints.** Visit candidates farthest from the car first. Drop candidates less than **2 m** from the car or less than **1 m** from an already kept candidate. The big-orange candidate is subject to the same spacing rules.
5. **Order and publish.** Start at the car and repeatedly take the nearest unused candidate within **7 m** of the current point. Stop when none is reachable. Prepend the car pose and publish the result. Before the first corrected odometry there is no path message. Once pose is known, fewer than two visible cones or an empty candidate chain can produce a car-only path.

The greedy output is a sequence of discrete points. It can take a diagonal or end early, and it carries no spline, target velocity, or guarantee of a complete skidpad lap. A downstream controller should check whether the newest path is usable before commanding motion. The Matplotlib plot displays geometry computed by the planner and can be disabled with `MPLBACKEND=Agg`.

| Planner parameter | Default |
| --- | --- |
| `topics.odom_in` | `/slam/odom` |
| `topics.map_in` | `/slam/map_cones` |
| `topics.path_out` | `/path_points` |
| `qos.best_effort`, `qos.depth` | `true`, 50 for input by default. The output remains best-effort. |
| `fov.vertex_offset_m`, `fov.radius_m`, `fov.angle_deg` | −5 m, 30 m, 60° |
| `fallback.virtual_cone_distance_m` | 1 m |
| `candidates.min_car_dist_m`, `candidates.min_spacing_m` | 2 m, 1 m |
| `greedy.max_hop_m`, `publish.rate_hz` | 7 m, 20 Hz |

## Upstream pipeline and changes on this checkout

### Ground-truth mode

`Gazebo /imu/data → /pred → /slam/odom_raw → /slam`  
`Gazebo /ground_truth/cones → /slam → /slam/map_cones and /slam/odom → /pp → /path_points`

- [Prediction](src/slam/slam/prediction_node.py) integrates IMU data and performs stationary bias calibration. Its odometry now publishes forward speed in the child frame, where the controller expects it.
- [SLAM](src/slam/slam/slam_node.py) accepts Gazebo's named cone arrays directly in `ground_truth` mode. It checks `base_footprint` or `base_link`, filters non-finite planar coordinates, and uses the same particle-filter update as the perception input. Use `/ground_truth/cones` as the observation stream. The full-track array on `/ground_truth/track` uses world coordinates and is for inspection.
- SLAM interpolates buffered odometry to the cone timestamp, projects the observations into its local map, and applies map correction once. A rolling map publishes tracks after two hits by default and ages them out after five seconds. Cone frames older than `cones.max_age_sec` (default 0.20 s) are dropped.
- The corrected odometry and map preserve the cone message's timestamp. Corrected odometry includes twist from the nearest buffered raw odometry sample. After initialization, fresh empty cone observations can still produce a prediction-only pose update.
- [SLAM packaging](src/slam/setup.py) now installs `slam.slam_utils` and `slam.debug`. The original copy install omitted the particle-filter module and caused `/slam` to exit before registering its publishers.
- [Bringup](bringup/launch/stack.launch.py) now defaults to `cone_source:=ground_truth`, which starts `/pred`, `/slam`, and `/pp` with simulation time. The planner was restored to the single-virtual-cone version. Its current hardening handles collinear Delaunay inputs through the existing nearest-neighbour fallback and allows plotting to fail without stopping path publication.

### Optional camera and LiDAR mode

Use `ros2 launch bringup stack.launch.py cone_source:=perception` to run the original perception nodes. [LiDAR](src/perception/perception/lidar_node.py) converts `/velodyne_points` to `/perception/cones`. [Camera](src/perception/perception/camera_node.py) uses the stereo ZED images and camera calibration to publish `/perception/cones_stereo`. [Fusion](src/perception/perception/fusion_node.py) pairs clouds within 80 ms by default and publishes `/perception/cones_fused` as `PointCloud2` with `x`, `y`, `z`, and `class_id`. SLAM consumes that cloud in perception mode and the controller interface stays the same.

The current perception sources retain two limitations. The camera copy install omits `yolov11/weights/best.onnx`, so that model must be available beside the installed module for camera startup. Camera class IDs use 0 for yellow and 1 for blue, while SLAM's cloud input ultimately publishes 0 as blue and 1 as yellow. Fusion forwards the camera class IDs. Correct that mapping before relying on cone colours in perception mode. Ground-truth mode classifies cones from the named arrays.

## Existing controls and visualization

The bundled controller uses a nominal maximum speed of 2.5 m/s in this checkout and reduces its target speed with steering. Treat that value and its five-pose rule as reference-controller settings. The path message contains only geometry. The `controls sim_viz` entry point subscribes to Gazebo ground-truth track, cones, and odometry plus `/path_points`. It is a viewer and sends no vehicle commands.

## Run and verify the integration

Use Ubuntu 20.04 or WSL Ubuntu 20.04 with ROS 2 Galactic, Gazebo Classic 11, and a built EUFS workspace. In each terminal, source the ROS and EUFS overlays and the RAMS-e-25 overlay after building it. From the repository root in Bash:

```bash
source /opt/ros/galactic/setup.bash
source "$HOME/eufs_ws/install/setup.bash"
colcon build
source install/setup.bash
```

Start EUFS Gazebo with the desired track and ground-truth publishing enabled. For example, from a terminal with EUFS sourced:

```bash
export EUFS_MASTER="$HOME/eufs_ws/src/eufs_sim"
ros2 launch eufs_launcher simulation.launch.py track:=skidpad \
  commandMode:=acceleration launch_group:=default pub_ground_truth:=true
```

In another terminal with the RAMS-e-25 overlay sourced, run `ros2 launch bringup stack.launch.py`. Keep the controller stopped while checking data. Run streaming checks one at a time:

```bash
ros2 node info /slam
ros2 node info /pp
ros2 topic info -v /path_points
ros2 topic echo /ground_truth/cones --qos-reliability best_effort
ros2 topic echo /slam/odom --qos-reliability best_effort
ros2 topic echo /slam/map_cones --qos-reliability best_effort
ros2 topic echo /path_points --qos-reliability best_effort
```

1. Confirm `/slam` publishes finite pose and velocity and that the map contains cones. If either stream is silent, check `/clock`, `/imu/data`, `/slam/odom_raw`, and the configured cone input.
2. Inspect `/path_points` in the `map` frame and verify that its first pose is the vehicle and that there are enough subsequent poses for the controller's lookahead. Verify the replacement controller receives it with best-effort QoS.
3. Test the controller in simulation with missing odometry, a car-only path, and a normal path before enabling its command output. Select the EUFS driving mode and command mode that match the controller.

### Reference cases

These inputs are SLAM-map cone coordinates in metres with the car at `(0, 0)`, yaw zero. Pose counts include the car. With SciPy available and default planner settings:

| Cone input | Expected behavior |
| --- | --- |
| Blue `(3,1), (6,1), (9,1)`, yellow `(3,-1), (6,-1), (9,-1)` | Six poses along `y=0` at `x=0, 3, 4.5, 6, 7.5, 9`. |
| The same three blue cones, zero yellow cones | Five poses `(0,0), (2,0.5), (3.5,0.5), (4.5,1), (7.5,1)`. |
| Blue `(3,0), (5,0)`, yellow `(4,0), (6,0)` | Degenerate triangulation falls back to nearest neighbours and publishes a path. |
| Empty map with valid odometry | One pose at the car. |

The updated pipeline was also tested by replaying Gazebo IMU and cone messages in an isolated ROS domain. It produced 71 nonempty cone maps, 71 corrected odometry messages, and 67 paths, with up to six poses. That check covered the planner and localization handoff with controls stopped.

For general setup, plot commands, and troubleshooting, see the [stack README](README.md). The built-in planner plot is part of `/pp`. `ros2 run slam slam_debug` and `ros2 run controls sim_viz` are separate viewers. Gazebo assets live in the separate EUFS workspace, so the examples use its installed track files.
