# Selectable LiDAR ground removal

The LiDAR node supports two ground-removal backends through the
`ground_removal_method` ROS parameter:

- `patchwork` (default): Patchwork++ with full-scan context.
- `ransac`: the existing plane model with vectorized hypothesis evaluation.

Install the Python dependency before building the workspace:

```bash
python3 -m pip install -r src/perception/requirements.txt
```

Build and source the workspace, then use the installed configuration:

```bash
colcon build --packages-select perception
source install/setup.bash
ros2 run perception lidar_node --ros-args \
  --params-file install/perception/share/perception/config/lidar.yaml
```

Override the backend at runtime:

```bash
ros2 run perception lidar_node --ros-args \
  -p ground_removal_method:=ransac
```

`cluster_eps` defaults to `0.0`, which selects the benchmarked radius for the
chosen backend: 0.50 m for Patchwork++ and 1.00 m for RANSAC. Set a positive
value to override it explicitly.

## Single-lap EUFS result

On the saved 574-scan figure-eight lap, vectorizing RANSAC preserved every
retained-point decision and improved its ground-removal median by 2.44x.

| Method | Precision | Recall | F1 | Ground median/p95 | Pipeline median/p95 |
|---|---:|---:|---:|---:|---:|
| Vectorized RANSAC | 99.59% | 79.09% | 88.16% | 13.75/20.85 ms | 48.39/70.17 ms |
| Tuned Patchwork++ | 98.31% | 79.77% | 88.07% | 4.35/6.59 ms | 46.47/65.59 ms |

These simulator results do not replace testing on sloped or uneven real-world
tracks, where a locally adaptive ground model may have a larger advantage.
