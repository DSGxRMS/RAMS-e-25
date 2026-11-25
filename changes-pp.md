# Path planning changes and next objectives

Branch `tel_soum_integration` · Ground-truth EUFS Small Track · Updated 25 November 2026

The active planner on this branch is **`pp_tls`**: the team planner `pp_node_skidpad` with **Predictive Corridor Delaunay (PCD)** tried first on every tick. It publishes the same `/path_points` message as `pp_node_skidpad`, so the controller needs no change. `pp_node_skidpad` itself is unchanged and is still launched by `stack.launch.py`. On the 25 November Small Track runs, the controller tracked the PCD path in every cycle (no pseudo-path feeds), with no recorded cone contact and a closest cone centre of 1.24 m. Lap time was 23.4 s simulation time at a 5 m/s cap and 20.8 s at a 6.5 m/s cap. The remaining planner-side weakness is a 0.4–0.6 m cross-track spike at the turn-in of the same three bends on every lap.

## How the planner works

### What it is for

The team Delaunay planner connects blue and yellow cones it can currently see. In a bend, one wall often leaves the camera sector before the other: the inside cones are hidden, the Delaunay triangles collapse, and the path is short, kinked or missing. PCD keeps the Delaunay idea and adds four things to it: a memory of cones already seen, walls traced as ordered chains, a prediction of where a wall continues into the blind part of a bend, and virtual partner cones where one wall is missing. Any scene PCD cannot handle well is handed back to the original team logic.

### Files

| File | Role |
| --- | --- |
| [pp_tls.py](src/pathplanning/pathplanning/pp_tls.py) | The node. Lines up to the `PCD PLANNER` banner are the embedded PCD planner (pure NumPy/SciPy, no ROS). The rest is `pp_node_skidpad` unchanged, plus `_try_pcd`, `_update_color_side`, `_cut_at_walls`, `_set_pcd_plot` and `_publish_debug`. |
| [pp_node_skidpad.py](src/pathplanning/pathplanning/pp_node_skidpad.py) | Team planner. Not edited. Its NORMAL/FALLBACK/big-orange logic is copied into `pp_tls` and used as the fallback. |
| [setup.py](src/pathplanning/setup.py) | Entry point `pp_tls = pathplanning.pp_tls:main`. |
| [stack_tls.launch.py](bringup/launch/stack_tls.launch.py) | `stack.launch.py` with `pp_tls` (node name `pp_tls`) in place of `pp_node_skidpad`. Same parameters, including `topics.odom_in: /slam/odom_raw`. |
| [birdview.py](tools/birdview.py) | Read-only bird's-eye viewer. Draws cones, car trail, `/path_points`, the controller's tracked path and rollouts, and the PCD internals from `/pp/debug_markers` and `/pp/mode`. |

Dependencies: `rclpy`, `nav_msgs`, `geometry_msgs`, `eufs_msgs`, `std_msgs`, `visualization_msgs` (optional, debug topics are skipped without it), NumPy, SciPy (`Delaunay`, `cKDTree`, `splprep`), Matplotlib (optional live plot).

### Inputs and outputs

| Topic | Type | Direction | Use |
| --- | --- | --- | --- |
| `/slam/odom_raw` (param `topics.odom_in`) | `nav_msgs/Odometry` | in | Car position and heading in `map` |
| `/slam/map_cones` (param `topics.map_in`) | `eufs_msgs/ConeArrayWithCovariance` | in | Blue, yellow, orange, big orange cones. The SLAM node publishes only cones near the car, typically 7–10 |
| `/path_points` (param `topics.path_out`) | `nav_msgs/Path` | out | Frame `map`, car pose first, then the path at 0.5 m spacing. BEST_EFFORT. Timer at `publish.rate_hz` (20 Hz) |
| `/pp/mode` | `std_msgs/String` | out | JSON `{mode, reason, width, left}` |
| `/pp/debug_markers` | `visualization_msgs/MarkerArray` | out | Walls, predicted cones, virtual cones, midpoints (PCD) or Delaunay edges (fallback). Also works in RViz |

All `pp_node_skidpad` parameters are kept. New parameters:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `pcd.enable` | `True` | `False` gives exactly the `pp_node_skidpad` behaviour |
| `pcd.color_mode` | `auto` | `auto` decides which colour is the left wall; `blue_left` / `blue_right` force it |
| `pcd.track_width_m` | 3.0 | Start value, or the fixed width when `measure_width` is off |
| `pcd.measure_width` | `True` | Measure the width from blue–yellow pairs (clipped to 2.5–6 m). Small Track measures 4.1–4.6 m |
| `pcd.use_memory` / `use_prediction` / `use_completion` | `True` | Switch the three PCD additions on or off |
| `pcd.smoothing` | 0.4 | Spline smoothing factor per point |
| `pcd.wall_margin_m` | 0.75 | Internal wall-clearance check on the smoothed path |
| `pcd.min_path_points` / `pcd.min_path_length_m` | 5 / 3.0 m | Shorter PCD paths go to the fallback |
| `pcd.cone_clearance_m` | 0.5 | Final cut against every real cone and wall segment (0 = off) |
| `pcd.defer_orange_count` | 4 | Hand over to the team logic when this many orange cones are in the sector (0 = never) |
| `debug.publish` | `True` | Publish `/pp/mode` and `/pp/debug_markers` |

### What happens on every tick

1. **Fallback gate.** With 4 or more orange or big orange cones in the camera sector (start line, skidpad lanes), or before the colour side is known, the original NORMAL/FALLBACK logic runs unchanged.
2. **Colour side.** The SLAM map in this stack has blue on the car's right. With `pcd.color_mode: auto`, the mean lateral offset of blue and yellow cones 0–15 m ahead decides the left wall. The decision latches after 5 agreeing ticks and flips only after 40 disagreeing ticks, which also resets PCD memory.
3. **Sector and memory.** Cones inside the planner sector (vertex 5 m behind the car, 30 m radius, 60°) are merged into a memory (0.6 m merge radius, forgotten after 40 m of driving without a sighting). Planning uses remembered cones from 4 m behind to 25 m ahead, so a cone that has left the sector still shapes the corridor.
4. **Walls.** Each colour is chained into an ordered wall by walking same-colour Delaunay edges forward from the car: an edge must be at most 6.5 m long and turn at most 70°. Where Delaunay offers nothing, a shorter (5.5 m) and straighter (40°) nearest-neighbour step is allowed. The chain stops when the turn changes by more than 45° from the previous step.
5. **Prediction.** If a wall leaves the sector in a bend, its last cones give a curvature and a curvature trend. One cone is placed along a clothoid continuation, with the bend capped at 45° and radii outside 3.5–40 m ignored. The guess is rejected if it would come within 0.6 w of the other wall, or if both walls are guessed to bend in opposite directions.
6. **Corridor completion.** Where a cone has no partner within 1.35 w on the other wall, a virtual cone is placed one track width away along the wall's inward normal. It is skipped in tight bends (|κ|·w ≥ 0.9), where the offset would land past the centre.
7. **Midpoints.** Real, predicted and virtual cones are triangulated again. Only blue–yellow edges between 0.55 w and 1.75 w long are kept. Each midpoint carries a weight: 1.0 for real–real, 0.6 for real–virtual, 0.3 when a predicted cone is involved.
8. **Ordering and smoothing.** Midpoints are ordered greedily from the car with a 7 m maximum hop and a 60° maximum turn per step. A weighted smoothing spline (factor 0.4 × points) is fitted and resampled at 0.5 m. If the spline comes within 0.75 m of a traced wall or virtual cone in the first 15 m, the unsmoothed chain is used instead, or, if that also fails, the path is cut before the problem.
9. **Final wall cut.** The path is cut at the first point closer than 0.5 m to any real cone or any same-colour Delaunay edge under 6.5 m, whether or not PCD traced that wall.
10. **Publish or hand over.** Paths with at least 5 points and 3 m are published. Otherwise the original logic runs for that tick. `/pp/mode` records which mode was used and why.

### Which problems each part addresses

| Problem | Seen as | Part that handles it |
| --- | --- | --- |
| Inside wall hidden in a bend | Short or kinked path at turn-in | Memory (3), prediction (5), virtual cones (6) |
| Only one wall visible | No blue–yellow edges at all | Virtual cones (6) |
| Delaunay edges across the track or to a far part of the track | Path jumping sideways | Wall chaining rules (4), edge length window (7), turn limit (8) |
| Smoothing cutting a corner into a wall | Path through cones | Spline wall check (8), final wall cut (9) |
| Map colours swapped relative to the car | Mirrored corridor | Colour latch (2) |
| Start line, skidpad lanes, orange-only scenes | PCD has no blue/yellow structure | Fallback gate (1) |
| Wrong 3 m width assumption on a 4.8 m track | Virtual cones inside the track | Width measurement (`pcd.measure_width`) |

### Run and compare

```bash
rams_abh && MPLBACKEND=Agg ros2 launch bringup stack_tls.launch.py   # PCD planner
rams_abh && MPLBACKEND=Agg ros2 launch bringup stack.launch.py       # team planner
rams_abh && python3 tools/birdview.py                                # viewer
```

`MPLBACKEND=Agg` turns off the planner's own Matplotlib window. That window costs about 50 ms per tick and held the planner to 7.5–10 Hz. Run one planner at a time.

## Next objectives

| Order | Objective | What to build or test | Evidence to collect |
| --- | --- | --- | --- |
| 1 | **Matched comparison with the team planner** · PP-01 | Run `stack.launch.py` and `stack_tls.launch.py` from fresh worlds with the same `V_MAX`, seed and GUI settings. Keep RViz and the planner plot closed. | Lap time, cross-track p95 and max, closest cone, contact count or unknown coverage, planner dropouts, pseudo-path feeds, recovery time. |
| 2 | **Reduce the turn-in cross-track spike** · PP-02 | Inspect `/pp/debug_markers` at the three Small Track bends where the spike repeats. Measure how far the published path moves between consecutive ticks at turn-in. Test a cap on path change per tick or a longer prediction once the inside wall is confirmed. | Path jump between ticks, cross-track error at turn-in, heading error, controller horizon in use. |
| 3 | **Planner rate** · PP-03 | Confirm 20 Hz with `MPLBACKEND=Agg`. Profile `_try_pcd` and `_cut_at_walls` on the laptop. | `/path_points` rate, tick time p50/p95 in wall and simulation time. |
| 4 | **Short and invalid messages** · PP-04 | The 19:38 run had 14 of 684 messages below the controller's 5-point minimum. Log the mode and reason for each. | Count, location, mode and reason of each short message. |
| 5 | **Skidpad mode** | Add a rulebook template fit (fixed figure-of-eight, three unknowns: position and rotation) as its own mode, used when the mission is skidpad. | Template fit residual, path error on the circles, lap count. |
| Later | **Robustness** | Test with simulated perception (noise, missed cones, colour errors) instead of ground truth. Repeat the matched comparison. | Off-track frames, dropouts, closest cone under each noise level. |

## Implemented changes

### `2efe7c6` · Add the PCD planner as its own node `pp_tls`

**Why.** The team planner joins only the cones it can currently see. In bends where the inside wall leaves the camera sector, its Delaunay triangles collapse and the path becomes short, kinked or missing. The aim was a planner that fills in the hidden part of the corridor, can be swapped in without touching controls, and falls back to the proven logic when unsure.

**What changed.** `pp_tls` runs PCD first on every tick and publishes its path when it is long enough and clear of the walls. Otherwise the original NORMAL/FALLBACK logic runs unchanged for that tick. The colour side is detected automatically, the track width is measured, and a final cut keeps the path 0.5 m from every real cone and wall segment. The node publishes `/pp/mode` and `/pp/debug_markers`. `stack_tls.launch.py` launches the full stack with `pp_tls`, and `tools/birdview.py` shows everything in one bird's-eye window. `pp_node_skidpad` is not modified.

**Files.** [pp_tls.py](src/pathplanning/pathplanning/pp_tls.py) holds the node and the embedded planner. [setup.py](src/pathplanning/setup.py) adds the entry point. [stack_tls.launch.py](bringup/launch/stack_tls.launch.py) launches it. [birdview.py](tools/birdview.py) is the viewer.

**Verification.** Offline, the node ran with stub ROS messages over the 9 real tracks of the StarkStrom FSD racetrack dataset (about 4,200 frames, blue on the right as in this stack). PCD planned about 99% of frames, with no path under 5 points (team planner: 77) and no frame where the path entered a wall with clean cones (team planner: 27). On Small Track in Gazebo, run `20261125-193037` used PCD in every controller cycle at a 5 m/s cap. Mean speed was 4.71 m/s, peak 5.10 m/s, lap time 23.4 s. Cross-track error had a median of 2.4 cm, p95 0.35 m and max 0.60 m. No cone contact was recorded, the closest cone centre was 1.24 m, and there were no pseudo-path feeds. Run `20261125-193810` at a 6.5 m/s cap gave mean 5.37 m/s, peak 6.60 m/s, lap time 20.8 s, the same cross-track error and a closest cone of 1.28 m; contact coverage was unknown in that run. The matched comparison with the team planner (PP-01) and the turn-in spike (PP-02) remain open.

### `41218b1` · Raise the Exp13 target-speed cap to 6.0 m/s (controls setting, recorded here for context)

**Why.** The planner tests are run above the team's 3 m/s baseline to see how the PCD path holds at speed.

**What changed.** `V_MAX` in `mppi_controller.py` is 6.0 m/s on this branch. No planner code changed.

**Files.** [mppi_controller.py](src/controls/controls/mppi_controller.py).

**Verification.** Not yet run at exactly 6.0 m/s. At 6.5 m/s the planner held the same cross-track error and cone distance as at 5 m/s. The controller overshot its own corner speed by up to 1.2 m/s, which is a controls item. Results at this cap are not comparable with the team's 3 m/s runs.

## Updating this file

For each planner change, add an entry under **Implemented changes**, newest first, with the commit, the run or problem that motivated it, the behaviour change, the affected files and the tests performed. Use the same four headings: **Why**, **What changed**, **Files**, **Verification**. Include what still needs a replay, GUI run or teammate check. If the change alters the algorithm, inputs, outputs or parameters, update **How the planner works** to match. Update the matching objective in **Next objectives** when its status changes, and update the date and summary at the top.
