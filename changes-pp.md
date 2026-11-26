# Path planning changes and next objectives

Branch `tel_soum_integration` · Ground-truth EUFS Small Track, comp_2021, garden_light and skidpad · Updated 26 November 2026

The branch has three planners, all publishing the same `/path_points` message, so the controller needs no change. **`pp_tls`** (v1) is the team planner `pp_node_skidpad` with **Predictive Corridor Delaunay (PCD)** tried first on every tick. **`pp_tels_new`** (v2, `stack_tels_new.launch.py`) adds route memory, hairpin-tip following, a curvature-limited line and the orange-gate fix; on Small Track (run `20261125-233516`) every message was a full PCD path, with no path jump over 1 m and no contact, lap 20.6 s at the 6 m/s cap (v1: 20.3 s at 6.5 m/s). **`pp_skidpad_tels`** (`stack_skidpad.launch.py`) fits the rule skidpad layout to the cones and publishes the exact figure of eight; its first Gazebo run locked the layout rotated, which is fixed and now locks only when the fit is well determined (re-test pending). `pp_node_skidpad` itself is unchanged and still launched by `stack.launch.py`. The bird view shows a steering-direction indicator. The hairpin failures on comp_2021 and garden_light are mainly controller limits (20° steering cap, corner speed, a 0.45–0.5 m steady offset on constant-radius turns), handed to controls with the racing-line study.

## How the planner works

### What it is for

The team Delaunay planner connects blue and yellow cones it can currently see. In a bend, one wall often leaves the camera sector before the other: the inside cones are hidden, the Delaunay triangles collapse, and the path is short, kinked or missing. PCD keeps the Delaunay idea and adds four things to it: a memory of cones already seen, walls traced as ordered chains, a prediction of where a wall continues into the blind part of a bend, and virtual partner cones where one wall is missing. Any scene PCD cannot handle well is handed back to the original team logic.

### Files

| File | Role |
| --- | --- |
| [pp_tls.py](src/pathplanning/pathplanning/pp_tls.py) | The node. Lines up to the `PCD PLANNER` banner are the embedded PCD planner (pure NumPy/SciPy, no ROS). The rest is `pp_node_skidpad` unchanged, plus `_try_pcd`, `_update_color_side`, `_cut_at_walls`, `_set_pcd_plot` and `_publish_debug`. |
| [pp_tels_new.py](src/pathplanning/pathplanning/pp_tels_new.py) | Version 2 of `pp_tls` as its own node `pp_tels_new`. The "v2" items in this section apply only to it. |
| [stack_tels_new.launch.py](bringup/launch/stack_tels_new.launch.py) | `stack_tls.launch.py` with `pp_tels_new` in place of `pp_tls`. |
| [pp_skidpad_tels.py](src/pathplanning/pathplanning/pp_skidpad_tels.py) | Skidpad-only planner `pp_skidpad_tels`: fits the rule layout (two circles) to the cones and publishes the exact reference. Same topics as `pp_tls`. |
| [stack_skidpad.launch.py](bringup/launch/stack_skidpad.launch.py) | `stack_tls.launch.py` with `pp_skidpad_tels` in place of `pp_tls`. |
| [pp_node_skidpad.py](src/pathplanning/pathplanning/pp_node_skidpad.py) | Team planner. Not edited. Its NORMAL/FALLBACK/big-orange logic is copied into `pp_tls` and used as the fallback. |
| [setup.py](src/pathplanning/setup.py) | Entry point `pp_tls = pathplanning.pp_tls:main`. |
| [stack_tls.launch.py](bringup/launch/stack_tls.launch.py) | `stack.launch.py` with `pp_tls` (node name `pp_tls`) in place of `pp_node_skidpad`. Same parameters, including `topics.odom_in: /slam/odom_raw`. |
| [birdview.py](tools/birdview.py) | Read-only bird's-eye viewer. Draws cones, car trail, `/path_points`, the controller's tracked path and rollouts, and the PCD internals from `/pp/debug_markers` and `/pp/mode`. A steering gauge shows commanded and applied front-wheel angle against the 20° controller cap and 29.8° EUFS lock; the car is drawn with its front wheels turned and the arc it would drive at the commanded steering. |

Dependencies: `rclpy`, `nav_msgs`, `geometry_msgs`, `eufs_msgs`, `std_msgs`, `visualization_msgs` (optional, debug topics are skipped without it), NumPy, SciPy (`Delaunay`, `cKDTree`, `splprep`), Matplotlib (optional live plot).

### Inputs and outputs

| Topic | Type | Direction | Use |
| --- | --- | --- | --- |
| `/slam/odom_raw` (param `topics.odom_in`) | `nav_msgs/Odometry` | in | Car position and heading in `map` |
| `/slam/map_cones` (param `topics.map_in`) | `eufs_msgs/ConeArrayWithCovariance` | in | Blue, yellow, orange, big orange and (v2) unknown-colour cones. The SLAM node publishes only cones near the car. On comp_2021 colours arrive only within about 15 m; farther cones are `unknown_color` |
| `/path_points` (param `topics.path_out`) | `nav_msgs/Path` | out | Frame `map`, car pose first, then the path at 0.5 m spacing. BEST_EFFORT. Timer at `publish.rate_hz` (20 Hz) |
| `/pp/mode` | `std_msgs/String` | out | JSON `{mode, reason, width, left, route, route_jump_m, confirmed_m, kappa_max, r_min_m, tight_for_20deg, tight_for_full_lock, unknown_used, min_curvature}` |
| `/pp/debug_markers` | `visualization_msgs/MarkerArray` | out | Walls, predicted cones, virtual cones, unknown-colour cones used (grey), midpoints (PCD) or Delaunay edges (fallback). Also works in RViz |

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
| `pcd.defer_orange_count` | 4 | Orange-scene threshold (0 = never hand over) |
| `pcd.orange_policy` | `validate` | v2. `validate`: with ≥ `defer_orange_count` orange cones, hand over only if fewer than 2 blue/yellow cones are in the sector. `defer`: v1 behaviour (always hand over) |
| `pcd.use_unknown` | `True` | v2. Unknown-colour cones may extend a wall (see step 4) |
| `pcd.min_curvature` | `True` | v2. Curvature-limited line inside the corridor (step 8) |
| `pcd.fit_through_car` | `False` | v2. `False`: the spline is fitted through the midpoints only and joined to the car by a straight piece. `True`: v1 behaviour |
| `route.enable` | `True` | v2. Route memory on every published path (step 11) |
| `route.max_jump_m` / `route.hold_ticks` / `route.switch_ticks` / `route.reset_jump_m` | 1.0 m / 10 / 3 / 5.0 m | Route memory limits |
| `gate.min_width_m` / `gate.max_width_m` | 2.5 / 6.5 m | v2. Big-orange gate: the two side centroids must be this far apart |
| `debug.publish` | `True` | Publish `/pp/mode` and `/pp/debug_markers` |

### What happens on every tick

1. **Fallback gate.** Before the colour side is known, the original NORMAL/FALLBACK logic runs. With 4 or more orange or big orange cones in the camera sector (start line, skidpad lanes) it also runs, but (v2, `orange_policy: validate`) only when fewer than 2 blue/yellow cones are in view; otherwise PCD plans through the gate. The colour side is updated before this check (v1 skipped the update on orange ticks). In the team logic, a big-orange gate is used only when big cones are present on both sides of the car, 2.5–6.5 m apart; its point is the midpoint of the two side centroids.
2. **Colour side.** The SLAM map in this stack has blue on the car's right. With `pcd.color_mode: auto`, the mean lateral offset of blue and yellow cones 0–15 m ahead decides the left wall. The decision latches after 5 agreeing ticks and flips only after 40 disagreeing ticks, which also resets PCD memory.
3. **Sector and memory.** Cones inside the planner sector (vertex 5 m behind the car, 30 m radius, 60°) are merged into a memory (0.6 m merge radius, forgotten after 40 m of driving without a sighting). Planning uses remembered cones from 4 m behind to 25 m ahead, so a cone that has left the sector still shapes the corridor.
4. **Walls.** Each colour is chained into an ordered wall by walking same-colour Delaunay edges forward from the car: an edge must be at most 6.5 m long and turn at most 70°. Where Delaunay offers nothing, a shorter (5.5 m) and straighter (40°) nearest-neighbour step is allowed. The chain stops when the turn changes by more than 45° from the previous step. v2: when both rules fail, a short step (≤ 2 m) may turn up to 110° in the direction the wall is already turning, if the cone is not back near the earlier part of the wall. This follows the inner tip of a hairpin (comp_2021: R ≈ 1.3 m, cones ≈ 1 m apart). v2: after the width is measured, each wall may be extended with unknown-colour cones (wider sector: 26 m, 170°): a step of 0.5–5.5 m, turn ≤ 40°, turn change ≤ 45°, 0.6–1.6 w from the other wall, at most 6 per wall, cheapest step first. Their midpoints weigh 0.8 with a real cone, 0.7 with another unknown cone.
5. **Prediction.** If a wall leaves the sector in a bend, its last cones give a curvature and a curvature trend. One cone is placed along a clothoid continuation, with the bend capped at 60° and the radius limited to 2.8–40 m (v1: 45°, 3.5 m). v2: a wall that already turns tighter than 1.25 / 2.8 m⁻¹ is not continued, because the capped, wider arc bulged into the lane at the comp_2021 hairpin tip. The guess is rejected if it would come within 0.6 w of the other wall, or if both walls are guessed to bend in opposite directions.
6. **Corridor completion.** Where a cone has no partner within 1.35 w on the other wall, a virtual cone is placed one track width away along the wall's inward normal. It is skipped in tight bends (|κ|·w ≥ 0.9), where the offset would land past the centre.
7. **Midpoints.** Real, predicted and virtual cones are triangulated again. Only blue–yellow edges between 0.55 w and 1.75 w long are kept. Each midpoint carries a weight: 1.0 for real–real, 0.6 for real–virtual, 0.3 when a predicted cone is involved.
8. **Ordering and smoothing.** Midpoints are ordered greedily from the car with a 7 m maximum hop and a 60° maximum turn per step. A weighted smoothing spline (factor 0.4 × points) is fitted through the midpoints (v2: not through the car) and resampled at 0.5 m; a straight piece joins the car to it. v2: if the path bends more than 0.12 m⁻¹ anywhere, lateral offsets along the normals are chosen by bounded least squares to minimise the second differences (≈ curvature), staying 1.0 m inside the walls and at most 2 m from the original line, with the car point fixed (after KA-RaceIng's online minimum-curvature line). If the line comes within 0.75 m of a traced wall or virtual cone in the first 15 m, the original fit is tried, then the unsmoothed chain, or the path is cut before the problem. v2: when the car itself is closer than 0.75 m to a wall, the required clearance starts at the car's own clearance and grows 0.5 m per metre, so a path leading away from the wall is allowed.
9. **Final wall cut.** The path is cut at the first point closer than 0.5 m to any real cone or any same-colour Delaunay edge under 6.5 m, whether or not PCD traced that wall. v2: the same ramp from the car's own clearance applies.
10. **Publish or hand over.** Paths with at least 5 points and 3 m are published. Otherwise the original logic runs for that tick. `/pp/mode` records which mode was used and why.
11. **Route memory (v2).** Every outgoing path (PCD or team logic) passes a filter. Team-logic paths are first resampled at 0.25 m and cut at real cones and wall segments, because their points are 2–6 m apart and a segment could jump a cone row. The new path replaces the previous route when it has ≥ 4 points and ≥ 3 m and its point 4 m ahead moved ≤ 1.0 m. Otherwise the previous route, trimmed to the car and re-cut against the cones, is kept for up to 10 ticks (0.5 s at 20 Hz). A different route is taken once it has persisted for 3 ticks, or when the hold runs out. A car jump > 5 m (simulator reset) clears the route and PCD's memory. `/pp/mode` reports the state (`new`, `follow`, `hold n/10`, `switch`, `no route`), the jump, the confirmed distance (path length backed by perceived cones), the published route's peak curvature and whether it is tighter than R 4.34 m (controller's 20° cap) or R 2.76 m (EUFS full lock).

### Which problems each part addresses

| Problem | Seen as | Part that handles it |
| --- | --- | --- |
| Inside wall hidden in a bend | Short or kinked path at turn-in | Memory (3), prediction (5), virtual cones (6) |
| Only one wall visible | No blue–yellow edges at all | Virtual cones (6) |
| Delaunay edges across the track or to a far part of the track | Path jumping sideways | Wall chaining rules (4), edge length window (7), turn limit (8) |
| Smoothing cutting a corner into a wall | Path through cones | Spline wall check (8), final wall cut (9) |
| Map colours swapped relative to the car | Mirrored corridor | Colour latch (2) |
| Start line, skidpad lanes, orange-only scenes | PCD has no blue/yellow structure | Fallback gate (1) |
| Hairpin inner tip tighter than the chaining limits | Wall stops at the tip, guessed arc bulges into the lane, path cut at the apex | Tip steps (4), no capped prediction (5) |
| Path changes shape between ticks, or the fallback jumps a cone row | Jump of the look-ahead point | Route memory (11) |
| Path bends through an off-line car or is tighter than needed | Curvature spikes at the start or apex | Fit without the car, curvature-limited line (8) |
| Wrong 3 m width assumption on a 4.8 m track | Virtual cones inside the track | Width measurement (`pcd.measure_width`) |

### Run and compare

```bash
rams_abh && MPLBACKEND=Agg ros2 launch bringup stack_tls.launch.py   # PCD planner v1
rams_abh && MPLBACKEND=Agg ros2 launch bringup stack_tels_new.launch.py   # PCD planner v2 (pp_tels_new)
rams_abh && ros2 launch bringup stack_skidpad.launch.py              # skidpad map only (pp_skidpad_tels)
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
| 4 | **Short and invalid messages** · PP-04 | Fixed in v2 and confirmed in Gazebo on Small Track (`20261125-233516`: 730/730 full PCD paths, no jump > 1 m). Still to do: log `/pp/mode` in the run recorder. | Jumps > 1 m of the 4 m look-ahead point, short messages with mode and reason. |
| 4a | **Verify v2 in Gazebo** | Small Track done (no regression, see the v2 entry). Still to run: comp_2021 and boa_constrictor with `stack_tels_new.launch.py`, same controller build as v1. Watch `route` and `tight_for_20deg` in `/pp/mode`. | Lap time, cross-track to the true centreline, contacts, `/pp/mode` states, hold/switch counts. |
| 4b | **Planning when the car is off the corridor** | comp_2021 replay and garden_light run (`20261125-234324`): once the car left the corridor, PCD found no path and the team logic aimed at other lanes; route memory switched to it after 3 ticks. Hold a PCD route against team paths until the hold expires, measure the hold in time instead of ticks, plan from the car's projection onto the last route, allow a larger first turn inside a U-turn. | Frames with no or cut paths after an excursion. |
| 5 | **Skidpad mode** | `pp_skidpad_tels`: first Gazebo run failed (rotated lock, fixed). Re-run on a fresh skidpad world; decide how the car should stop after the exit (planner or controller); radius as a parameter (8.625 m once the controller's offset is < 0.25 m). | Fit RMS and lock time from `/pp/mode`, cross-track on the circles, phase sequence, contacts. |
| 6 | **Racing line** | Global minimum-curvature line (TUM QP) for mapped laps and the skidpad, same `/path_points` format plus curvature and planned speed (`/pp/path_info`, to agree with controls). Modelled gain at today's limits: 9.4% Small Track, 15.8% boa_constrictor, 2.7% skidpad. Needs map closure first for trackdrive, and controller requirements R1–R4 (handoff package `racing_line_controls_handoff`). | Lap time vs the centreline on the same controller build, closest cone, contacts. |
| Later | **Robustness** | Test with simulated perception (noise, missed cones, colour errors) instead of ground truth. Repeat the matched comparison. | Off-track frames, dropouts, closest cone under each noise level. |

## Implemented changes

All entries down to `2efe7c6` arrive in one commit, the first after `6c9b47a` (26 November 2026). They are kept as separate entries because each solved a different problem; the commit's hash is the one on GitHub after `6c9b47a`.

### 26 Nov 2026 · Skidpad planner: lock only when the layout is determined; soft colour term

**Why.** Review of `pp_skidpad_tels`: (1) the orange lane half width (1.6 m) did not match the 3 m track of the circles; (2) 12 inliers with low RMS can all sit on one arc, so the lock did not prove the pose was determined; (3) colour was thrown away; (4) observability of the three fitted numbers was not checked.

**What changed.**
- `lane_half_width` 1.5 m (rule lane width 3 m; EUFS puts its orange cones at 1.64 m, which the robust loss absorbs); the timing-line offset 1.78 m is now a parameter.
- Lock requires all of: ≥ 12 inliers, RMS ≤ 0.25 m, ≥ 4 inliers on each circle, ≥ 60° angular coverage on each circle, 1-sigma pose uncertainty from rms² (JᵀJ)⁻¹ (numerical Jacobian of the inlier residuals) ≤ 0.05 m and ≤ 0.3°, and 3 consecutive fits agreeing within 0.05 m / 0.3°. `/pp/mode` lists the checks still failing, the uncertainty and the coverage.
- Before locking, a fit that agrees with the car and has ≤ 0.15 m / 1° uncertainty is used provisionally (refitted every tick, search centred on it); with no such fit the path is straight along the car's heading.
- Colour: memory keeps blue/yellow votes per cone. Once the orientation is fixed by the car, the colour convention is read off the fit (normal, or swapped as in some SLAM maps, ≥ 80% agreement) and used as a soft term: a boundary of the wrong colour costs 0.6 m. `skidpad.colour_mode` auto / normal / swapped / off.
- The car-agreement test, before any progress, needs the car in the entry lane; afterwards it needs the car within 3 m of the reference near its arc length and heading along it.

**Files.** [pp_skidpad_tels.py](src/pathplanning/pathplanning/pp_skidpad_tels.py), [birdview.py](tools/birdview.py) (shows layout state, uncertainty, coverage, colour).

**Verification.** Offline.
- Earlier simulated cases unchanged (layout exact or within 0.09 m, all six phases); the recorded run `20261126-010253` replay and the wrong-first-heading cases lock the correct layout after the 3-fit stability window.
- Cones from the first circle only (30 cones): old rule would lock with a 1.23 m error; now not locked (one circle, rotation unobservable, σ infinite). Both circles but only ±35° far-side arcs (12 cones): old rule would lock; now waits for coverage.
- Short view (10 m / ±30°, 8 m / ±25°, 12 m / ±40° with noise): provisional layout used on the entry, locked at 3.9 s, 10.0 s and 8.8 s, all six phases, car within 0.32 m of the true reference. Before the provisional fix the 10 m case drove straight past the crossing.
- Swapped colours: detected as swapped, layout exact.
- Basin of the local fit from random starts (±6 m, ±90°): 186/200 without colour, 191/200 with the soft colour term. Colour does not resolve the 180° copy in a local fit (92 vs 96 of 200 at ±180°); the car-agreement test does that.

### 26 Nov 2026 · Skidpad planner: fix the rotated layout lock

**Why.** First Gazebo run on the EUFS skidpad (`20261126-010253`) went badly: from the first tick the published path ran from the car at (0, 0) straight to (15.2, 4.9) and then round a circle centred near (25, 5.5), so the car swerved right in the entry lane, hit orange cones 62 and 63, blue 28 and big orange 81 (25 contact episodes), and then circled one wrong circle. Replaying the recorded cones offline gives the correct layout (crossing at (14.41, 0.00), 0.00°, RMS 0.095 m), so the cones were fine. The layout had locked rotated. The figure of eight looks the same after a 180° turn, and the start-pose prior was taken from the very first odometry message only; with a wrong first heading (reproduced offline with 90° and 180°) the fit locks onto a rotated copy with exactly the same RMS.

**What changed.** The prior now follows the car until the layout locks. Both the fitted orientation and the 180° copy are checked, and the fit is accepted only if the car is in the entry lane (behind the crossing, within 3 m of the line) and heading along the entry direction within 35°; otherwise the planner keeps waiting. Until the layout locks it publishes a straight path along the car's heading (the entry lane), and `/pp/mode` says why it is waiting. The arc-length search starts only after the lock.

**Files.** [pp_skidpad_tels.py](src/pathplanning/pathplanning/pp_skidpad_tels.py).

**Verification.** Offline replay of the run's recorded map messages, with the first odometry message forced to headings 0°, 90°, 180° and other wrong poses: correct layout every time (crossing at (14.41, 0.00), 0.00°), path starting along the entry line. The simulated runs of the first version give the same results as before. In the bad run the car also sat a steady 0.46 m outside the circle with constant steering (−0.13 rad); that offset on a constant radius is on the controller side.

### 26 Nov 2026 · Skidpad template planner `pp_skidpad_tels`

**Why.** The skidpad layout is fixed by the rules, so planning it cone by cone (and resolving the crossing where both circles overlap) is unnecessary work and a source of error. Objective 5.

**What changed.** New node `pp_skidpad_tels`, used instead of `pp_tls` / `pp_tels_new` on the skidpad. It keeps every cone seen, fits three numbers (layout position and rotation) so that blue, yellow and unknown cones lie on the four rule circles (inner R 7.625 m, outer R 10.625 m, centres 18.25 m apart) and orange cones on the lane lines, with a robust (soft L1) least-squares fit started from a grid around a prior (crossing 15 m ahead of the start pose, searched ±6 m, ±2 m, ±12°). The fit locks at ≥ 12 cones on the circles and RMS ≤ 0.25 m; afterwards only consistent refinements are blended in. The reference is the exact rule geometry: entry line, two laps of the right circle (clockwise), two of the left (counter-clockwise), exit line, all R 9.125 m and tangent at the crossing. The car's progress along it only moves forward and is searched in a −1 / +6 m window, so the five passes through the crossing cannot be confused. Output is the usual `/path_points` (car pose, then 20 m at 0.5 m); `/pp/mode` gives phase, lap, fit RMS, cones used and lock state; the fitted circles are drawn in the bird view. Colour is not used, so swapped colours in the SLAM map do not matter. All rule dimensions, the lap count and the first side are parameters. `birdview.py` shows the skidpad fields in its panel.

**Files.** [pp_skidpad_tels.py](src/pathplanning/pathplanning/pp_skidpad_tels.py), [setup.py](src/pathplanning/setup.py), [stack_skidpad.launch.py](bringup/launch/stack_skidpad.launch.py), [birdview.py](tools/birdview.py).

**Verification.** Offline only, on the EUFS `skidpad.csv` layout with simulated perception (cones within 20 m and ±60°) and a pure-pursuit car at 5 m/s with a 20° steering cap:
- clean cones: layout found exactly (EUFS cones sit about 0.1 m outside the rule radii, RMS 0.085 m), locked on the first tick, phases entry → right 1 → right 2 → left 1 → left 2 → exit, car within 0.12 m (p95) of the true reference;
- 0.1 m cone noise and 20% of cones missing: layout error 6 mm and 0.08°;
- 0.15 m noise, 30% missing, 30 false cones: layout error 0.08 m and 0.03°, reference within 0.09 m of the truth;
- start pose 5 m short or long, 1.5 m to the side or 10° off: layout still found exactly.
- The ROS node was run with stub messages (path, `/pp/mode`, markers published). Not yet run in Gazebo. After the exit the path shortens to nothing; how the car stops there is not handled by the planner.

### 26 Nov 2026 · Steering indicator in the bird view

**Why.** Requested by a senior: see the steering direction and angle while watching the bird view, to judge corners such as the garden_light and comp_2021 hairpins where the controller sat at its 20° cap.

**What changed.** `tools/birdview.py` subscribes to `/ground_truth/wheel_speeds` (applied wheel angle) in addition to `/cmd`. Under the status panel a dial shows the commanded angle (orange needle) and the applied angle (white needle), with zones for inside the controller cap (green), cap to EUFS lock (amber) and beyond the lock (red), and a large LEFT/RIGHT readout with the angle and the turning radius R = L / tan(δ). On the map the car's front wheels are drawn at the applied angle and a dashed orange arc shows the 8 m the car would drive at the commanded steering (key `w` toggles it). The viewer still only subscribes.

**Files.** [birdview.py](tools/birdview.py).

**Verification.** Rendered offline with a stubbed node and data from run `20261125-234324` for left (18.4°) and right (−6°) steering; needles, readout, wheels and arc point the right way. Not yet opened live in WSL.

### 26 Nov 2026 · New node `pp_tels_new` (v2): route memory, hairpin tips, curvature-limited line, orange-gate fix (PP-04 to PP-13)

**Why.** Two sources. (1) Soumil's PP-04 review of Small Track run `20261125-203828`: four jumps of more than 1 m at the orange gate. At frame 2929 the team logic took the midpoint of two big orange cones on the same side while the colour side was still unknown; at frames 3334–3336 the orange-count rule handed over to the team FALLBACK with 4 blue and 0 yellow cones in view. (2) The comp_2021 hairpin run `20261125-203532` (PP-05 to PP-13 in the comp_2021 issue package): no route memory, fallback at the apex, a hook where the spline bent through an off-line car, prediction that could not describe a hairpin, and no record of the planner mode. Ideas taken from KA-RaceIng 2022 (online minimum-curvature line, arXiv 2210.10933), AMZ 2020 (keep uncertain cones, arXiv 2003.05266), StarkStrom 2024 (warm start from the previous lane) and Soumil's advice to fix lane continuity before extending prediction.

**What changed.** The PCD pipeline is the same; v2 adds to it. Orange scenes hand over only without blue/yellow structure, and the colour side updates first. Big-orange gates need cones on both sides. Walls follow hairpin tips with short sharp steps and may be extended by unknown-colour cones. Prediction allows R 2.8 m and 60° but skips walls already tighter than that. The spline is fitted without the car point, then a bounded least-squares curvature-limited line is applied and kept only if it passes the wall check. The wall checks allow a car close to a wall to steer away. A route-memory filter sits in front of the publisher for both PCD and team paths. `/pp/mode` carries the route state, confirmed distance, curvature and feasibility flags; unknown cones used are drawn grey. Every addition has a parameter to switch it off (see the parameter table).

**Files.** [pp_tels_new.py](src/pathplanning/pathplanning/pp_tels_new.py) (copy of `pp_tls.py` with the v2 changes; `pp_tls.py` unchanged), [setup.py](src/pathplanning/setup.py) adds the entry point `pp_tels_new`, [stack_tels_new.launch.py](bringup/launch/stack_tels_new.launch.py) launches it.

**Verification.** First Gazebo run: Small Track, run `20261125-233516` (V_MAX 6.0, Exp13 seed 42). All 730 planner messages were full PCD paths (minimum 18 points), including the orange start gate; no jump > 1 m of the 4 m look-ahead point; largest gap in simulation time 0.13 s. Lap 20.6 s, mean 5.17 m/s, peak 6.26 m/s, controller cross-track p95 0.33 m and max 0.60 m, no cone contact, closest cone centre 1.17 m. The nearest v1 run (`20261125-193810`, 6.5 m/s cap) gave lap 20.3 s, cross-track p95 0.35 m and max 0.58 m, closest cone 0.96 m (same measure). The turn-in cross-track spike (PP-02) is still there at the same bends (13 of 229 cycles above 0.3 m vs 20 of 315). comp_2021 not yet run with v2.

Offline:
- Soumil's replay harness on run `20261125-203828` (478 moving frames): jumps > 1 m 4 → 0, usable paths 477 → 478, PCD on all 478 frames (v1: 433 PCD, 43 NORMAL, 2 FALLBACK). The path crosses the orange gate near its centre on both passes. Peak path curvature p95 0.97 → 0.27 m⁻¹. Against v1 the path moved by a median of 0.20 m (2–10 m ahead); the larger differences are the frames where v1 used the team logic.
- comp_2021 open-loop replay of all 3,631 recorded map messages (with unknown-colour cones), same car poses: before 255 s (car in the corridor) mean offset to the true centreline 0.30 → 0.27 m, peak curvature p95 1.54 → 0.62 m⁻¹, PCD share 99% in both. Approaching the left hairpin (225–256 s) the path went all the way round into lane E in 242 of 278 frames (v1: 165), first at 229.0 s (v1: 231.3 s). v1 was cutting the path at the apex because its guessed inner-wall arc bulged into the lane. After the car reached the inner cone row (≥ 256 s, open loop, so the car keeps its recorded wrong course) team-logic paths that crossed cone rows towards lane D are now cut at the first row.
- 9 real tracks of the StarkStrom dataset (about 4,200 frames): off-track frames 27 → 26 on track 4 and 24 → 22 on track 9, 0 elsewhere as before; no short paths.
- Unknown-colour cones were used little on comp_2021 (0.4 cones per frame on average): colours were already available for the hairpin when it mattered. Their benefit on a sparser colour range is untested.
- Not covered: skidpad or other orange-heavy layouts, perception noise, closed-loop behaviour of the route hold.

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
