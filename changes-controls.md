# Controls changes and next objectives

Branch `abh_soum_integration` · Ground-truth EUFS Small Track · Updated 27 November 2026

The active controller is the mirror-symmetric Exp13 Stage 1 Neural ODE with MPPI. The current target-speed cap is 3 m/s. The long-term target is **at least 7 m/s average over a complete, repeatable lap**. The latest local drive averaged 2.60 m/s from first motion to the last command and 2.62 m/s gate to gate. The return still overlapped two surveyed cone positions in the chassis projection, and the contact recorder had no payload.

## Next objectives

### Corner braking and speed preview

This branch uses a 3 m/s maximum. Running `ros2 run controls control_node` defaults to Exp13 with seed 42, corner braking, and corner speed preview. For an A/B run, prefix that command with `MPPI_CORNER_PREVIEW=0` to test corner braking alone. `MPPI_CORNER_BRAKING=0` also disables the braking change. Both switches can be overridden independently. Exp10 defaults to both switches off. In `src/controls/controls/mppi_controller.py`, the braking profile now accepts curvature above 0.3 per metre when the path tangent over a wheelbase supports the same bend. The preview takes the lowest speed ceiling within the distance travelled in one proportional speed-loop time constant (0.67 s). The steering cap remains 0.349 rad and the physical minimum turning radius is approximately 4.3 m. A requested radius below that still needs a feasible line. `src/controls/controls/run_session.py` records the switch, and `src/controls/test/test_corner_braking.py` covers both turn directions, the intervening apex, stopping, and the disabled baseline.

On the longer Boa recording, the ceiling fell by over 0.25 m/s on 150 of 418 moving tracking samples. In the comp_2021 left-hairpin window it fell on all 55 samples by a median 1.50 m/s using the published planner path as a proxy for the controller buffer. These offline results measure speed ceilings. Verify cone contacts, gate time, actual entry speed and overslowing in matched GUI runs.

Telson's separate 26 November Boa run recorded 16 native contact episodes involving 12 distinct cones. The first right-front-wheel hit occurred during tracking at 3.93 m/s against a 3.93 m/s raw target. His moving skidpad run recorded 34 episodes involving 21 cones, including repeated blue-side contacts at roughly 0.48 m cross-track error. Inspect the swept wheel and body clearance of both the reference and the chosen rollout at the first contact. The second skidpad recording stayed stationary with 825 one-point paths.

**Telson comparison runs.** The 6.0 m/s Small Track run `20261125-203828-oelkohd_` completed a 20.71 s lap averaging 5.22 m/s. The later run `20261125-224833-u5hfpg30` verified CTRL-19 in Gazebo. The CTRL-20 corner braking change passed fixed-input replay and is ready for a matched GUI comparison. Then test slower and braking alternatives with the cone-clearance check before evaluating Shield-MPPI. The table below records the earlier shared-branch plan.

| Order | Objective | What to build or test | Evidence to collect |
| --- | --- | --- | --- |
| 1 | **Check the whole car against cones** · CTRL-04 | Implemented as an opt-in Small Track experiment. The recorded geometry passed offline verification. Compare enabled and disabled runs in fresh GUI sessions. This is the geometric first step toward [Shield-MPPI](https://arxiv.org/abs/2302.11719). | Check clearance through the two return-gate scenarios, feasible candidate counts, false stops, physical braking and p95 solve overhead. |
| 2 | **Handle stale state, late solves and resets** · CTRL-01, 06, 09, 18 | Represent missing speed explicitly. Reject stale commands before publish, and brake if the queued steering fills the active prediction horizon. Verify pose jumps and clock resets clear old plan, path and command history. Check the CUDA worker on the RTX 5050. | Replay slow CPU conditions, delayed feedback and resets. Confirm stale acceleration is suppressed and measure actual brake onset, stopping distance and recovery. Record decision-to-publish p50/p95 in simulation time. |
| 3 | **Check planner recovery and command resumption** · CTRL-02, 03, 04 | Check the first returning planner route for forward progress and clearance. Preserve conservative speed while clearance or usable lookahead is poor. Send the planner team the frame IDs and geometry for gaps, discontinuities and unsafe routes. Planner algorithm edits are owned by that team. | Replay the 4.4 s gap and subsequent gate return from 15:25. Compare clearance, time spent near steering lock, time to resume tracking, stop rate and finish rate. |
| 4 | **Finish state and steering timing** · CTRL-08, 09, 13, 16 | Align measured lateral velocity and yaw rate with the pose. Verify the slip convention and compare Exp13 predictions with measured motion before changing the state seed. Anchor warm-start reuse to the actual previous publication and test the backward-shift case. Keep measured wheel sync enabled for the warm-start A/B. | Compare short-horizon yaw and pose prediction errors, physical yaw reversals and lateral error on the opening, corner turn-in, and solve latency. The latest run had 12 backward shifts and fewer command movements but more yaw reversals than the prior drive. |
| 5 | **Compare smoother MPPI sampling** | Audit the existing cost, temperature and sampling correction against [Williams et al.](https://arxiv.org/abs/1707.02342). Test [LP-MPPI](https://arxiv.org/html/2503.11717v3) first by filtering sample noise before rollouts. Then test [Smooth MPPI](https://arxiv.org/abs/2112.09988) by sampling steering rate. Change one option at a time with the same rollout budget. | Measure requested and applied steering movement, yaw reversals, lateral drift, turn-in, clearance and p95 solve time. Repeat fresh-world GUI trials with matched seed schedules. |
| 6 | **Establish a repeatable clean-lap score** | Validate contact coverage, add oriented gate-to-gate lap scoring and flag resets or departures. Start collecting this evidence during objective 1. Use the Small Track GUI for comparable runs. | Report complete laps, contact count or unknown coverage, minimum physical clearance, mean and peak speed, recovery time and run-to-run variation at the 3 m/s target cap. |
| 7 | **Increase pace toward 7 m/s average** | Calibrate braking, lateral grip, curvature and visible path length. Build a speed envelope from those measurements and add bounded progress only within the checked corridor. Test higher caps in steps once the clean-lap baseline holds. [Liniger et al.](https://arxiv.org/abs/1711.07300) · [Stanford tire-limit speed control](https://ddl.stanford.edu/publications/conference-proceeding/speed-control-robust-path-tracking-automated-vehicles-tire-road) | Score gate-to-gate speed alongside contact coverage, clearance, finish rate and braking margin. |
| Later | **Test robustness and maintain diagnostics** · CTRL-05, 07, 10–12 | Use measured model error to decide whether [Robust MPPI](https://arxiv.org/abs/2102.09027) or [Tube-MPPI](https://arxiv.org/abs/2110.07744) adds value. Repair the legacy smoke test and topic diagnostic. Track active legacy callers before changing old buffer and tangent helpers. Keep full shutdown logs for any distinct exit traceback. | Compare under measured disturbances. Confirm the smoke test runs with the active interface and the diagnostic shows the actual pose and speed topics. |

## Implemented changes

### Use PP as the sole path source for active MPPI

The path planner owns the one-sided-cone fallback. The active controller now follows usable PP references, including PP-generated one-sided paths. If there is no usable planner reference, it holds the last steering request, sets a zero speed target, and requests 1.2 m/s² of deceleration. The existing heading guard still rejects badly aligned references when cones are one-sided or absent.

`control_node.py` no longer tries a cone-wall fit, publishes a synthetic path, or applies its 1.5 m/s bridge cap and handback ramp. `ros_connect.py` and `path_reference.py` remove the wall-fit code and recovery path storage. The normal planner freshness and forward-distance checks still apply. Old run reports can still display recorded pseudo-path events.

### CTRL-20 · Apply a lower corner-entry speed target immediately

**Why.** In Boa Constrictor run `20261125-231353-me0ekac4`, the raw target fell from 5.99 to 2.88 m/s at 1.598 s after first motion. The issued target was still 5.53 m/s and the controller asked for +1.31 m/s². Its first braking request came 205 ms later. The planner path was available.

**What changed.** With `MPPI_CORNER_BRAKING=1`, the controller applies the computed speed ceiling immediately when it falls. The target can rise by up to 3.0 m/s per simulation second, based on the existing acceleration limit. While speed exceeds the target, stored positive PI throttle cannot reduce the proportional braking request. The issued target still caps the rollout speed profile. The clearance check and PP recovery retain their existing interfaces. Use `MPPI_CORNER_BRAKING=0` for the original speed filter and PI behavior.

**Files.** [mppi_controller.py](src/controls/controls/mppi_controller.py) applies the target and PI changes. [run_session.py](src/controls/controls/run_session.py) records the toggle. [test_corner_braking.py](src/controls/test/test_corner_braking.py) covers the target step, throttle integral, rebound and recovery cap.

**Verification.** The opt-out replay reproduced the original recorded speed targets and acceleration commands exactly on both Boa runs and the Small Track shutdown run. With the new policy enabled, the worst Boa entrance sample changes from +1.31 to −2.66 m/s² on the same inputs. On the longer Boa recording, 13 cycles of positive acceleration more than 0.25 m/s above the raw target dropped to zero. Tracking requests at or below −2.99 m/s² rose from 3 to 16. The replay holds speed and raw targets fixed, so this counts requests under recorded inputs. Twenty-six targeted controls tests passed, and the rebuilt Exp13 checkpoint smoke test verified the issued and rollout targets. Verify actual corner-entry speed, time spent braking, steering, clearance and lap time in fresh matched GUI runs.

### CTRL-19 · Brake before controller shutdown in EUFS

**Why.** The last tracking command in the Telson Small Track run requested positive acceleration. After the command, the car traveled another 14.31 m in 2.69 simulation seconds and was still moving at 4.04 m/s at the end of recording.

**What changed.** On normal Ctrl+C or SIGTERM, the controller keeps its ROS interface active, discards the next tracking command if a solve was in progress, and requests zero target speed and `-3.0 m/s²` while holding the last published steering angle. It confirms standstill from advancing ground-truth planar speed samples no faster than 0.05 m/s for 0.20 simulation seconds after the steering actuator delay. It reports an unconfirmed stop if the simulator resets, ROS closes, or the 60 s wall-time bound expires. The recorder stays active through braking and plotting.

**Files.** [control_node.py](src/controls/controls/control_node.py) handles signals and brake commands. [shutdown_brake.py](src/controls/controls/shutdown_brake.py) verifies the stop with stamped planar feedback from [ros_connect.py](src/controls/controls/ros_connect.py). [run_session.py](src/controls/controls/run_session.py) keeps the recorder active and saves `shutdown.json`. [report.py](src/controls/controls/run_audit/report.py) and [dashboard.html](src/controls/controls/run_audit/dashboard.html) display standstill and recorded travel. [Regression tests](src/controls/test/test_shutdown_brake.py) cover standstill, feedback gaps, duplicates, pauses and resets.

**Verification.** Twenty controls regression tests passed in Galactic WSL. In an isolated ROS domain, the installed Exp13 node drove the normal `ros2 run` entry point, received SIGINT after three tracking commands, published braking, and saved an HTML report and trajectory plot. Scripted GT fell from 2.00 m/s to rest and the report computed 1.06 s and 1.06 m from the first brake request to confirmation. Measure Gazebo brake onset and stopping distance in a GUI run with the same planner, seed and speed cap as the Telson baseline.

**Gazebo verification.** Small Track run `20261125-224833-u5hfpg30` used Telson PP, Exp13 seed 42, a 6.0 m/s cap and clearance off. Shutdown began at 5.02 m/s on the left-side turn. Ground-truth speed had fallen by 0.10 m/s after 257 ms, reached zero after 1.85 simulation seconds, and standstill was confirmed at 2.03 s. Full-rate GT shows approximately 5.17 m of travel after the first brake request. The 30 shutdown commands requested −3.0 m/s², with no positive acceleration command during shutdown. Native cone contact coverage was unavailable. This verifies the normal stop in that run.

### Run report can be read in script-blocking previews

**Why.** A report could appear blank when opened through a preview that blocks scripts. The original file rendered in Chrome. **What changed.** [report.py](src/controls/controls/run_audit/report.py) now generates the key statistics and working export links as HTML. [dashboard.html](src/controls/controls/run_audit/dashboard.html) displays the summary until the interactive view has loaded. **Verification.** Chrome rendered six charts and the final shutdown frame from the regenerated run. With scripts disabled, it displayed a readable stop summary and export links.

### CTRL-04 · Check swept cone clearance in the Exp13 rollouts

**Why.** Both recorded orange-gate returns had projected chassis overlap near the blue-side cones even with a small reported cross-track error. Clearance depends on the complete footprint and heading along the predicted trajectory.

**What changed.** `MPPI_CONE_CLEARANCE=1` uses recent map-frame cones, including orange and unknown classes. The footprint is five conservative boxes covering the chassis and four tyres. Cones are discs with approximate collision-mesh radii and a 0.10 m margin. The check covers predicted rollout motion between geometry samples. MPPI averages the feasible candidates and independently re-rolls the weighted sequence with its usual dynamics and actuator model. If that sequence fails, MPPI chooses the lowest-cost feasible sampled sequence. If the map is missing or stale, or all candidates fail, the node holds the last issued steering and requests −3.0 m/s² and zero target speed. This fallback is an explicit braking request. Stopping distance still depends on vehicle motion and the command queue. The experiment is off by default.

**Files.** [cone_clearance.py](src/controls/controls/cone_clearance.py) defines geometry and swept clearance. [ros_connect.py](src/controls/controls/ros_connect.py) snapshots stamped map cones. [mppi_controller.py](src/controls/controls/mppi_controller.py) screens candidates and re-rolls the weighted sequence. [control_node.py](src/controls/controls/control_node.py) publishes the fallback and its audit mode. [run_session.py](src/controls/controls/run_session.py), [report.py](src/controls/controls/run_audit/report.py), and [dashboard.html](src/controls/controls/run_audit/dashboard.html) expose the setting, candidate counts, margins, decisions and timing. [Geometry tests](src/controls/test/test_cone_clearance.py) and [INTEGRATION.md](INTEGRATION.md) cover verification and launch.

**Verification.** On stamped ground-truth poses and surveyed orange cone positions, the 15:02 first pass and return scored +1.03 m and −0.99 m. The 15:25 pass and return scored +1.00 m and −0.84 m. These are padded footprint margins from the geometry replay. All 14 targeted tests passed. Offline Exp13 CUDA tests on the RTX 4050 exercised both horizons, braking with missing or blocked cone data, and the fallback with an injected rejection of the weighted sequence. The warmed 200-step synthetic case took roughly 59 ms median for clearance and selection. The first solve was slower. Live cone visibility, false stops, overhead and braking distance still need GUI verification.

### `b1645cc` and `038ca0e` · Separate CUDA worker and team runbook

**Why.** The teammate's Ubuntu/ROS Galactic venv uses Python 3.8 and CPU-only PyTorch. In the recorded CPU run, the median solve consumed 0.93 simulation seconds, 188 of 238 solves had no actionable steering horizon, and ground truth reached 7.20 m/s against a 3.00 m/s target.

**What changed.** The opt-in `MPPI_WORKER_PYTHON` path runs the existing solver and checkpoint in another interpreter over a local socket. ROS and recording stay in the Galactic interpreter. The worker checks CUDA with a real tensor operation and the run manifest records the worker path. The later documentation commit adds the build, CUDA setup and three-terminal Small Track launch sequence.

**Files.** [mppi_worker.py](src/controls/controls/mppi_worker.py) manages the solver subprocess and request/reply exchange. [control_node.py](src/controls/controls/control_node.py) selects and cleans up the worker. [run_session.py](src/controls/controls/run_session.py) records the interpreter. [INTEGRATION.md](INTEGRATION.md) gives setup and launch commands.

**Verification.** On the RTX 4050, both CUDA horizons captured and six offline commands matched the in-process solver. A clean checkout built all five packages and exposed both launch files, the controller and the packaged checkpoint. The RTX 5050 smoke test and GUI run are pending. The stale-solve guard remains open under CTRL-18.

### `c970cc7` · Match wheel and pose timestamps, and shift the previous plan by elapsed time

**Why.** The pose and wheel stamps often arrived ahead of the controller's local simulator clock. In the first actuator-enabled drive, only 70 of 995 moving tracking cycles used measured wheel angle. MPPI also advanced the previous plan by one 5 ms step despite roughly 40–50 ms between commands.

**What changed.** The controller selects the latest valid wheel sample at or before the stamped pose, with age and reset handling. MPPI resamples the previous scored sequence on the new action timeline, taking the first action hold and active horizon into account. `MPPI_STEERING_SYNC` and `MPPI_ELAPSED_WARM_START` can be toggled independently.

**Files.** [ros_connect.py](src/controls/controls/ros_connect.py) keeps stamped pose and wheel samples. [steering_actuator.py](src/controls/controls/steering_actuator.py) aligns the actuator timeline. [control_node.py](src/controls/controls/control_node.py) passes the sample and timestamps. [mppi_controller.py](src/controls/controls/mppi_controller.py) implements plan reuse. [run_session.py](src/controls/controls/run_session.py) and [run_audit/report.py](src/controls/controls/run_audit/report.py) record the switches and timing. [Actuator tests](src/controls/test/test_steering_actuator.py), [warm-start tests](src/controls/test/test_steering_warm_start.py), and [INTEGRATION.md](INTEGRATION.md) cover verification and launch.

**Verification.** The 15:25 GUI run used measured wheels in all 1,046 solves. Median plan advance was 50 ms. The combined change reduced cumulative requested steering movement on the opening from 125° to 55°, while measured yaw reversals rose from one to two and sideways error span rose from 0.33 m to 0.49 m. The runs started with different wheel angles. A matched warm-start A/B and the backward-shift fix remain open.

### `fb7226b` · Model the Gazebo steering actuator

**Why.** Rollouts previously applied requested steering immediately. EUFS queues commands for 0.20 simulation seconds and the dry DynamicBicycle wheels travel at up to 1.04 rad/s.

**What changed.** Rollouts start from measured wheel angle or a history-based estimate, replay already queued commands, account for estimated solve latency, and apply the steering-rate limit. Commands carry simulation timestamps. `MPPI_STEERING_ACTUATOR` selects the actuator model, and the recorder captures applied steering.

**Files.** [steering_actuator.py](src/controls/controls/steering_actuator.py), [mppi_controller.py](src/controls/controls/mppi_controller.py), [control_node.py](src/controls/controls/control_node.py), and [ros_connect.py](src/controls/controls/ros_connect.py) implement the model and command history. [run_session.py](src/controls/controls/run_session.py), [capture.py](src/controls/controls/run_audit/capture.py), [report.py](src/controls/controls/run_audit/report.py), and [dashboard.html](src/controls/controls/run_audit/dashboard.html) expose requested and applied steering. [Actuator tests](src/controls/test/test_steering_actuator.py) and [INTEGRATION.md](INTEGRATION.md) document the checks and toggle.

**Verification.** CPU, CUDA and isolated ROS tests passed. At roughly the same opening speed, the first actuator-on GUI drive had one substantial yaw reversal versus three before, and its lateral error span fell from 0.68 m to 0.33 m. Cumulative requested steering movement rose from 76° to 125°. This was a single observed comparison.

### `3c981dd` · Repeat the random sample sequence

**Why.** Unrecorded random sampling made comparisons hard to repeat.

**What changed.** Exp13 uses a dedicated generator on its compute device. `MPPI_SEED` defaults to 42 and the run manifest records it.

**Files.** [random_seed.py](src/controls/controls/random_seed.py) validates the seed, [mppi_controller.py](src/controls/controls/mppi_controller.py) uses it for both steering draws, [run_session.py](src/controls/controls/run_session.py) records it, and [INTEGRATION.md](INTEGRATION.md) documents comparable runs.

**Verification.** Matching seed and inputs produced identical offline samples and commands. Different seeds changed samples. Live ROS arrivals and simulator timing can still change a drive.

### `441c0b8` and `47dd07a` · Preserve the planner route and fix a recovery-log crash

**Why.** The controls buffer could mix waypoints across updates, change the route through a gate and retain old geometry. After the refactor, a leftover variable in a recovery log caused a `NameError`.

**What changed.** Active Exp13 consumes an ordered planner message with a 0.5 s freshness bound and separately tracks recovery geometry with a 1.5 s bound. It keys the spline cache on all points and projects progress using segments. Telemetry records which path source was used. The follow-up fixes the recovery log's stale variable.

**Files.** [path_reference.py](src/controls/controls/path_reference.py), [ros_connect.py](src/controls/controls/ros_connect.py), [control_node.py](src/controls/controls/control_node.py), [mppi_controller.py](src/controls/controls/mppi_controller.py), [report.py](src/controls/controls/run_audit/report.py), and [test_path_reference.py](src/controls/test/test_path_reference.py). The crash fix is in `control_node.py`.

**Verification.** The same recorded input replay reduced more-than-1 m reference jumps from 85 to 16 across 1,204 comparable update pairs. Subsequent Gazebo runs completed recording without the recovery-log crash. These results leave physical clearance through recovery open.

### `201d2da` and `49a030a` · Brake on rejection and limit recovery speed

**Why.** In acceleration mode a zero target-speed field alone does not brake the car. At recovery entry, the smoothed target could also remain above the intended crawl cap.

**What changed.** Rejected paths request −1.2 m/s² and zero target. Recovery caps both the filtered command and rollout targets at 1.5 m/s. Normal tracking holds that cap for the first handback cycle and releases it at 0.75 m/s per simulation second.

**Files.** [control_node.py](src/controls/controls/control_node.py) handles rejection, recovery and handback. [mppi_controller.py](src/controls/controls/mppi_controller.py) logs rejected geometry and enforces the cap in the filtered target and rollouts.

**Verification.** Isolated command tests passed. Six 10:08 handbacks logged a first tracking target of 1.5 m/s. A sustained braking and stopping-distance test on rejection remains pending.

### `d7e8380` · Establish the shared Exp13 baseline and per-run recorder

**Why.** The team needed the same ground-truth planner/controls launch and a repeatable record for every test. Exp13 addresses the learned heading-dependent yaw bias seen with the older model.

**What changed.** The integration packages the mirror-symmetric checkpoint and launches the active controller with automatic per-run logs, raw events, trajectory and HTML report. The setup script connects Galactic, EUFS and the team workspace.

**Files.** [mppi_controller.py](src/controls/controls/mppi_controller.py) loads the [Exp13 checkpoint](src/controls/controls/fs_model/exp13_stage1_pretrained.pt). [control_node.py](src/controls/controls/control_node.py) publishes control commands and diagnostics. [run_session.py](src/controls/controls/run_session.py), [run_audit](src/controls/controls/run_audit/), and [setup.py](src/controls/setup.py) package and run the recorder. [setup_integration.sh](scripts/setup_integration.sh) and [INTEGRATION.md](INTEGRATION.md) configure the runtime. [stack.launch.py](bringup/launch/stack.launch.py) establishes the ground-truth pipeline and planner launch.

**Verification.** CPU and GPU symmetry and zero-yaw tests passed. Both CUDA rollout horizons ran. The GUI controller generated per-run reports. Contact coverage varies by recording and is reported as unknown when no payload is captured.

## Updating this file

For each controls change, add an entry with the commit or PR, the run or defect that motivated it, the behavior change, the affected files and the tests performed. Include what still needs a replay, GUI test or teammate verification. Update the corresponding objective above when its status changes. The Desktop tracker retains the issue IDs and detailed run evidence.
