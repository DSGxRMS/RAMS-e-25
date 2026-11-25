# Controls changes and next objectives

Branch `abh_soum_integration` · Ground-truth EUFS Small Track · Updated 25 November 2026

The active controller is the mirror-symmetric Exp13 Stage 1 Neural ODE with MPPI. The current target-speed cap is 3 m/s. The long-term target is **at least 7 m/s average over a complete, repeatable lap**. The latest local drive averaged 2.60 m/s from first motion to the last command and 2.62 m/s gate to gate. The return still overlapped two surveyed cone positions in the chassis projection, and the contact recorder had no payload.

## Next objectives

| Order | Objective | What to build or test | Evidence to collect |
| --- | --- | --- | --- |
| 1 | **Check the whole car against cones** · CTRL-04 | Start with the recorded orange-gate returns at 15:02 and 15:25. Check swept body and wheel clearance along recovery and normal-tracking rollouts. Recheck the final weighted action using the same delayed, rate-limited actuator model. Give the controller an explicit braking response when the available candidates fail the check. This is the geometric first step toward [Shield-MPPI](https://arxiv.org/abs/2302.11719). | Flag both recorded overlaps and accept the centered opening pass. Report minimum predicted clearance, feasible candidates, the chosen action, false stops and added solve time. |
| 2 | **Handle stale state, late solves and resets** · CTRL-01, 06, 09, 18 | Represent missing speed explicitly. Reject stale commands before publish, and brake if the queued steering fills the active prediction horizon. Verify pose jumps and clock resets clear old plan, path and command history. Check the CUDA worker on the RTX 5050. | Replay slow CPU conditions, delayed feedback and resets. Confirm stale acceleration is suppressed and measure actual brake onset, stopping distance and recovery. Record decision-to-publish p50/p95 in simulation time. |
| 3 | **Improve recovery and the return to PP** · CTRL-02, 03, 04 | Check a recovery route and the first returning planner route for forward progress and clearance. Preserve conservative speed while clearance or usable lookahead is poor. Send the planner team the frame IDs and geometry for gaps, discontinuities and unsafe routes. Planner algorithm edits are owned by that team. | Replay the 4.4 s gap and subsequent gate return from 15:25. Compare clearance, time spent near steering lock, time to resume tracking, stop rate and finish rate. |
| 4 | **Finish state and steering timing** · CTRL-08, 09, 13, 16 | Align measured lateral velocity and yaw rate with the pose. Verify the slip convention and compare Exp13 predictions with measured motion before changing the state seed. Anchor warm-start reuse to the actual previous publication and test the backward-shift case. Keep measured wheel sync enabled for the warm-start A/B. | Compare short-horizon yaw and pose prediction errors, physical yaw reversals and lateral error on the opening, corner turn-in, and solve latency. The latest run had 12 backward shifts and fewer command movements but more yaw reversals than the prior drive. |
| 5 | **Compare smoother MPPI sampling** | Audit the existing cost, temperature and sampling correction against [Williams et al.](https://arxiv.org/abs/1707.02342). Test [LP-MPPI](https://arxiv.org/html/2503.11717v3) first by filtering sample noise before rollouts. Then test [Smooth MPPI](https://arxiv.org/abs/2112.09988) by sampling steering rate. Change one option at a time with the same rollout budget. | Measure requested and applied steering movement, yaw reversals, lateral drift, turn-in, clearance and p95 solve time. Repeat fresh-world GUI trials with matched seed schedules. |
| 6 | **Establish a repeatable clean-lap score** | Validate contact coverage, add oriented gate-to-gate lap scoring and flag resets or departures. Start collecting this evidence during objective 1. Use the Small Track GUI for comparable runs. | Report complete laps, contact count or unknown coverage, minimum physical clearance, mean and peak speed, recovery time and run-to-run variation at the 3 m/s target cap. |
| 7 | **Increase pace toward 7 m/s average** | Calibrate braking, lateral grip, curvature and visible path length. Build a speed envelope from those measurements and add bounded progress only within the checked corridor. Test higher caps in steps once the clean-lap baseline holds. [Liniger et al.](https://arxiv.org/abs/1711.07300) · [Stanford tire-limit speed control](https://ddl.stanford.edu/publications/conference-proceeding/speed-control-robust-path-tracking-automated-vehicles-tire-road) | Score gate-to-gate speed alongside contact coverage, clearance, finish rate and braking margin. |
| Later | **Test robustness and maintain diagnostics** · CTRL-05, 07, 10–12 | Use measured model error to decide whether [Robust MPPI](https://arxiv.org/abs/2102.09027) or [Tube-MPPI](https://arxiv.org/abs/2110.07744) adds value. Repair the legacy smoke test and topic diagnostic. Track active legacy callers before changing old buffer and tangent helpers. Keep full shutdown logs for any distinct exit traceback. | Compare under measured disturbances. Confirm the smoke test runs with the active interface and the diagnostic shows the actual pose and speed topics. |

## Implemented changes

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
