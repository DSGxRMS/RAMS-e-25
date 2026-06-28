#!/usr/bin/env python3
"""
control_node.py  —  MPPI controller ENTRY POINT for RAMS-e-25 (EUFS Gazebo).

This is the active control node (per the branch layout: control logic is driven
from here; helpers in control_utils.py; sim comms in ros_connect.py). The heavy
MPPI rollout lives in mppi_controller.py and is invoked via MPPIController.compute().

Velocity note: SLAM /slam/odom publishes twist=0, so ros_connect.py reads the real
speed from /ground_truth/odom while keeping position/yaw from /slam/odom.

Headless: set MPPI_VIZ=0 (and MPLBACKEND=Agg) to run without the telemetry plot
(needed for headless / on-car operation). Default MPPI_VIZ=1 shows the plot.

Run:  ros2 run controls control_node
      MPPI_VIZ=0 ros2 run controls control_node   # headless
"""
import os
import rclpy
from rclpy.time import Time
import threading, time, math
import numpy as np

# Force a non-interactive matplotlib backend at import time so the exit plot
# never tries to open a Tk window (which crashes on Ctrl+C). MPPI_VIZ=1 overrides.
if os.environ.get("MPPI_VIZ", "0") != "1":
    import matplotlib
    matplotlib.use("Agg")

from controls.ros_connect import ROSInterface
from controls.control_utils import compute_signed_curvature
from controls.mppi_controller import MPPIController

# ================================
# Control Constants
# ================================
MAX_VELOCITY        = 4.0    # m/s  — NN lower training bound (real steering authority)
WHEELBASE_M         = 1.5
MAX_STEER_RAD       = 0.349  # model training limit; EUFS hardware cap is 0.7
ROUTE_IS_LOOP       = False
STOP_SPEED_THRESHOLD = -10.1

# ================================
# Visualization (optional / headless)
# ================================
# Default OFF: the live Tk telemetry window is laggy and crashes on Ctrl+C.
# The exit trajectory plot (_plot_trajectory, Agg backend) is the useful one and
# always saves to D:\RAMS-e-25\mppi_trajectory.png. Set MPPI_VIZ=1 to force live.
VIZ = os.environ.get("MPPI_VIZ", "0") == "1"
VIZ_UPDATE_HZ = 20

# PP-dropout behaviour (senior's spec): if PP stops publishing a usable path,
# DON'T hard-brake — hold the last steer command and coast down gently until PP recovers.
COAST_DECEL_FRAC = 0.4       # fraction of ACCEL_MAX applied as gentle decel while waiting

# ONE-SIDED-CONE RECOVERY (senior's option c): when the path is unusable (PP gave nothing,
# OR it points the wrong way — the one-sided-cone heading flip) AND only one cone colour is
# in view, steer a fixed gentle angle toward the MISSING wall at a slow creep until PP
# recovers, then hand back to MPPI. FS convention: blue=LEFT edge, yellow=RIGHT edge.
#   only blue (left) visible  -> turn RIGHT  (steer < 0)
#   only yellow (right) visible-> turn LEFT   (steer > 0)
RECOVER_STEER = 0.20         # rad (gentle, not full lock)
RECOVER_SPEED = 1.0          # m/s creep during recovery

if VIZ:
    import matplotlib.pyplot as plt
    from controls.telemetryplot import TelemetryVisualizer, generate_turning_arc


def main():
    rclpy.init()
    node = ROSInterface()

    # CRITICAL: Sync with Simulator Clock
    node.set_parameters([
        rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
    ])

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    # ---- wait for first odometry ----
    print("Waiting for first odometry message...", flush=True)
    while True:
        cx, cy, yaw, speed, have_odom = node.get_state()
        if have_odom:
            print("First position received.", flush=True)
            break
        time.sleep(0.1)

    # ---- initialise MPPI ----
    mppi = MPPIController()

    # ---- state initialisation ----
    last_ros_time        = node.get_clock().now()
    last_viz_update_real = time.perf_counter()

    cur_idx   = 0
    viz       = None
    last_steer = 0.0          # held during PP dropout
    was_recovering = False     # one-sided-cone recovery state (for transition logging)

    # ---- trajectory recorder (for the exit plot) ----
    rec_cx, rec_cy, rec_yaw, rec_steer, rec_cte = [], [], [], [], []
    rec_path_pts = []         # snapshots of (route_x, route_y) every N cycles
    rec_ctr = 0

    if VIZ:
        plt.ion()

    try:
      while rclpy.ok():
        # ---- sim-clock dt ----
        current_ros_time = node.get_clock().now()
        dt_nano = (current_ros_time - last_ros_time).nanoseconds
        dt = dt_nano / 1e9
        last_ros_time = current_ros_time

        if dt < 0.001:
            time.sleep(0.001)
            continue

        # ---- get path and state ----
        new_path   = node.get_path()
        cx, cy, yaw, speed, have_odom = node.get_state()

        # Push new path into MPPI (no-op if path hasn't changed)
        mppi.update_path(new_path)

        # ---- decide if the path is usable ----
        # Two failure modes count as "PP not giving a path":
        #   (1) empty path, or
        #   (2) MPPI flags path_bad (one-sided-cone heading flip — points the wrong way).
        info = None
        steer = accel_cmd = None
        path_unusable = (not new_path)
        if not path_unusable:
            steer, accel_cmd, info = mppi.compute(cx, cy, yaw, speed, dt)
            path_unusable = bool(info.get("path_bad", False))

        if path_unusable:
            # ---- ONE-SIDED-CONE RECOVERY (option c) ----
            nb, ny = node.get_local_cone_counts()
            if nb > 0 and ny == 0:
                rsteer = -RECOVER_STEER          # only blue (left) -> turn right
                recovering = True
            elif ny > 0 and nb == 0:
                rsteer = +RECOVER_STEER          # only yellow (right) -> turn left
                recovering = True
            else:
                recovering = False               # can't tell side (both/neither in view)

            if recovering:
                racc = float(np.clip(0.6 * (RECOVER_SPEED - speed),
                                     -mppi.ACCEL_MAX, mppi.ACCEL_MAX))
                node.send_command(steering=rsteer, speed=RECOVER_SPEED, accel=racc)
                last_steer = rsteer
                if not was_recovering:
                    side = "blue->RIGHT" if rsteer < 0 else "yellow->LEFT"
                    print(f"[REC] one-sided cones ({side}) nb={nb} ny={ny} "
                          f"steer={rsteer:+.2f} creep={RECOVER_SPEED:.1f} m/s", flush=True)
            else:
                # ambiguous (both or no cones near) -> safe coast, hold last steer
                node.send_command(steering=last_steer, speed=0.0,
                                  accel=-COAST_DECEL_FRAC * mppi.ACCEL_MAX)
                if not was_recovering:
                    print(f"[REC] path unusable, side ambiguous nb={nb} ny={ny} -> coast/stop",
                          flush=True)
            was_recovering = True
            time.sleep(0.05)
            continue

        if was_recovering:
            print("[REC] PP recovered -> back to MPPI", flush=True)
            was_recovering = False

        path_points = np.array(new_path)
        route_x, route_y = path_points[:, 0], path_points[:, 1]

        # ---- init telemetry visualizer on first good path ----
        if VIZ and viz is None:
            vt_viz = np.full_like(route_x, MAX_VELOCITY)
            viz = TelemetryVisualizer(route_x, route_y, vt_viz)
            plt.show()

        # Clamp to hardware limits (safety)
        steer     = float(np.clip(steer,     -MAX_STEER_RAD,  MAX_STEER_RAD))
        accel_cmd = float(np.clip(accel_cmd, -mppi.ACCEL_MAX, mppi.ACCEL_MAX))
        last_steer = steer

        # ---- record for the exit trajectory plot ----
        rec_cx.append(cx); rec_cy.append(cy); rec_yaw.append(yaw)
        rec_steer.append(steer); rec_cte.append(info["cte"])
        rec_ctr += 1
        if rec_ctr % 5 == 1:                      # snapshot the PP path occasionally
            rec_path_pts.append((route_x.copy(), route_y.copy()))

        target_speed = info["target_speed"]
        cte          = info["cte"]
        heading_err  = info["heading_err"]
        mean_traj    = info["mean_traj"]   # list of (x,y) for arc display

        # ---- telemetry ----
        if VIZ and viz is not None:
            vt_viz = np.full_like(route_x, MAX_VELOCITY)
            viz.update_path_data(route_x, route_y, vt_viz)

            arc_pts = generate_turning_arc(cx, cy, yaw, steer, WHEELBASE_M)
            lookahead_pt = mean_traj[0] if mean_traj else None

            viz.log_state(
                x=cx, y=cy, yaw=yaw, speed=speed,
                steering_cmd=steer / MAX_STEER_RAD,   # normalised
                lookahead_pt=lookahead_pt,
                future_pts=mean_traj,
                arc_pts=arc_pts,
                target_speed=target_speed,
                cross_track_error=cte,
                heading_error=heading_err,
            )

            if time.perf_counter() - last_viz_update_real >= (1.0 / VIZ_UPDATE_HZ):
                viz.update_plot_manual()
                plt.pause(0.001)
                last_viz_update_real = time.perf_counter()

        # ---- actuation ----
        node.send_command(steering=steer, speed=target_speed, accel=accel_cmd)

        # ---- stop condition ----
        if (not ROUTE_IS_LOOP) and cur_idx >= len(route_x) - 5 and speed < STOP_SPEED_THRESHOLD:
            print("End of route.", flush=True)
            break

        time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[control_node] stopped by user.", flush=True)
    finally:
        print("[control_node] generating trajectory plot...", flush=True)
        try:
            _plot_trajectory(rec_cx, rec_cy, rec_yaw, rec_steer, rec_cte, rec_path_pts)
        except Exception as e:
            print(f"[plot] failed: {e}", flush=True)
        rclpy.shutdown()


def _plot_trajectory(cx, cy, yaw, steer, cte, path_snaps):
    """Save GT car trajectory vs the PP path snapshots + diagnostics on exit.
    Uses the Agg backend so it ALWAYS saves to a PNG (never blocks/crashes on
    a second Ctrl+C). Saved to the D drive so it's visible from Windows."""
    import matplotlib
    matplotlib.use("Agg")          # non-interactive: always saves, never hangs
    import matplotlib.pyplot as plt
    import numpy as np
    if len(cx) < 2:
        print("[plot] not enough data", flush=True)
        return
    cx = np.array(cx); cy = np.array(cy)
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[2, 1], hspace=0.4, wspace=0.25)

    # main XY: car path + all PP path snapshots
    ax = fig.add_subplot(gs[:, 0])
    for (px, py) in path_snaps:
        ax.plot(px, py, '-', color='lightblue', lw=0.6, alpha=0.5)
    if path_snaps:
        ax.plot([], [], '-', color='lightblue', label='PP path snapshots')
    sc = ax.scatter(cx, cy, c=np.arange(len(cx)), cmap='viridis', s=10, zorder=3)
    ax.plot(cx, cy, '-', color='gray', lw=0.5, alpha=0.5)
    ax.scatter([cx[0]], [cy[0]], c='lime', s=140, marker='o', edgecolor='k', zorder=5, label='start')
    ax.scatter([cx[-1]], [cy[-1]], c='red', s=140, marker='X', edgecolor='k', zorder=5, label='end')
    ax.set_title('Car trajectory (GT) vs PP path'); ax.set_xlabel('E (m)'); ax.set_ylabel('N (m)')
    ax.axis('equal'); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.colorbar(sc, ax=ax, label='time step', shrink=0.5)

    t = np.arange(len(cx))
    a1 = fig.add_subplot(gs[0, 1]); a1.plot(t, np.degrees(yaw), '-b'); a1.set_title('car yaw (deg)'); a1.grid(alpha=0.3)
    a2 = fig.add_subplot(gs[1, 1]); a2.plot(t, steer, '-m'); a2.set_title('steer cmd (rad)'); a2.grid(alpha=0.3)
    a3 = fig.add_subplot(gs[2, 1]); a3.plot(t, cte, '-c'); a3.set_title('CTE (m)'); a3.set_xlabel('step'); a3.grid(alpha=0.3)

    # save to the D drive so it's visible from Windows (\\wsl... or D:\)
    out = "/mnt/d/RAMS-e-25/mppi_trajectory.png"
    try:
        plt.savefig(out, dpi=120, bbox_inches='tight')
        print(f"[plot] saved {out}  (Windows: D:\\RAMS-e-25\\mppi_trajectory.png)", flush=True)
    except Exception as e:
        print(f"[plot] save failed: {e}", flush=True)


if __name__ == '__main__':
    main()
