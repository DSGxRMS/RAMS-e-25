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
import json
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
from nav_msgs.msg import Path as PathMsg
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

# ================================
# Control Constants
# ================================
MAX_VELOCITY        = 5.0    # m/s  — target cruise velocity matched to MPPIController.V_MAX
WHEELBASE_M         = 1.5
MAX_STEER_RAD       = 0.349  # model training limit; EUFS hardware cap is 0.7

# ================================
# Visualization (optional / headless)
# ================================
# Default OFF: the live Tk telemetry window is laggy and crashes on Ctrl+C.
# The exit trajectory plot (_plot_trajectory, Agg backend) is the useful one and
# Set MPPI_VIZ=1 to force live plotting.
VIZ = os.environ.get("MPPI_VIZ", "0") == "1"
VIZ_UPDATE_HZ = 20

# PP-dropout behaviour (senior's spec): if PP stops publishing a path entirely,
# DON'T hard-brake — hold the last steer command and coast down gently.
COAST_DECEL_FRAC = 0.4       # fraction of ACCEL_MAX applied as gentle decel while waiting

# ONE-SIDED cone evidence: one colour must outnumber the other by this ratio
# (and have >=2 cones) to count as "only one wall visible". FS convention:
# blue = LEFT edge, yellow = RIGHT edge. Feeds both the pseudo-path trigger
# and MPPI's bad-path guard.
RECOVER_RATIO = 3.0

# Recovery uses a separately fitted reference and a crawl speed cap.
BRIDGE_SPEED_MPS = 1.5
BRIDGE_RELEASE_MPS2 = 0.75    # raise the speed ceiling in simulation time

if VIZ:
    import matplotlib.pyplot as plt
    from controls.telemetryplot import TelemetryVisualizer, generate_turning_arc


def main():
    rclpy.init()
    node = ROSInterface(ordered_reference=True)

    # CRITICAL: Sync with Simulator Clock
    node.set_parameters([
        rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
    ])

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    def spin():
        try:
            executor.spin()
        except rclpy.executors.ExternalShutdownException:
            pass

    executor_thread = threading.Thread(target=spin, daemon=True)
    executor_thread.start()

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

    # ---- live debug publishers (for controls/debug/pseudo_debug.py) ----
    pseudo_pub = node.create_publisher(PathMsg, "/debug/pseudo_path", 10)
    buffer_pub = node.create_publisher(PathMsg, "/debug/buffer_path", 10)
    rollout_pub = node.create_publisher(MarkerArray, "/debug/rollouts", 10)
    audit_pub = (node.create_publisher(String, "/debug/mppi_cycle", 10)
                 if mppi.audit_enabled else None)

    def _audit_command(mode, steering, target, acceleration, info=None, **details):
        """Record the command issued this cycle, including fallback branches."""
        if audit_pub is None:
            return
        info = info or {}
        row = {
            **info.get("audit", {}),
            **details,
            "model": os.environ.get("MPPI_MODEL", "exp13"),
            "mode": mode,
            "sim_ns": current_ros_time.nanoseconds,
            "cycle_ms": (time.perf_counter() - cycle_begin) * 1000,
            "pose_used": [cx, cy, yaw],
            "speed_used": speed,
            "buffer": new_path,
            "path_source": path_source,
            "command": {"steering_angle": steering, "speed": target, "acceleration": acceleration},
        }
        for key in ("cte", "heading_err", "hd_bad", "path_bad", "path_exhausted", "ahead_eff",
                    "n_eff", "ff_part", "mppi_part"):
            row[key] = info.get(key)
        # Map non-finite numerical diagnostics to null for strict JSON readers.
        encoded = json.dumps(row, separators=(",", ":"))
        row = json.loads(encoded, parse_constant=lambda _: None)
        msg = String()
        msg.data = json.dumps(row, separators=(",", ":"), allow_nan=False)
        audit_pub.publish(msg)

    def _to_path_msg(pts):
        m = PathMsg()
        m.header.frame_id = "map"
        m.header.stamp = node.get_clock().now().to_msg()
        for (px, py) in pts:
            ps = PoseStamped()
            ps.header = m.header
            ps.pose.position.x = float(px)
            ps.pose.position.y = float(py)
            m.poses.append(ps)
        return m

    def _publish_rollouts(top_trajs):
        """P14-c: the 10 cheapest MPPI rollouts, best first, as LINE_STRIP
        markers. Rank 0 (lowest cost) is drawn bright green and thick; the
        rest fade out with rank."""
        marr = MarkerArray()
        stamp = node.get_clock().now().to_msg()
        for rank, traj in enumerate(top_trajs):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = stamp
            m.ns = "rollouts"
            m.id = rank
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.14 if rank == 0 else 0.04
            m.color.a = 0.95 if rank == 0 else max(0.12, 0.45 - 0.045 * rank)
            if rank == 0:
                m.color.r, m.color.g, m.color.b = 0.0, 1.0, 0.3
            else:
                m.color.r, m.color.g, m.color.b = 1.0, 0.55, 0.1
            for (px, py) in traj:
                p = Point()
                p.x, p.y, p.z = float(px), float(py), 0.06
                m.points.append(p)
            marr.markers.append(m)
        rollout_pub.publish(marr)

    # ---- state initialisation ----
    last_ros_time        = node.get_clock().now()
    last_viz_update_real = time.perf_counter()

    viz        = None
    last_steer = 0.0          # held during PP dropout

    # ---- trajectory recorder (for the exit plot) ----
    rec_cx, rec_cy, rec_yaw, rec_steer, rec_cte = [], [], [], [], []
    rec_neff, rec_tgt_spd, rec_act_spd = [], [], []   # MPPI diversity + speed tracking
    rec_ff, rec_mppi = [], []                         # steer breakdown: FF part vs MPPI part
    rec_path_pts = []         # snapshots of (route_x, route_y) every N cycles
    pseudo_pts = []           # (x, y) car positions where a pseudo feed fired (exit plot)
    rec_ctr = 0
    pseudo_events = 0         # how many times the pseudo centreline extended the buffer
    rejected_planner = None
    prefer_recovery = False
    last_reject_print = 0.0   # throttle for the [PSEUDO] no-fit diagnostic
    bridge_cap = None
    bridge_was_fresh = False

    if VIZ:
        plt.ion()

    try:
      while rclpy.ok():
        # ---- sim-clock dt ----
        current_ros_time = node.get_clock().now()
        dt_nano = (current_ros_time - last_ros_time).nanoseconds
        dt = dt_nano / 1e9
        last_ros_time = current_ros_time

        if dt_nano < 0:
            bridge_cap = None
            bridge_was_fresh = False
            rejected_planner = None
            prefer_recovery = False
            node.invalidate_buffer()
            mppi.update_path(())
        if dt < 0.001:
            time.sleep(0.001)
            continue

        cycle_begin = time.perf_counter()
        pseudo_added = 0
        pseudo_reject = None

        # ---- get path and state ----
        new_path = ()
        path_source = "none"
        cx, cy, yaw, speed, have_odom = node.get_state()

        # ---- P16: frame-split guard ----
        # The cone map lives in the SLAM-corrected frame; we plan in the GT
        # frame. When snap-back splits them (~5 m measured), every cone, the
        # buffer and any bridge are displaced garbage -> hold a safe stop and
        # drop the poisoned buffer until the frames re-converge.
        if not node.frame_ok():
            node.invalidate_buffer()
            mppi.update_path(())
            rejected_planner = None
            prefer_recovery = False
            bridge_cap = BRIDGE_SPEED_MPS
            bridge_was_fresh = True
            node.send_command(steering=last_steer, speed=0.0,
                              accel=-COAST_DECEL_FRAC * mppi.ACCEL_MAX)
            _audit_command("frame_split", last_steer, 0.0, -COAST_DECEL_FRAC * mppi.ACCEL_MAX,
                           frame_split_m=node.frame_split_distance())
            now_s = time.perf_counter()
            if now_s - last_reject_print > 2.0:
                last_reject_print = now_s
                print(f"[FRAME] SLAM map split from GT pose by "
                      f"{node.frame_split_distance():.1f} m (snap-back?) "
                      f"-> buffer dropped, holding stop", flush=True)
            time.sleep(0.05)
            continue

        # ---- cone evidence for controls recovery ----
        nb, ny = node.get_local_cone_counts()
        one_sided = ((nb >= 2 and (ny == 0 or nb >= RECOVER_RATIO * ny)) or
                     (ny >= 2 and (nb == 0 or ny >= RECOVER_RATIO * nb)) or
                     (nb == 0 and ny == 0))

        now = current_ros_time.nanoseconds * 1e-9
        pose = (cx, cy, yaw)
        planner_path = node.reference.planner_path(now, pose)
        planner_available = bool(planner_path)
        can_track = bool(planner_path) and (not one_sided or planner_path != rejected_planner)
        if can_track and not prefer_recovery:
            new_path, path_source = planner_path, "planner"
        else:
            new_path = node.reference.recovery_path(now, pose)
            if new_path:
                path_source = "recovery"
        if not new_path:
            pseudo = node.pseudo_wall_path(allow_both_walls=(not one_sided))
            if node.reference.accept_recovery(pseudo, now):
                new_path = node.reference.recovery_path(now, pose)
                pseudo_added = len(new_path)
            if new_path:
                path_source = "recovery"
                pseudo_events += 1
                pseudo_pts.append((cx, cy))
                pseudo_pub.publish(_to_path_msg(new_path))
                wall = "blue" if nb > ny else "yellow"
                mode = "UNLOCK" if (pp_silent and not one_sided) else "one-sided"
                print(f"[PSEUDO] feed #{pseudo_events}: {wall} wall ({mode}), "
                      f"+{len(pseudo)} pts, nb={nb} ny={ny} (speed cap 1.5)",
                      flush=True)
            else:
                pseudo_reject = node.last_pseudo_reject if not pseudo else "too little recovery path ahead"
                now_s = time.perf_counter()
                if now_s - last_reject_print > 2.0:
                    last_reject_print = now_s
                    print(f"[PSEUDO] no fit: {pseudo_reject}", flush=True)

        if not new_path and can_track:
            new_path, path_source = planner_path, "planner"
        prefer_recovery = False
        if new_path:
            buffer_pub.publish(_to_path_msg(new_path))
        mppi.update_path(new_path)

        if new_path:
            fresh = path_source == "recovery"
            one_sided_for_compute = one_sided
            if fresh:
                bridge_cap = BRIDGE_SPEED_MPS
            elif bridge_cap is not None and not bridge_was_fresh:
                bridge_cap += BRIDGE_RELEASE_MPS2 * dt
                if bridge_cap >= mppi.V_MAX:
                    bridge_cap = None
            # Hold the crawl cap on the first cycle after the bridge expires.
            bridge_was_fresh = fresh
            steer, accel_cmd, info = mppi.compute(cx, cy, yaw, speed, dt,
                                                  one_sided=one_sided_for_compute,
                                                  speed_cap=bridge_cap)
            last_path_bad  = bool(info.get("path_bad", False))
            if path_source == "planner":
                rejected_planner = planner_path if last_path_bad else None
                prefer_recovery = last_path_bad
            if last_path_bad:
                # A zero speed target alone leaves Gazebo coasting in acceleration mode.
                accel_cmd = -COAST_DECEL_FRAC * mppi.ACCEL_MAX
                if fresh:
                    node.reference.recovery = None
            tt = info.get("top_trajs")
            if tt:
                _publish_rollouts(tt)
        else:
            # PP dropout with nothing to fit (no lone wall either): hold last
            # steer and coast down gently (senior's spec). MPPI's own guards
            # (vr caps, STOP_PATH_M) handle every partial-path case.
            node.send_command(steering=last_steer, speed=0.0,
                              accel=-COAST_DECEL_FRAC * mppi.ACCEL_MAX)
            _audit_command("no_path", last_steer, 0.0, -COAST_DECEL_FRAC * mppi.ACCEL_MAX,
                           blue_in_fov=nb, yellow_in_fov=ny, planner_available=planner_available,
                           pseudo_added=pseudo_added, pseudo_reject=pseudo_reject)
            time.sleep(0.05)
            continue

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
        rec_neff.append(info.get("n_eff", float('nan')))
        rec_tgt_spd.append(info.get("target_speed", float('nan')))
        rec_act_spd.append(info.get("actual_speed", speed))
        rec_ff.append(info.get("ff_part", float('nan')))
        rec_mppi.append(info.get("mppi_part", float('nan')))
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
        if audit_pub is not None:
            if last_path_bad:
                audit_mode = "path_rejected"
            elif info.get("path_exhausted"):
                audit_mode = "path_exhausted"
            elif fresh:
                audit_mode = "pseudo_bridge"
            else:
                audit_mode = "tracking"
            _audit_command(audit_mode, steer, target_speed, accel_cmd, info,
                           blue_in_fov=nb, yellow_in_fov=ny, planner_available=planner_available,
                           bridge_fresh=fresh, one_sided_for_compute=one_sided_for_compute,
                           pseudo_added=pseudo_added, pseudo_reject=pseudo_reject)

        # No artificial sleep here: the MPPI rollout already takes ~110 ms, so this loop
        # runs as fast as the compute allows. The old 50 ms sleep dropped the control rate
        # to ~4.7 Hz (0.64 m of travel per cycle at 3 m/s) — a huge steering delay that
        # made the car overshoot and zig-zag. Odom/path callbacks run in a separate
        # executor thread, so the loop needs no spin here. (~1 ms yield keeps CPU sane.)
        time.sleep(0.001)
    except KeyboardInterrupt:
        print("\n[control_node] stopped by user.", flush=True)
    finally:
        print("[control_node] generating trajectory plot...", flush=True)
        try:
            _plot_trajectory(rec_cx, rec_cy, rec_yaw, rec_steer, rec_cte, rec_path_pts,
                             rec_neff, rec_tgt_spd, rec_act_spd, mppi.K,
                             pseudo_events, rec_ff, rec_mppi, off_pts=pseudo_pts)
        except Exception as e:
            print(f"[plot] failed: {e}", flush=True)
        executor.shutdown(timeout_sec=2.0)
        executor_thread.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _plot_trajectory(cx, cy, yaw, steer, cte, path_snaps,
                     neff=None, tgt_spd=None, act_spd=None, K=1024,
                     pseudo_events=0, ff=None, mppi=None, off_pts=None):
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
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(6, 2, width_ratios=[2, 1], hspace=0.6, wspace=0.25)

    # main XY: car path + all PP path snapshots
    ax = fig.add_subplot(gs[:, 0])
    for (px, py) in path_snaps:
        ax.plot(px, py, '-', color='lightblue', lw=0.6, alpha=0.5)
    if path_snaps:
        ax.plot([], [], '-', color='lightblue', label='PP path snapshots')
    if off_pts:
        ax.scatter([p[0] for p in off_pts], [p[1] for p in off_pts],
                   c='red', s=8, zorder=4, label='pseudo-path feed positions')
    sc = ax.scatter(cx, cy, c=np.arange(len(cx)), cmap='viridis', s=10, zorder=3)
    ax.plot(cx, cy, '-', color='gray', lw=0.5, alpha=0.5)
    ax.scatter([cx[0]], [cy[0]], c='lime', s=140, marker='o', edgecolor='k', zorder=5, label='start')
    ax.scatter([cx[-1]], [cy[-1]], c='red', s=140, marker='X', edgecolor='k', zorder=5, label='end')
    ax.set_title(f'Car trajectory (GT) vs PP path\n'
                 f'pseudo-path feeds: {pseudo_events}',
                 fontsize=11)
    ax.set_xlabel('E (m)'); ax.set_ylabel('N (m)')
    ax.axis('equal'); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.colorbar(sc, ax=ax, label='time step', shrink=0.5)

    t = np.arange(len(cx))
    a1 = fig.add_subplot(gs[0, 1]); a1.plot(t, np.degrees(yaw), '-b'); a1.set_title('car yaw (deg)'); a1.grid(alpha=0.3)
    a2 = fig.add_subplot(gs[1, 1]); a2.plot(t, steer, '-m'); a2.set_title('steer cmd (rad)'); a2.grid(alpha=0.3)
    a3 = fig.add_subplot(gs[2, 1]); a3.plot(t, cte, '-c'); a3.set_title('CTE (m)'); a3.grid(alpha=0.3)

    # nEff: MPPI sample diversity. K = degenerate (MPPI doing nothing), low = peaked/useful.
    a4 = fig.add_subplot(gs[3, 1])
    if neff is not None and len(neff):
        a4.plot(t, np.array(neff), '-', color='darkorange')
        a4.axhline(K, color='r', ls='--', lw=0.8, alpha=0.7, label=f'K={K} (MPPI dead)')
        a4.set_ylim(0, K * 1.05); a4.legend(fontsize=7, loc='lower right')
    a4.set_title('MPPI nEff (lower = MPPI useful)'); a4.grid(alpha=0.3)

    # target vs actual speed (replaces the telemetry window you don't like)
    a5 = fig.add_subplot(gs[4, 1])
    if tgt_spd is not None and len(tgt_spd):
        a5.plot(t, np.array(tgt_spd), '-', color='green', label='target')
        a5.plot(t, np.array(act_spd), '-', color='black', lw=0.9, label='actual')
        a5.legend(fontsize=7, loc='upper right')
    a5.set_title('speed: target vs actual (m/s)'); a5.grid(alpha=0.3)

    # STEER BREAKDOWN: who is steering? net = FF part + MPPI part (before clip).
    # If 'net' tracks 'FF' -> feedforward drives; if it tracks 'MPPI' -> MPPI drives.
    a6 = fig.add_subplot(gs[5, 1])
    if ff is not None and len(ff):
        a6.axhline(0, color='gray', lw=0.5)
        a6.plot(t, np.array(ff),    '-', color='tab:blue',   lw=1.0, label='FF part (K_FF·ff0)')
        a6.plot(t, np.array(mppi),  '-', color='tab:red',    lw=1.0, label='MPPI part')
        a6.plot(t, np.array(steer), '-', color='k',          lw=1.2, alpha=0.8, label='net steer')
        a6.legend(fontsize=6, loc='upper right', ncol=1)
    a6.set_title('steer breakdown (rad): FF vs MPPI vs net'); a6.set_xlabel('step'); a6.grid(alpha=0.3)

    # The run session supplies a per-run path.
    out = os.environ.get("MPPI_TRAJECTORY_PNG", "mppi_trajectory.png")
    try:
        plt.savefig(out, dpi=120, bbox_inches='tight')
        print(f"[plot] saved {out}", flush=True)
    except Exception as e:
        print(f"[plot] save failed: {e}", flush=True)


if __name__ == '__main__':
    main()
