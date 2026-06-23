#!/usr/bin/env python3
"""
control_node.py  —  MPPI controller entry point for RAMS-e-25 (EUFS Gazebo).

Replaces the previous geometric forward-search controller with the exp09
Neural ODE MPPI (mppi_controller.py).  All ROS wiring, telemetry, and
sim-time management are kept identical so nothing else in the stack changes.

Run:  ros2 run controls control_node
"""
import rclpy
from rclpy.time import Time
import threading, time, math
import numpy as np
import matplotlib.pyplot as plt

from controls.ros_connect import ROSInterface
from controls.control_utils import compute_signed_curvature
from controls.telemetryplot import TelemetryVisualizer, generate_turning_arc
from controls.mppi_controller import MPPIController

# ================================
# Control Constants
# ================================
MAX_VELOCITY        = 4.0    # m/s  — raise once MPPI is validated
WHEELBASE_M         = 1.5
MAX_STEER_RAD       = 0.349  # model training limit; EUFS hardware cap is 0.7
ROUTE_IS_LOOP       = False
STOP_SPEED_THRESHOLD = -10.1

# ================================
# Visualization Constants
# ================================
VIZ_UPDATE_HZ = 20


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
    print("Waiting for first odometry message...")
    while True:
        cx, cy, yaw, speed, have_odom = node.get_state()
        if have_odom:
            print("First position received.")
            break
        time.sleep(0.1)

    # ---- initialise MPPI ----
    mppi = MPPIController()

    # ---- state initialisation ----
    last_ros_time       = node.get_clock().now()
    last_viz_update_real = time.perf_counter()

    cur_idx = 0
    viz     = None

    plt.ion()

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

        if not new_path:
            # No path yet — brake and wait
            node.send_command(steering=0.0, speed=0.0, accel=-mppi.ACCEL_MAX)
            time.sleep(0.05)
            continue

        path_points = np.array(new_path)
        route_x, route_y = path_points[:, 0], path_points[:, 1]

        # ---- init telemetry visualizer on first good path ----
        if viz is None:
            vt_viz = np.full_like(route_x, MAX_VELOCITY)
            viz = TelemetryVisualizer(route_x, route_y, vt_viz)
            plt.show()

        # ---- run MPPI ----
        steer, accel_cmd, info = mppi.compute(cx, cy, yaw, speed, dt)

        # Clamp to hardware limits (safety)
        steer     = float(np.clip(steer,     -MAX_STEER_RAD,  MAX_STEER_RAD))
        accel_cmd = float(np.clip(accel_cmd, -mppi.ACCEL_MAX, mppi.ACCEL_MAX))

        target_speed = info["target_speed"]
        cte          = info["cte"]
        heading_err  = info["heading_err"]
        mean_traj    = info["mean_traj"]   # list of (x,y) for arc display

        # ---- telemetry ----
        if viz is not None:
            vt_viz = np.full_like(route_x, MAX_VELOCITY)
            viz.update_path_data(route_x, route_y, vt_viz)

            arc_pts = generate_turning_arc(cx, cy, yaw, steer, WHEELBASE_M)

            # Use MPPI mean trajectory as "future_pts" for the path overlay
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
            print("End of route.")
            break

        time.sleep(0.05)

    rclpy.shutdown()
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
