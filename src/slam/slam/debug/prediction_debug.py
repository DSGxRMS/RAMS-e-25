#!/usr/bin/env python3
# imu_odom_debug_plotter.py
#
# Simple debug visualizer for IMU-based odometry:
#  - Subscribes to a primary odom topic (e.g., /slam/odom_raw)
#  - Optionally subscribes to a reference odom (e.g., /ground_truth/odom)
#  - Plots trajectories in real time.
#
import math
import threading  # still used for locks, not for GUI

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy

from nav_msgs.msg import Odometry

import matplotlib
matplotlib.use("TkAgg")  # GUI backend
import matplotlib.pyplot as plt


class ImuOdomDebugPlotter(Node):
    def __init__(self):
        super().__init__("imu_odom_debug_plotter")

        # ---- Parameters ----
        self.declare_parameter("topics.odom_in", "/slam/odom_raw")
        self.declare_parameter("topics.ref_odom_in", "")  # empty => no reference
        self.declare_parameter("qos.best_effort", True)
        self.declare_parameter("qos.depth", 200)
        self.declare_parameter("plot.trail_max_points", 10000)
        self.declare_parameter("plot.fixed_window", False)
        self.declare_parameter("plot.window_half_size", 50.0)

        gp = self.get_parameter
        odom_topic = str(gp("topics.odom_in").value)
        ref_odom_topic = str(gp("topics.ref_odom_in").value)
        best_effort = bool(gp("qos.best_effort").value)
        depth = int(gp("qos.depth").value)
        self.trail_max_points = int(gp("plot.trail_max_points").value)
        self.fixed_window = bool(gp("plot.fixed_window").value)
        self.window_half_size = float(gp("plot.window_half_size").value)

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=depth,
            reliability=(QoSReliabilityPolicy.BEST_EFFORT if best_effort else QoSReliabilityPolicy.RELIABLE),
            durability=QoSDurabilityPolicy.VOLATILE
        )

        # ---- Subscriptions ----
        self.create_subscription(Odometry, odom_topic, self.cb_odom, qos)
        self.ref_enabled = len(ref_odom_topic) > 0
        if self.ref_enabled:
            self.create_subscription(Odometry, ref_odom_topic, self.cb_ref_odom, qos)

        # ---- State for plotting ----
        self.traj_x = []
        self.traj_y = []
        self.car_x = None
        self.car_y = None

        self.ref_traj_x = []
        self.ref_traj_y = []
        self.ref_x = None
        self.ref_y = None

        self.data_lock = threading.Lock()

        # ---- Matplotlib figure ----
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.grid(True)
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_title("IMU Odom Debug Plotter")

        self.get_logger().info(
            f"[imu_odom_debug_plotter] odom={odom_topic}, "
            f"ref_odom={ref_odom_topic or 'NONE'}"
        )

    # --------------------- Callbacks ---------------------
    def cb_odom(self, msg: Odometry):
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)

        with self.data_lock:
            self.car_x = x
            self.car_y = y
            self.traj_x.append(x)
            self.traj_y.append(y)
            if len(self.traj_x) > self.trail_max_points:
                self.traj_x = self.traj_x[-self.trail_max_points:]
                self.traj_y = self.traj_y[-self.trail_max_points:]

    def cb_ref_odom(self, msg: Odometry):
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)

        with self.data_lock:
            self.ref_x = x
            self.ref_y = y
            self.ref_traj_x.append(x)
            self.ref_traj_y.append(y)
            if len(self.ref_traj_x) > self.trail_max_points:
                self.ref_traj_x = self.ref_traj_x[-self.trail_max_points:]
                self.ref_traj_y = self.ref_traj_y[-self.trail_max_points:]

    # --------------------- Plot update (NO plt.pause here) ---------------------
    def update_plot(self):
        with self.data_lock:
            traj_x = list(self.traj_x)
            traj_y = list(self.traj_y)
            car_x = self.car_x
            car_y = self.car_y

            ref_traj_x = list(self.ref_traj_x)
            ref_traj_y = list(self.ref_traj_y)
            ref_x = self.ref_x
            ref_y = self.ref_y

        self.ax.clear()
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.grid(True)
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_title("IMU Odom Debug Plotter")

        # IMU odom trajectory
        if traj_x and traj_y:
            self.ax.plot(traj_x, traj_y, "-", linewidth=1.0, color="blue", label="IMU odom traj")

        # Current IMU pose
        if car_x is not None and car_y is not None:
            self.ax.plot(car_x, car_y, "bo", markersize=5, label="IMU pose")

        # Reference trajectory
        if ref_traj_x and ref_traj_y:
            self.ax.plot(ref_traj_x, ref_traj_y, "--", linewidth=1.0,
                         color="green", label="ref traj")

        # Current reference pose
        if ref_x is not None and ref_y is not None:
            self.ax.plot(ref_x, ref_y, "go", markersize=5, label="ref pose")

        # Windowing
        all_x = []
        all_y = []
        all_x.extend(traj_x)
        all_y.extend(traj_y)
        all_x.extend(ref_traj_x)
        all_y.extend(ref_traj_y)
        if car_x is not None:
            all_x.append(car_x)
            all_y.append(car_y)
        if ref_x is not None:
            all_x.append(ref_x)
            all_y.append(ref_y)

        if self.fixed_window and (car_x is not None and car_y is not None):
            L = self.window_half_size
            self.ax.set_xlim(car_x - L, car_x + L)
            self.ax.set_ylim(car_y - L, car_y + L)
        else:
            if all_x and all_y:
                xmin, xmax = min(all_x), max(all_x)
                ymin, ymax = min(all_y), max(all_y)
                pad = 2.0
                if xmax - xmin < 1e-3:
                    xmax = xmin + 1.0
                if ymax - ymin < 1e-3:
                    ymax = ymin + 1.0
                self.ax.set_xlim(xmin - pad, xmax + pad)
                self.ax.set_ylim(ymin - pad, ymax + pad)

        self.ax.legend(loc="upper right", fontsize=8)


def main():
    rclpy.init()
    node = ImuOdomDebugPlotter()
    plt.ion()  # interactive mode

    try:
        while rclpy.ok():
            # Process ROS callbacks (runs in main thread)
            rclpy.spin_once(node, timeout_sec=0.01)

            # Update plot, then let GUI process events
            node.update_plot()
            plt.pause(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
