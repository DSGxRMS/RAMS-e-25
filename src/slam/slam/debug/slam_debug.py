#!/usr/bin/env python3
# slam_debug_plotter.py
#
# Simple debug visualizer for FastSLAM:
#  - Subscribes to /slam/odom and /slam/map_cones
#  - Plots vehicle pose (triangle) and map cones in real time.
#
import math
import threading

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
)

from nav_msgs.msg import Odometry
from eufs_msgs.msg import ConeArrayWithCovariance

import matplotlib
matplotlib.use("TkAgg")  # GUI backend
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def yaw_from_quat(qx, qy, qz, qw) -> float:
    """Extract yaw (Z) from quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class SlamDebugPlotter(Node):
    def __init__(self):
        super().__init__("slam_debug_plotter")

        # ---- Parameters ----
        self.declare_parameter("topics.odom_in", "/slam/odom")
        self.declare_parameter("topics.map_in", "/slam/map_cones")
        self.declare_parameter("qos.best_effort", True)
        self.declare_parameter("qos.depth", 50)
        self.declare_parameter("plot.fixed_window", False)
        self.declare_parameter("plot.window_half_size", 50.0)

        gp = self.get_parameter
        odom_topic = str(gp("topics.odom_in").value)
        map_topic = str(gp("topics.map_in").value)
        best_effort = bool(gp("qos.best_effort").value)
        depth = int(gp("qos.depth").value)
        self.fixed_window = bool(gp("plot.fixed_window").value)
        self.window_half_size = float(gp("plot.window_half_size").value)

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=depth,
            reliability=(
                QoSReliabilityPolicy.BEST_EFFORT
                if best_effort
                else QoSReliabilityPolicy.RELIABLE
            ),
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # ---- Subscriptions ----
        self.create_subscription(Odometry, odom_topic, self.cb_odom, qos)
        self.create_subscription(ConeArrayWithCovariance, map_topic, self.cb_map, qos)

        # ---- State for plotting ----
        self.car_x = None
        self.car_y = None
        self.car_yaw = None  # radians

        self.blue_pts = []
        self.yellow_pts = []
        self.orange_pts = []
        self.big_pts = []

        self.data_lock = threading.Lock()

        # ---- Matplotlib figure ----
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.grid(True)
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_title("SLAM Debug Plotter (/slam/odom + /slam/map_cones)")

        self.get_logger().info(
            f"[slam_debug_plotter] Subscribing to odom={odom_topic}, map={map_topic}"
        )

    # --------------------- Callbacks ---------------------
    def cb_odom(self, msg: Odometry):
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

        with self.data_lock:
            self.car_x = x
            self.car_y = y
            self.car_yaw = yaw

    def cb_map(self, msg: ConeArrayWithCovariance):
        blue, yellow, orange, big = [], [], [], []

        for c in msg.blue_cones:
            blue.append((float(c.point.x), float(c.point.y)))
        for c in msg.yellow_cones:
            yellow.append((float(c.point.x), float(c.point.y)))
        for c in msg.orange_cones:
            orange.append((float(c.point.x), float(c.point.y)))
        for c in msg.big_orange_cones:
            big.append((float(c.point.x), float(c.point.y)))

        with self.data_lock:
            self.blue_pts = blue
            self.yellow_pts = yellow
            self.orange_pts = orange
            self.big_pts = big

    # --------------------- Plot update ---------------------
    def update_plot(self):
        with self.data_lock:
            car_x = self.car_x
            car_y = self.car_y
            car_yaw = self.car_yaw
            blue = list(self.blue_pts)
            yellow = list(self.yellow_pts)
            orange = list(self.orange_pts)
            big = list(self.big_pts)

        self.ax.clear()
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.grid(True)
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_title("SLAM Debug Plotter (/slam/odom + /slam/map_cones)")

        # Plot cones
        if blue:
            bx, by = zip(*blue)
            self.ax.scatter(bx, by, s=20, c="b", marker="o", label="blue cones")
        if yellow:
            yx, yy = zip(*yellow)
            self.ax.scatter(
                yx, yy, s=20, c="y", marker="o", edgecolors="k", label="yellow cones"
            )
        if orange:
            ox, oy = zip(*orange)
            self.ax.scatter(ox, oy, s=20, c="orange", marker="o", label="orange cones")
        if big:
            gx, gy = zip(*big)
            self.ax.scatter(
                gx, gy, s=50, c="magenta", marker="^", label="big orange cones"
            )

        # Plot current car pose as a triangle
        if (car_x is not None) and (car_y is not None) and (car_yaw is not None):
            # Define a small triangle in the car local frame
            # Tip forward, base behind.
            tri_len = 1.0   # visual length in meters (purely for display)
            tri_width = 0.5  # visual width in meters

            pts_local = np.array([
                [tri_len, 0.0],                    # tip
                [-tri_len * 0.5, -tri_width],      # rear-left
                [-tri_len * 0.5, +tri_width],      # rear-right
            ])

            c = math.cos(car_yaw)
            s = math.sin(car_yaw)
            R = np.array([[c, -s],
                          [s,  c]])

            pts_world = (R @ pts_local.T).T + np.array([car_x, car_y])

            car_tri = Polygon(
                pts_world,
                closed=True,
                facecolor="black",
                edgecolor="white",
                linewidth=1.0,
                label="car",
            )
            self.ax.add_patch(car_tri)

        # Dynamic or fixed window
        all_x = []
        all_y = []

        for lst in (blue, yellow, orange, big):
            if lst:
                xs, ys = zip(*lst)
                all_x.extend(xs)
                all_y.extend(ys)

        if (car_x is not None) and (car_y is not None):
            all_x.append(car_x)
            all_y.append(car_y)

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
    node = SlamDebugPlotter()
    plt.ion()  # interactive mode

    try:
        while rclpy.ok():
            # Process ROS callbacks in main thread
            rclpy.spin_once(node, timeout_sec=0.01)

            # Update plot and let GUI process events
            node.update_plot()
            plt.pause(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
