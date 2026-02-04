#!/usr/bin/env python3
# slam_odom_compare_plotter.py
#
# SLAM debug plotter (GT vs PRED vs SLAM) + optional cone-map overlay.
#
# Subscribes:
#   - predicted/primary odom (e.g. /slam/odom_raw)  -> "pred"
#   - slam-corrected odom (e.g. /slam/odom)         -> "slam"
#   - ground-truth odom (e.g. /ground_truth/odom)   -> "gt"
# Optional:
#   - map cones (e.g. /slam/map_cones) for scatter overlay on XY plot
#
# Plots (3 subplots):
#   1) XY trajectories
#   2) Speed vs time
#   3) Yaw vs time (unwrapped for readability)
#
import math
import threading
from typing import Optional, List, Tuple

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


def yaw_from_quat(x, y, z, w) -> float:
    """Return yaw (rad) from quaternion."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def ros_time_to_sec(stamp) -> float:
    """Convert builtin_interfaces/Time to float seconds."""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _unwrap(yaws: List[float]) -> List[float]:
    if not yaws:
        return []
    return list(np.unwrap(np.array(yaws, dtype=np.float64)))


class SlamOdomComparePlotter(Node):
    def __init__(self):
        super().__init__("slam_odom_compare_plotter")

        # ---- Parameters ----
        self.declare_parameter("topics.pred_odom_in", "/slam/odom_raw")
        self.declare_parameter("topics.slam_odom_in", "/slam/odom")
        self.declare_parameter("topics.gt_odom_in", "/ground_truth/odom")

        # Optional cone overlay on XY plot
        self.declare_parameter("topics.map_cones_in", "/slam/map_cones")
        self.declare_parameter("cones.enable", True)

        # QoS
        self.declare_parameter("qos.best_effort", True)
        self.declare_parameter("qos.depth", 50)

        # Plot control
        self.declare_parameter("plot.trail_max_points", 20000)
        self.declare_parameter("plot.fixed_window", False)
        self.declare_parameter("plot.window_half_size", 50.0)

        gp = self.get_parameter
        self.pred_topic = str(gp("topics.pred_odom_in").value)
        self.slam_topic = str(gp("topics.slam_odom_in").value)
        self.gt_topic = str(gp("topics.gt_odom_in").value)

        self.map_cones_topic = str(gp("topics.map_cones_in").value)
        self.cones_enable = bool(gp("cones.enable").value)

        best_effort = bool(gp("qos.best_effort").value)
        depth = int(gp("qos.depth").value)

        self.trail_max_points = int(gp("plot.trail_max_points").value)
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
        self.create_subscription(Odometry, self.pred_topic, self.cb_pred, qos)
        self.create_subscription(Odometry, self.slam_topic, self.cb_slam, qos)
        self.create_subscription(Odometry, self.gt_topic, self.cb_gt, qos)

        if self.cones_enable and self.map_cones_topic:
            self.create_subscription(ConeArrayWithCovariance, self.map_cones_topic, self.cb_cones, qos)

        # ---- State ----
        self.data_lock = threading.Lock()
        self.t0_abs: Optional[float] = None  # global reference time

        # Each stream keeps: traj_x, traj_y, last_x, last_y, t[], speed[], yaw[]
        self.pred = self._make_stream()
        self.slam = self._make_stream()
        self.gt = self._make_stream()

        # Cone overlay
        self.blue_pts: List[Tuple[float, float]] = []
        self.yellow_pts: List[Tuple[float, float]] = []
        self.orange_pts: List[Tuple[float, float]] = []
        self.big_pts: List[Tuple[float, float]] = []

        # ---- Matplotlib figure: 3 subplots ----
        self.fig, (self.ax_traj, self.ax_speed, self.ax_yaw) = plt.subplots(3, 1, figsize=(8, 11))

        self.ax_traj.set_aspect("equal", adjustable="datalim")
        self.ax_traj.grid(True)
        self.ax_traj.set_xlabel("X [m]")
        self.ax_traj.set_ylabel("Y [m]")
        self.ax_traj.set_title("XY Trajectory (GT vs PRED vs SLAM)")

        self.ax_speed.grid(True)
        self.ax_speed.set_xlabel("time [s]")
        self.ax_speed.set_ylabel("speed [m/s]")
        self.ax_speed.set_title("Speed vs time")

        self.ax_yaw.grid(True)
        self.ax_yaw.set_xlabel("time [s]")
        self.ax_yaw.set_ylabel("yaw [rad]")
        self.ax_yaw.set_title("Yaw vs time (unwrapped)")

        self.fig.tight_layout()

        self.get_logger().info(
            f"[slam_odom_compare_plotter] pred={self.pred_topic}, slam={self.slam_topic}, gt={self.gt_topic}, "
            f"cones={'ON' if self.cones_enable else 'OFF'} ({self.map_cones_topic})"
        )

    @staticmethod
    def _make_stream():
        return {
            "traj_x": [],
            "traj_y": [],
            "x": None,
            "y": None,
            "t": [],
            "speed": [],
            "yaw": [],
        }

    def _push_stream(self, stream, msg: Odometry):
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        speed = math.hypot(vx, vy)

        t_abs = ros_time_to_sec(msg.header.stamp)

        if self.t0_abs is None:
            self.t0_abs = t_abs
        t_rel = t_abs - self.t0_abs

        stream["x"] = x
        stream["y"] = y
        stream["traj_x"].append(x)
        stream["traj_y"].append(y)

        stream["t"].append(t_rel)
        stream["speed"].append(speed)
        stream["yaw"].append(yaw)

        # trim
        n = self.trail_max_points
        if len(stream["traj_x"]) > n:
            stream["traj_x"] = stream["traj_x"][-n:]
            stream["traj_y"] = stream["traj_y"][-n:]
        if len(stream["t"]) > n:
            stream["t"] = stream["t"][-n:]
            stream["speed"] = stream["speed"][-n:]
            stream["yaw"] = stream["yaw"][-n:]

    # --------------------- Callbacks ---------------------

    def cb_pred(self, msg: Odometry):
        with self.data_lock:
            self._push_stream(self.pred, msg)

    def cb_slam(self, msg: Odometry):
        with self.data_lock:
            self._push_stream(self.slam, msg)

    def cb_gt(self, msg: Odometry):
        with self.data_lock:
            self._push_stream(self.gt, msg)

    def cb_cones(self, msg: ConeArrayWithCovariance):
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
            pred = {k: list(v) if isinstance(v, list) else v for k, v in self.pred.items()}
            slam = {k: list(v) if isinstance(v, list) else v for k, v in self.slam.items()}
            gt = {k: list(v) if isinstance(v, list) else v for k, v in self.gt.items()}

            blue = list(self.blue_pts)
            yellow = list(self.yellow_pts)
            orange = list(self.orange_pts)
            big = list(self.big_pts)

        # ---- XY TRAJECTORY ----
        self.ax_traj.clear()
        self.ax_traj.set_aspect("equal", adjustable="datalim")
        self.ax_traj.grid(True)
        self.ax_traj.set_xlabel("X [m]")
        self.ax_traj.set_ylabel("Y [m]")
        self.ax_traj.set_title("XY Trajectory (GT vs PRED vs SLAM)")

        # Optional cone overlay
        if self.cones_enable:
            if blue:
                bx, by = zip(*blue)
                self.ax_traj.scatter(bx, by, s=10, marker="o", label="cones blue")
            if yellow:
                yx, yy = zip(*yellow)
                self.ax_traj.scatter(yx, yy, s=10, marker="o", label="cones yellow")
            if orange:
                ox, oy = zip(*orange)
                self.ax_traj.scatter(ox, oy, s=10, marker="o", label="cones orange")
            if big:
                gx, gy = zip(*big)
                self.ax_traj.scatter(gx, gy, s=25, marker="^", label="cones big")

        # Trajectories
        if pred["traj_x"]:
            self.ax_traj.plot(pred["traj_x"], pred["traj_y"], "-", linewidth=1.0, label="PRED traj")
        if pred["x"] is not None:
            self.ax_traj.plot(pred["x"], pred["y"], "o", markersize=4, label="PRED pose")

        if slam["traj_x"]:
            self.ax_traj.plot(slam["traj_x"], slam["traj_y"], "-", linewidth=1.0, label="SLAM traj")
        if slam["x"] is not None:
            self.ax_traj.plot(slam["x"], slam["y"], "s", markersize=4, label="SLAM pose")

        if gt["traj_x"]:
            self.ax_traj.plot(gt["traj_x"], gt["traj_y"], "--", linewidth=1.0, label="GT traj")
        if gt["x"] is not None:
            self.ax_traj.plot(gt["x"], gt["y"], "d", markersize=4, label="GT pose")

        # Windowing (center on SLAM if available, else PRED, else GT)
        cx = slam["x"] if slam["x"] is not None else (pred["x"] if pred["x"] is not None else gt["x"])
        cy = slam["y"] if slam["y"] is not None else (pred["y"] if pred["y"] is not None else gt["y"])

        all_x, all_y = [], []
        for s in (pred, slam, gt):
            all_x.extend(s["traj_x"])
            all_y.extend(s["traj_y"])
            if s["x"] is not None:
                all_x.append(s["x"])
                all_y.append(s["y"])
        if self.cones_enable:
            for lst in (blue, yellow, orange, big):
                if lst:
                    xs, ys = zip(*lst)
                    all_x.extend(xs)
                    all_y.extend(ys)

        if self.fixed_window and (cx is not None and cy is not None):
            L = self.window_half_size
            self.ax_traj.set_xlim(cx - L, cx + L)
            self.ax_traj.set_ylim(cy - L, cy + L)
        else:
            if all_x and all_y:
                xmin, xmax = min(all_x), max(all_x)
                ymin, ymax = min(all_y), max(all_y)
                pad = 2.0
                if xmax - xmin < 1e-3:
                    xmax = xmin + 1.0
                if ymax - ymin < 1e-3:
                    ymax = ymin + 1.0
                self.ax_traj.set_xlim(xmin - pad, xmax + pad)
                self.ax_traj.set_ylim(ymin - pad, ymax + pad)

        self.ax_traj.legend(loc="upper right", fontsize=8)

        # ---- SPEED VS TIME ----
        self.ax_speed.clear()
        self.ax_speed.grid(True)
        self.ax_speed.set_xlabel("time [s]")
        self.ax_speed.set_ylabel("speed [m/s]")
        self.ax_speed.set_title("Speed vs time")

        if pred["t"] and pred["speed"]:
            self.ax_speed.plot(pred["t"], pred["speed"], "-", linewidth=1.0, label="PRED speed")
        if slam["t"] and slam["speed"]:
            self.ax_speed.plot(slam["t"], slam["speed"], "-", linewidth=1.0, label="SLAM speed")
        if gt["t"] and gt["speed"]:
            self.ax_speed.plot(gt["t"], gt["speed"], "--", linewidth=1.0, label="GT speed")

        self.ax_speed.legend(loc="upper right", fontsize=8)

        # ---- YAW VS TIME (unwrapped) ----
        self.ax_yaw.clear()
        self.ax_yaw.grid(True)
        self.ax_yaw.set_xlabel("time [s]")
        self.ax_yaw.set_ylabel("yaw [rad]")
        self.ax_yaw.set_title("Yaw vs time (unwrapped)")

        pred_yaw_u = _unwrap(pred["yaw"])
        slam_yaw_u = _unwrap(slam["yaw"])
        gt_yaw_u = _unwrap(gt["yaw"])

        if pred["t"] and pred_yaw_u:
            self.ax_yaw.plot(pred["t"], pred_yaw_u, "-", linewidth=1.0, label="PRED yaw")
        if slam["t"] and slam_yaw_u:
            self.ax_yaw.plot(slam["t"], slam_yaw_u, "-", linewidth=1.0, label="SLAM yaw")
        if gt["t"] and gt_yaw_u:
            self.ax_yaw.plot(gt["t"], gt_yaw_u, "--", linewidth=1.0, label="GT yaw")

        self.ax_yaw.legend(loc="upper right", fontsize=8)

        self.fig.tight_layout()


def main():
    rclpy.init()
    node = SlamOdomComparePlotter()
    plt.ion()  # interactive mode

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            node.update_plot()
            plt.pause(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
