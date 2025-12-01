#!/usr/bin/env python3
# imu_odom_debug_plotter.py
#
# Debug visualizer for odometry:
#  - Subscribes to primary odom (e.g., /slam/odom_raw)
#  - Subscribes to reference odom (e.g., /ground_truth/odom)
#  - Plots:
#       * XY trajectories
#       * Speed vs time
#       * Yaw vs time
#
import math
import threading  # still used for locks, not for GUI

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
)
from nav_msgs.msg import Odometry

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


class ImuOdomDebugPlotter(Node):
    def __init__(self):
        super().__init__("imu_odom_debug_plotter")

        # ---- Parameters ----
        self.declare_parameter("topics.odom_in", "/slam/odom_raw")
        self.declare_parameter("topics.ref_odom_in", "/ground_truth/odom")
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
            reliability=(
                QoSReliabilityPolicy.BEST_EFFORT
                if best_effort
                else QoSReliabilityPolicy.RELIABLE
            ),
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # ---- Subscriptions ----
        self.create_subscription(Odometry, odom_topic, self.cb_odom, qos)
        self.ref_enabled = len(ref_odom_topic) > 0
        if self.ref_enabled:
            self.create_subscription(Odometry, ref_odom_topic, self.cb_ref_odom, qos)

        # ---- State for plotting (primary) ----
        self.traj_x = []
        self.traj_y = []
        self.car_x = None
        self.car_y = None

        self.t = []          # time (s, relative)
        self.speed = []      # m/s
        self.yaw = []        # rad
        self.t0 = None       # first timestamp seen (for relative time)

        # ---- State for plotting (reference) ----
        self.ref_traj_x = []
        self.ref_traj_y = []
        self.ref_x = None
        self.ref_y = None

        self.ref_t = []
        self.ref_speed = []
        self.ref_yaw = []
        self.ref_t0 = None

        self.data_lock = threading.Lock()

        # ---- Matplotlib figure: 3 subplots ----
        self.fig, (self.ax_traj, self.ax_speed, self.ax_yaw) = plt.subplots(
            3, 1, figsize=(7, 10)
        )

        self.ax_traj.set_aspect("equal", adjustable="datalim")
        self.ax_traj.grid(True)
        self.ax_traj.set_xlabel("X [m]")
        self.ax_traj.set_ylabel("Y [m]")
        self.ax_traj.set_title("XY Trajectory")

        self.ax_speed.grid(True)
        self.ax_speed.set_xlabel("time [s]")
        self.ax_speed.set_ylabel("speed [m/s]")
        self.ax_speed.set_title("Speed vs time")

        self.ax_yaw.grid(True)
        self.ax_yaw.set_xlabel("time [s]")
        self.ax_yaw.set_ylabel("yaw [rad]")
        self.ax_yaw.set_title("Yaw vs time")

        self.fig.tight_layout()

        self.get_logger().info(
            f"[imu_odom_debug_plotter] odom={odom_topic}, "
            f"ref_odom={ref_odom_topic or 'NONE'}"
        )

    # --------------------- Callbacks ---------------------
    def cb_odom(self, msg: Odometry):
        # Pose
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

        # Speed from twist
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        speed = math.hypot(vx, vy)

        # Time
        t_abs = ros_time_to_sec(msg.header.stamp)
        with self.data_lock:
            if self.t0 is None:
                self.t0 = t_abs
            t_rel = t_abs - self.t0

            # XY
            self.car_x = x
            self.car_y = y
            self.traj_x.append(x)
            self.traj_y.append(y)

            # time series
            self.t.append(t_rel)
            self.speed.append(speed)
            self.yaw.append(yaw)

            # trim
            if len(self.traj_x) > self.trail_max_points:
                self.traj_x = self.traj_x[-self.trail_max_points:]
                self.traj_y = self.traj_y[-self.trail_max_points:]
            if len(self.t) > self.trail_max_points:
                self.t = self.t[-self.trail_max_points:]
                self.speed = self.speed[-self.trail_max_points:]
                self.yaw = self.yaw[-self.trail_max_points:]

    def cb_ref_odom(self, msg: Odometry):
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        speed = math.hypot(vx, vy)

        t_abs = ros_time_to_sec(msg.header.stamp)
        with self.data_lock:
            if self.ref_t0 is None:
                self.ref_t0 = t_abs
            t_rel = t_abs - self.ref_t0

            self.ref_x = x
            self.ref_y = y
            self.ref_traj_x.append(x)
            self.ref_traj_y.append(y)

            self.ref_t.append(t_rel)
            self.ref_speed.append(speed)
            self.ref_yaw.append(yaw)

            if len(self.ref_traj_x) > self.trail_max_points:
                self.ref_traj_x = self.ref_traj_x[-self.trail_max_points:]
                self.ref_traj_y = self.ref_traj_y[-self.trail_max_points:]
            if len(self.ref_t) > self.trail_max_points:
                self.ref_t = self.ref_t[-self.trail_max_points:]
                self.ref_speed = self.ref_speed[-self.trail_max_points:]
                self.ref_yaw = self.ref_yaw[-self.trail_max_points:]

    # --------------------- Plot update ---------------------
    def update_plot(self):
        with self.data_lock:
            traj_x = list(self.traj_x)
            traj_y = list(self.traj_y)
            car_x = self.car_x
            car_y = self.car_y

            t = list(self.t)
            speed = list(self.speed)
            yaw = list(self.yaw)

            ref_traj_x = list(self.ref_traj_x)
            ref_traj_y = list(self.ref_traj_y)
            ref_x = self.ref_x
            ref_y = self.ref_y

            ref_t = list(self.ref_t)
            ref_speed = list(self.ref_speed)
            ref_yaw = list(self.ref_yaw)

        # ---- XY TRAJECTORY ----
        self.ax_traj.clear()
        self.ax_traj.set_aspect("equal", adjustable="datalim")
        self.ax_traj.grid(True)
        self.ax_traj.set_xlabel("X [m]")
        self.ax_traj.set_ylabel("Y [m]")
        self.ax_traj.set_title("XY Trajectory")

        if traj_x and traj_y:
            self.ax_traj.plot(traj_x, traj_y, "-", linewidth=1.0, label="odom_raw traj")
        if car_x is not None and car_y is not None:
            self.ax_traj.plot(car_x, car_y, "o", markersize=5, label="odom_raw pose")

        if ref_traj_x and ref_traj_y:
            self.ax_traj.plot(ref_traj_x, ref_traj_y, "--", linewidth=1.0, label="GT traj")
        if ref_x is not None and ref_y is not None:
            self.ax_traj.plot(ref_x, ref_y, "s", markersize=5, label="GT pose")

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
            self.ax_traj.set_xlim(car_x - L, car_x + L)
            self.ax_traj.set_ylim(car_y - L, car_y + L)
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

        if t and speed:
            self.ax_speed.plot(t, speed, "-", linewidth=1.0, label="odom_raw speed")
        if ref_t and ref_speed:
            self.ax_speed.plot(ref_t, ref_speed, "--", linewidth=1.0, label="GT speed")

        self.ax_speed.legend(loc="upper right", fontsize=8)

        # ---- YAW VS TIME ----
        self.ax_yaw.clear()
        self.ax_yaw.grid(True)
        self.ax_yaw.set_xlabel("time [s]")
        self.ax_yaw.set_ylabel("yaw [rad]")
        self.ax_yaw.set_title("Yaw vs time")

        if t and yaw:
            self.ax_yaw.plot(t, yaw, "-", linewidth=1.0, label="odom_raw yaw")
        if ref_t and ref_yaw:
            self.ax_yaw.plot(ref_t, ref_yaw, "--", linewidth=1.0, label="GT yaw")

        self.ax_yaw.legend(loc="upper right", fontsize=8)

        self.fig.tight_layout()


def main():
    rclpy.init()
    node = ImuOdomDebugPlotter()
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
