#!/usr/bin/env python3
"""
pseudo_debug.py — LIVE view of the P14 pseudo centreline + MPPI rollouts.

Draws, in one matplotlib window (redraws at 10 Hz):
  blue / yellow dots   SLAM cone map            (/slam/map_cones)
  grey dots            PP's raw published path  (/path_points)
  cyan line            rolling buffer (what MPPI plans on) (/debug/buffer_path)
  orange dashed line   PSEUDO centreline feeds  (/debug/pseudo_path)
  thin fading lines    the 10 cheapest MPPI rollouts, cheapest = bright green
                       thick                    (/debug/rollouts, MarkerArray)
  black triangle       car pose                 (/slam/odom_raw)

Layers auto-clear 3 s after their topic goes silent, so a stale orange line
from a previous run can never masquerade as a live feed (restart NOT needed
between runs, but stopping the node is still cleanest).

Run (WSL, ROS env sourced):
    python3 src/controls/controls/debug/pseudo_debug.py
"""
import math
import time

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from nav_msgs.msg import Odometry, Path as PathMsg
from eufs_msgs.msg import ConeArrayWithCovariance
from visualization_msgs.msg import MarkerArray, Marker

STALE_S = 3.0        # a layer older than this is cleared
VIEW_R = 22.0        # window half-width following the car (m)


def _path_pts(msg):
    return [(p.pose.position.x, p.pose.position.y) for p in msg.poses]


class PseudoDebug(Node):
    def __init__(self):
        super().__init__("pseudo_debug")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE)

        self.blue, self.yellow = [], []
        self.car = (0.0, 0.0, 0.0)
        self.buffer, self.pseudo, self.pp = [], [], []
        self.rollouts = []                      # [(x,y) lists, best first]

        self.t_cones = 0.0
        self.t_buffer = 0.0
        self.t_pseudo = 0.0
        self.t_pp = 0.0
        self.t_rollouts = 0.0

        self.create_subscription(ConeArrayWithCovariance, "/slam/map_cones",
                                 self.cb_cones, qos)
        self.create_subscription(Odometry, "/slam/odom_raw", self.cb_odom, qos)
        self.create_subscription(PathMsg, "/debug/buffer_path", self.cb_buffer, qos)
        self.create_subscription(PathMsg, "/debug/pseudo_path", self.cb_pseudo, qos)
        self.create_subscription(PathMsg, "/path_points", self.cb_pp, qos)
        self.create_subscription(MarkerArray, "/debug/rollouts", self.cb_rollouts, qos)

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        self.fig.canvas.manager.set_window_title("P14 pseudo-path + MPPI rollouts")
        self.fig.patch.set_facecolor("white")
        self.timer = self.create_timer(0.1, self.redraw)

    # ---------------- callbacks -----------------------------
    def cb_cones(self, msg):
        self.blue = [(c.point.x, c.point.y) for c in msg.blue_cones]
        self.yellow = [(c.point.x, c.point.y) for c in msg.yellow_cones]
        self.t_cones = time.time()

    def cb_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.car = (p.x, p.y, math.atan2(siny, cosy))

    def cb_buffer(self, msg):
        self.buffer = _path_pts(msg)
        self.t_buffer = time.time()

    def cb_pseudo(self, msg):
        self.pseudo = _path_pts(msg)
        self.t_pseudo = time.time()

    def cb_pp(self, msg):
        self.pp = _path_pts(msg)
        self.t_pp = time.time()

    def cb_rollouts(self, msg):
        self.rollouts = []
        for m in msg.markers:
            if m.type == Marker.LINE_STRIP and m.points:
                self.rollouts.append([(p.x, p.y) for p in m.points])
        # markers arrive best-first (rank 0 = cheapest)
        self.t_rollouts = time.time()

    # ---------------- draw -----------------------------
    def redraw(self):
        now = time.time()
        if now - self.t_cones > STALE_S:
            self.blue, self.yellow = [], []
        if now - self.t_buffer > STALE_S:
            self.buffer = []
        if now - self.t_pseudo > STALE_S:
            self.pseudo = []
        if now - self.t_pp > STALE_S:
            self.pp = []
        if now - self.t_rollouts > STALE_S:
            self.rollouts = []

        ax = self.ax
        ax.clear()
        ax.set_facecolor("white")

        ax = self.ax
        ax.clear()
        ax.set_facecolor("white")

        # MPPI rollouts: rank 0 (cheapest) bright green thick, rest fading purple
        for rank, traj in enumerate(self.rollouts):
            if len(traj) < 2:
                continue
            xs, ys = zip(*traj)
            if rank == 0:
                ax.plot(xs, ys, "-", color=(0.0, 0.9, 0.25), lw=2.8, zorder=6,
                        label="best rollout (lowest cost)")
            else:
                ax.plot(xs, ys, "-", color=(0.58, 0.2, 0.85),
                        alpha=max(0.10, 0.45 - 0.045 * rank), lw=0.9, zorder=4)

        if self.buffer:
            bx, by = zip(*self.buffer)
            ax.plot(bx, by, "-", color=(0.0, 0.75, 1.0), lw=2.4, alpha=0.95,
                    zorder=5, label="buffer (MPPI plans on this)")
        if self.pseudo:
            sx, sy = zip(*self.pseudo)
            ax.plot(sx, sy, "--", color="darkorange", lw=2.6, zorder=7,
                    label="PSEUDO bridge")
            ax.scatter(sx, sy, c="darkorange", s=34, zorder=7, edgecolors="k",
                       linewidths=0.5)
        if self.pp:
            px, py = zip(*self.pp)
            ax.scatter(px, py, c=(0.75, 0.75, 0.75), s=16, zorder=5,
                       label="PP path (raw)")
        for cx, cy in self.blue:
            ax.add_patch(Circle((cx, cy), 0.28, facecolor="tab:blue",
                                edgecolor="navy", lw=0.7, zorder=3))
        for cx, cy in self.yellow:
            ax.add_patch(Circle((cx, cy), 0.28, facecolor="gold",
                                edgecolor="darkgoldenrod", lw=0.7, zorder=3))
        if self.blue or self.yellow:
            ax.plot([], [], "o", color="tab:blue", label="blue cones")
            ax.plot([], [], "o", color="gold", label="yellow cones")

        x, y, yaw = self.car
        ax.plot(x, y, marker=(3, 0, math.degrees(yaw)), markersize=14,
                color="black", markeredgecolor="white", markeredgewidth=1.2,
                zorder=8)
        ax.plot([], [], "^", color="black", label="car")

        ax.set_title("P14 pseudo-path + MPPI rollouts  "
                     "(green = cheapest rollout)", fontsize=11)
        ax.set_xlabel("E (m)")
        ax.set_ylabel("N (m)")
        ax.set_aspect("equal")
        ax.grid(True, color=(0.92, 0.92, 0.92), lw=0.7)
        for spine in ax.spines.values():
            spine.set_color((0.85, 0.85, 0.85))
        # follow the car
        ax.set_xlim(x - VIEW_R, x + VIEW_R)
        ax.set_ylim(y - VIEW_R, y + VIEW_R)
        leg = ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        if leg:
            leg.set_zorder(10)
        plt.pause(0.001)


def main():
    rclpy.init()
    node = PseudoDebug()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
