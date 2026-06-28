#!/usr/bin/env python3
"""
diag_listen.py — subscribe EXACTLY like the controller (best_effort, spinning node)
and report whether odom/path actually arrive + their values. This bypasses the
`ros2 topic echo` discovery quirk by living inside a real spinning node.

Run: python3 diag_listen.py   (sources already set)
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry, Path
import math, time

class Diag(Node):
    def __init__(self):
        super().__init__("diag_listen")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.odom_n = 0
        self.path_n = 0
        self.last_odom = None
        self.last_path_len = 0
        self.create_subscription(Odometry, "/slam/odom", self.cb_odom, qos)
        self.create_subscription(Odometry, "/ground_truth/odom", self.cb_gt, qos)
        self.create_subscription(Path, "/path_points", self.cb_path, qos)
        self.gt_n = 0
        self.last_gt = None
        self.t0 = time.time()
        self.create_timer(1.0, self.report)

    def cb_odom(self, m):
        self.odom_n += 1
        p = m.pose.pose.position
        self.last_odom = (p.x, p.y)

    def cb_gt(self, m):
        self.gt_n += 1
        p = m.pose.pose.position
        self.last_gt = (p.x, p.y)

    def cb_path(self, m):
        self.path_n += 1
        self.last_path_len = len(m.poses)

    def report(self):
        t = time.time() - self.t0
        o = f"{self.last_odom[0]:.2f},{self.last_odom[1]:.2f}" if self.last_odom else "NONE"
        g = f"{self.last_gt[0]:.2f},{self.last_gt[1]:.2f}" if self.last_gt else "NONE"
        print(f"[{t:4.1f}s] slam/odom: {self.odom_n} msgs last=({o}) | "
              f"gt/odom: {self.gt_n} msgs last=({g}) | "
              f"path: {self.path_n} msgs lastlen={self.last_path_len}", flush=True)
        if t > 8:
            rclpy.shutdown()

def main():
    rclpy.init()
    rclpy.spin(Diag())

if __name__ == "__main__":
    main()
