#!/usr/bin/env python3
"""
gt_vs_slam.py — the decisive diagnostic. Run this WHILE the sim is running
(no controller needed, or with it). It prints, 2x/sec:

  GT   : true car position, true heading, true body velocity (Vx fwd, Vy lateral)
  SLAM : what /slam/odom reports for position + heading

This answers ONE question: is the car physically crabbing sideways (GT Vy large,
GT yaw changing), or is SLAM reporting a frozen/wrong yaw while the car drives normally?

Run:
  source /opt/ros/galactic/setup.bash
  source ~/eufs_ws/install/setup.bash
  source /mnt/d/RAMS-e-25/install/setup.bash
  python3 /mnt/d/RAMS-e-25/src/controls/controls/gt_vs_slam.py
"""
import math, time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry, Path


def yaw_of(m):
    o = m.pose.pose.orientation
    return math.degrees(math.atan2(2*(o.w*o.z + o.x*o.y), 1 - 2*(o.y*o.y + o.z*o.z)))


class Chk(Node):
    def __init__(self):
        super().__init__("gt_vs_slam")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.gt = None
        self.slam = None
        self.raw = None
        self.path0 = None   # first point of /path_points
        self.pathN = 0
        self.create_subscription(Odometry, "/ground_truth/odom", self.cb_gt, qos)
        self.create_subscription(Odometry, "/slam/odom", self.cb_slam, qos)
        self.create_subscription(Odometry, "/slam/odom_raw", self.cb_raw, qos)
        self.create_subscription(Path, "/path_points", self.cb_path, qos)
        self.create_timer(0.5, self.report)

    def cb_gt(self, m):
        p = m.pose.pose.position; t = m.twist.twist.linear
        self.gt = (p.x, p.y, yaw_of(m), t.x, t.y)

    def cb_slam(self, m):
        p = m.pose.pose.position
        self.slam = (p.x, p.y, yaw_of(m))

    def cb_raw(self, m):
        p = m.pose.pose.position
        self.raw = (p.x, p.y, yaw_of(m))

    def cb_path(self, m):
        self.pathN = len(m.poses)
        if m.poses:
            p0 = m.poses[0].pose.position
            pl = m.poses[-1].pose.position
            self.path0 = (p0.x, p0.y, pl.x, pl.y)

    def report(self):
        g = self.gt
        def fmt(v): return f"({v[0]:6.2f},{v[1]:6.2f})" if v else "  NONE  "
        gtxt = f"GT{fmt(g)} Vx={g[3]:5.2f}" if g else "GT NONE"
        rtxt = f"raw{fmt(self.raw)}" if self.raw else "raw NONE"
        stxt = f"slam{fmt(self.slam)}" if self.slam else "slam NONE"
        if self.path0:
            ptxt = f"path[0]=({self.path0[0]:6.2f},{self.path0[1]:6.2f}) path[-1]=({self.path0[2]:6.2f},{self.path0[3]:6.2f}) N={self.pathN}"
        else:
            ptxt = "path NONE"
        print(f"{gtxt} || {rtxt} || {stxt} || {ptxt}", flush=True)


def main():
    rclpy.init()
    rclpy.spin(Chk())


if __name__ == "__main__":
    main()
