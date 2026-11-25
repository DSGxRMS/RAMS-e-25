#!/usr/bin/env python3
"""
slam_diag.py — flight recorder for the GT-vs-SLAM question.

Pure observer: subscribes to the pose/cone topics and writes ONE CSV row
every 0.1 s with the latest known values. Zero effect on the car.

Columns:
  wall_t            wall-clock seconds (same clock as mppi_trace.csv 'time')
  gt_x/y/yaw        /ground_truth/odom        (the simulator's truth)
  raw_x/y/yaw       /slam/odom_raw           (pred_node relay = GT today)
  slam_x/y/yaw      /slam/odom               (SLAM's estimated pose)
  split_gt_slam_m   |GT pose - SLAM pose|    (the P16 drift metric)
  split_raw_slam_m  |odom_raw - SLAM pose|   (~0 with P17 passthrough)
  map_blue/yellow/orange/big   cone counts in /slam/map_cones
  fused_cones       cone count in /perception/cones_fused (raw GT, body frame)
  fov_blue/yellow   map cones inside the PP/controls wedge, using odom_raw
                    pose -- replicates what controls actually 'sees'
  gt/raw/slam/cones_age_s   seconds since each topic's last message
                    (large = that voice went silent)

Output: slam_diag.csv in the working directory (overwritten each launch;
redirect with  SLAM_DIAG_CSV=/path/to/file.csv)

Run (WSL, ROS env sourced, any time -- ideally start before the run):
    python3 src/controls/controls/debug/slam_diag.py
Stop with Ctrl+C (file closes cleanly).
"""
import math
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from nav_msgs.msg import Odometry
from eufs_msgs.msg import ConeArrayWithCovariance
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

# Same wedge geometry as pp_node_skidpad / ros_connect
FOV_VERTEX = -5.0
FOV_RADIUS = 30.0
FOV_HALF = math.radians(45.0)

HEADER = ("wall_t,gt_x,gt_y,gt_yaw,raw_x,raw_y,raw_yaw,slam_x,slam_y,slam_yaw,"
          "split_gt_slam_m,split_raw_slam_m,map_blue,map_yellow,map_orange,map_big,"
          "fused_cones,fov_blue,fov_yellow,gt_age_s,raw_age_s,slam_age_s,cones_age_s")


class SlamDiag(Node):
    def __init__(self):
        super().__init__("slam_diag")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE)

        self.gt = None          # (x, y, yaw)
        self.raw = None
        self.slam = None
        self.map_blue, self.map_yellow, self.map_orange, self.map_big = [], [], [], []
        self.fused_count = 0
        self.t_gt = self.t_raw = self.t_slam = self.t_cones = None

        self.create_subscription(Odometry, "/ground_truth/odom", self.cb_gt, qos)
        self.create_subscription(Odometry, "/slam/odom_raw", self.cb_raw, qos)
        self.create_subscription(Odometry, "/slam/odom", self.cb_slam, qos)
        self.create_subscription(ConeArrayWithCovariance, "/slam/map_cones", self.cb_map, qos)
        self.create_subscription(PointCloud2, "/perception/cones_fused", self.cb_fused, qos)

        out = os.environ.get("SLAM_DIAG_CSV", "slam_diag.csv")
        self.f = open(out, "w")
        self.f.write(HEADER + "\n")
        self.get_logger().info(f"[slam_diag] recording -> {out}")

        self.timer = self.create_timer(0.1, self.tick)

    # ---------------- callbacks: store latest only ----------------
    @staticmethod
    def _pose(msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return (float(p.x), float(p.y), math.atan2(siny, cosy))

    def cb_gt(self, msg):
        self.gt = self._pose(msg)
        self.t_gt = time.time()

    def cb_raw(self, msg):
        self.raw = self._pose(msg)
        self.t_raw = time.time()

    def cb_slam(self, msg):
        self.slam = self._pose(msg)
        self.t_slam = time.time()

    def cb_map(self, msg):
        self.map_blue = [(c.point.x, c.point.y) for c in msg.blue_cones]
        self.map_yellow = [(c.point.x, c.point.y) for c in msg.yellow_cones]
        self.map_orange = [(c.point.x, c.point.y) for c in msg.orange_cones]
        self.map_big = [(c.point.x, c.point.y) for c in msg.big_orange_cones]
        self.t_cones = time.time()

    def cb_fused(self, msg):
        try:
            self.fused_count = sum(1 for _ in pc2.read_points(
                msg, field_names=("x", "y", "z", "class_id"), skip_nans=True))
        except Exception:
            self.fused_count = -1
        self.t_cones = self.t_cones or time.time()

    # ---------------- 10 Hz writer ----------------
    def tick(self):
        now = time.time()

        def age(t):
            return (now - t) if t is not None else -1.0

        def dist(a, b):
            if a is None or b is None:
                return -1.0
            return math.hypot(a[0] - b[0], a[1] - b[1])

        gt, raw, slam = self.gt, self.raw, self.slam
        gtx, gty, gtyaw = gt if gt else (-1, -1, -1)
        rax, ray, rayaw = raw if raw else (-1, -1, -1)
        slx, sly, slyaw = slam if slam else (-1, -1, -1)

        # wedge counts with the pose controls actually uses (odom_raw)
        fb = fy = 0
        if raw:
            cs, sn = math.cos(raw[2]), math.sin(raw[2])
            for pts, cnt in ((self.map_blue, "b"), (self.map_yellow, "y")):
                for (px, py) in pts:
                    dx, dy = px - raw[0], py - raw[1]
                    lx = cs * dx + sn * dy
                    ly = -sn * dx + cs * dy
                    vx, vy = lx - FOV_VERTEX, ly
                    r2 = vx * vx + vy * vy
                    if r2 < 1e-6 or r2 > FOV_RADIUS ** 2:
                        continue
                    if abs(math.atan2(vy, vx)) > FOV_HALF:
                        continue
                    if cnt == "b":
                        fb += 1
                    else:
                        fy += 1

        row = (f"{now:.3f},{gtx:.3f},{gty:.3f},{gtyaw:.4f},"
               f"{rax:.3f},{ray:.3f},{rayaw:.4f},"
               f"{slx:.3f},{sly:.3f},{slyaw:.4f},"
               f"{dist(gt, slam):.3f},{dist(raw, slam):.3f},"
               f"{len(self.map_blue)},{len(self.map_yellow)},"
               f"{len(self.map_orange)},{len(self.map_big)},"
               f"{self.fused_count},{fb},{fy},"
               f"{age(self.t_gt):.2f},{age(self.t_raw):.2f},"
               f"{age(self.t_slam):.2f},{age(self.t_cones):.2f}")
        self.f.write(row + "\n")
        self.f.flush()


def main():
    rclpy.init()
    node = SlamDiag()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.f.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
