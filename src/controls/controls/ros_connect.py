# control_v2/ros_connect.py

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from nav_msgs.msg import Odometry, Path
from ackermann_msgs.msg import AckermannDriveStamped
import math


class ROSInterface(Node):
    def __init__(self, odom_topic="/slam/odom", cmd_topic="/cmd", path_topic="/path_points"):
        super().__init__('ros_interface')

        self.cx, self.cy, self.yaw, self.speed = 0.0, 0.0, 0.0, 0.0
        self.have_odom = False

        # Path buffering / gating
        self.latest_path = None               # last received Path message
        self._last_good_path = []            # last "usable" path as list[(x,y)]
        self._min_path_points = 5            # ignore tiny paths (fixes IndexError in controller)

        # Best Effort QoS (typical for high-rate sensor-ish streams)
        self.qos_best_effort = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        self.create_subscription(Odometry, odom_topic, self._odom_cb, self.qos_best_effort)
        self.create_subscription(Path, path_topic, self._path_cb, self.qos_best_effort)

        self.pub = self.create_publisher(AckermannDriveStamped, cmd_topic, 10)

    def _odom_cb(self, msg: Odometry):
        self.cx = float(msg.pose.pose.position.x)
        self.cy = float(msg.pose.pose.position.y)

        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

        vx, vy = float(msg.twist.twist.linear.x), float(msg.twist.twist.linear.y)
        self.speed = math.hypot(vx, vy)

        self.have_odom = True

    def _path_cb(self, msg: Path):
        # Always store the latest message, but only promote to "usable" if it has enough points.
        self.latest_path = msg

        pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        if len(pts) >= self._min_path_points:
            self._last_good_path = pts

    def get_state(self):
        """Returns (x, y, yaw, speed, have_odom)"""
        return (self.cx, self.cy, self.yaw, self.speed, self.have_odom)

    def get_path(self):
        """
        Returns list of (x, y) path points in global coordinates.

        Fix for "2-point path" crashes:
        - If the latest path is too short (< _min_path_points), ignore it and return the last good path.
        - If no good path has ever been received, return [] so the control loop waits.
        """
        # Prefer last known good path
        if self._last_good_path:
            return self._last_good_path

        # Fallback: if we somehow never had a good path, return empty to make controller wait
        return []

    def send_command(self, steering, speed=None, accel=None):
        # Removed per-frame logger spam (was slowing sim / lowering RTF)
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = float(steering)
        if accel is not None:
            msg.drive.acceleration = float(accel)
        if speed is not None:
            msg.drive.speed = float(speed)

        self.pub.publish(msg)
