#!/usr/bin/env python3
# controls_node.py

import time
import math
import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry, Path
from ackermann_msgs.msg import AckermannDriveStamped

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from utils.control_utils import (
    PID,
    compute_signed_curvature,
    local_closest_index,
    preprocess_path,
    calc_lookahead_point,
    pure_pursuit_steer,
)

MAX_VELOCITY = 2.0
VEL_LIMIT_FACTOR = 0.3
LOOK_AHEAD_UPDATE_INTERVAL = 1.2
ROUTE_IS_LOOP = False
STOP_SPEED_THRESHOLD = 0.1
WHEELBASE_M = 1.5
MAX_STEER_RAD = math.pi / 2


def yaw_from_quat(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class ControlsNode(Node):
    def __init__(self):
        super().__init__("controls_node")

        self.odom_topic = "/slam/odom"
        self.path_topic = "/path_points"
        self.cmd_topic = "/cmd"

        # ---- Best-effort QoS for subscriptions ----
        sub_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # ---- State ----
        self.cx = 0.0
        self.cy = 0.0
        self.yaw = 0.0
        self.speed = 0.0
        self.have_odom = False
        self.latest_path = None

        # ---- Sub / Pub ----
        self.create_subscription(Odometry, self.odom_topic, self._odom_cb, sub_qos)
        self.create_subscription(Path, self.path_topic, self._path_cb, sub_qos)

        # Keep publisher default unless your /cmd expects best-effort too
        self.cmd_pub = self.create_publisher(AckermannDriveStamped, self.cmd_topic, 10)

        self.th_pid = PID(3.2, 0, 0)

        self.last_lookahead_update = 0.0
        self.last_control_time = time.perf_counter()
        self.cur_idx = 0

        self.look_ahead_x = None
        self.look_ahead_y = None
        self.look_ahead_dist = 0.0
        self.look_ahead_idx = 0

        self.timer = self.create_timer(0.05, self._step)

    def _odom_cb(self, msg: Odometry):
        self.cx = float(msg.pose.pose.position.x)
        self.cy = float(msg.pose.pose.position.y)
        self.yaw = yaw_from_quat(msg.pose.pose.orientation)

        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        self.speed = math.sqrt(vx * vx + vy * vy)

        self.have_odom = True

    def _path_cb(self, msg: Path):
        self.latest_path = msg

    def _get_path_points(self):
        if self.latest_path is None:
            return np.array([]), np.array([])
        pts = self.latest_path.poses
        if len(pts) == 0:
            return np.array([]), np.array([])

        route_x = np.array([p.pose.position.x for p in pts], dtype=float)
        route_y = np.array([p.pose.position.y for p in pts], dtype=float)
        return route_x, route_y

    def _publish_cmd(self, steering_rad, target_speed, accel_cmd):
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = float(steering_rad)
        msg.drive.speed = float(target_speed)
        msg.drive.acceleration = float(accel_cmd)
        self.cmd_pub.publish(msg)

    def _step(self):
        if not self.have_odom:
            return

        route_x, route_y = self._get_path_points()
        if route_x.size < 3:
            return

        now = time.perf_counter()
        dt = now - self.last_control_time
        self.last_control_time = now

        curve = compute_signed_curvature(route_x, route_y)

        self.cur_idx = local_closest_index((self.cx, self.cy), route_x, route_y, self.cur_idx, loop=ROUTE_IS_LOOP)
        if np.ndim(self.cur_idx) > 0:
            self.cur_idx = self.cur_idx[0]
        self.cur_idx = int(self.cur_idx)

        if now - self.last_lookahead_update >= LOOK_AHEAD_UPDATE_INTERVAL:
            self.last_lookahead_update = now
            _, _, s, route_len = preprocess_path(route_x, route_y, loop=ROUTE_IS_LOOP)
            self.look_ahead_x, self.look_ahead_y, self.look_ahead_dist, self.look_ahead_idx = calc_lookahead_point(
                self.speed, route_x, route_y, self.cur_idx, s, route_len, loop=ROUTE_IS_LOOP
            )

        if self.look_ahead_x is not None:
            steering_norm = pure_pursuit_steer(
                (self.cx, self.cy), self.yaw, self.look_ahead_x, self.look_ahead_y, self.look_ahead_dist
            )
        else:
            steering_norm = 0.0

        actual_steering_rad = steering_norm * MAX_STEER_RAD

        safe_idx = min(self.cur_idx, len(curve) - 1)
        target_speed = MAX_VELOCITY * (1 - VEL_LIMIT_FACTOR * abs(curve[safe_idx]))

        speed_error = target_speed - self.speed
        accel_cmd = self.th_pid.update(speed_error, dt=dt)
        accel_cmd = max(-3.0, min(2.0, accel_cmd))

        self._publish_cmd(actual_steering_rad, target_speed, accel_cmd)

        if (not ROUTE_IS_LOOP) and self.cur_idx >= len(route_x) - 1 and self.speed < STOP_SPEED_THRESHOLD:
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = ControlsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
