#!/usr/bin/env python3
import math
from collections import deque
import bisect
from typing import List, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
)
from rclpy.time import Time as RclTime

from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from eufs_msgs.msg import ConeArrayWithCovariance, ConeWithCovariance

from .slam_utils.pf_slam import ParticleFilterSLAM, wrap


def yaw_from_quat(qx, qy, qz, qw) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class PFSlamNode(Node):
    """
    PF-SLAM node (no visualisation).

      - Input odom: /slam/odom_raw (or param topics.odom_in)
      - Input cones: /perception/cones_fused (BASE/BODY frame, x,y,z,class_id)

      - Output SLAM odom: /slam/odom         (nav_msgs/Odometry)
      - Output map cones: /slam/map_cones    (eufs_msgs/ConeArrayWithCovariance)

    Logic:
      - Buffer odom poses with timestamps.
      - For each cones msg at time t_c:
          * interpolate odom pose (x_o, y_o, yaw_o) at t_c
          * use odom increment since last cones frame as PF motion
          * PF update with cones in body frame
          * publish best-particle pose as /slam/odom
          * publish best-particle landmarks as /slam/map_cones
    """

    def __init__(self):
        super().__init__("pf_slam_node")

        # ---- Parameters ----
        self.declare_parameter("topics.odom_in", "/slam/odom_raw")
        self.declare_parameter("topics.cones_in", "/perception/cones_fused")
        self.declare_parameter("topics.slam_odom_out", "/slam/odom")
        self.declare_parameter("topics.map_cones_out", "/slam/map_cones")

        self.declare_parameter("qos.best_effort", True)
        self.declare_parameter("qos.depth", 200)
        self.declare_parameter("odom_buffer_sec", 5.0)

        gp = self.get_parameter
        odom_topic = str(gp("topics.odom_in").value)
        cones_topic = str(gp("topics.cones_in").value)
        slam_odom_topic = str(gp("topics.slam_odom_out").value)
        map_cones_topic = str(gp("topics.map_cones_out").value)

        best_effort = bool(gp("qos.best_effort").value)
        depth = int(gp("qos.depth").value)
        self.odom_buffer_sec = float(gp("odom_buffer_sec").value)

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
        self.create_subscription(PointCloud2, cones_topic, self.cb_cones, qos)

        # ---- Publishers ----
        self.pub_slam_odom = self.create_publisher(Odometry, slam_odom_topic, 10)
        self.pub_map_cones = self.create_publisher(
            ConeArrayWithCovariance, map_cones_topic, 10
        )

        # ---- State ----

        # odom buffer: (t, x, y, yaw)
        self.odom_buf: deque = deque()

        # last interpolated odom pose at a cones frame
        self.slam_initialised = False
        self.last_cone_odom_pose: Tuple[float, float, float] = None

        # SLAM: PF core
        self.pf = ParticleFilterSLAM(
            num_particles=80,
            process_std_xy=0.03,
            process_std_yaw=0.01,
            meas_sigma_xy=0.20,
            birth_sigma_xy=0.40,
            gate_prob=0.997,
            resample_neff_ratio=0.5,
        )

        self.get_logger().info(
            f"[pf_slam] odom_in={odom_topic}, cones_in={cones_topic}, "
            f"slam_odom_out={slam_odom_topic}, map_cones_out={map_cones_topic}"
        )

    # ---------------------- Odom buffer helper ----------------------

    def _pose_at(self, t_query: float):
        """
        Interpolate odom pose (x,y,yaw) at time t_query using odom_buf.
        Returns (x,y,yaw) or None if no odom yet.
        """
        if not self.odom_buf:
            return None

        times = [it[0] for it in self.odom_buf]
        idx = bisect.bisect_left(times, t_query)

        if idx == 0:
            _, x0, y0, yaw0 = self.odom_buf[0]
            return x0, y0, yaw0
        if idx >= len(self.odom_buf):
            _, x1, y1, yaw1 = self.odom_buf[-1]
            return x1, y1, yaw1

        t0, x0, y0, yaw0 = self.odom_buf[idx - 1]
        t1, x1, y1, yaw1 = self.odom_buf[idx]

        if t1 == t0:
            return x0, y0, yaw0

        a = (t_query - t0) / (t1 - t0)
        x = x0 + a * (x1 - x0)
        y = y0 + a * (y1 - y0)
        dyaw = ((yaw1 - yaw0 + math.pi) % (2.0 * math.pi)) - math.pi
        yaw = yaw0 + a * dyaw
        return x, y, wrap(yaw)

    # ---------------------- Callbacks ----------------------

    def cb_odom(self, msg: Odometry):
        t = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

        # push into buffer
        self.odom_buf.append((t, x, y, yaw))
        tmin = t - self.odom_buffer_sec
        while self.odom_buf and self.odom_buf[0][0] < tmin:
            self.odom_buf.popleft()

    def cb_cones(self, msg: PointCloud2):
        # read cones in BASE/BODY frame (aligned with odom base)
        try:
            cones_body = list(
                pc2.read_points(
                    msg,
                    field_names=("x", "y", "z", "class_id"),
                    skip_nans=True,
                )
            )
        except Exception as e:
            self.get_logger().warn(f"[pf_slam] Error parsing cones PointCloud2: {e}")
            return

        if not cones_body:
            return

        t_c = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9

        pose_now = self._pose_at(t_c)
        if pose_now is None:
            # no odom yet
            return

        x_o, y_o, yaw_o = pose_now

        # --------- PF motion: odom increment between cone frames ---------
        if not self.slam_initialised or self.last_cone_odom_pose is None:
            # first time: initialise PF at odom pose
            self.pf.init_pose(x_o, y_o, yaw_o)
            self.slam_initialised = True
            self.last_cone_odom_pose = (x_o, y_o, yaw_o)
        else:
            x_prev, y_prev, yaw_prev = self.last_cone_odom_pose
            dx = x_o - x_prev
            dy = y_o - y_prev
            dyaw = wrap(yaw_o - yaw_prev)
            self.pf.predict((dx, dy, dyaw))
            self.last_cone_odom_pose = (x_o, y_o, yaw_o)

        # --------- PF measurement update ---------
        meas_body = [
            (float(bx), float(by), int(cls_id))
            for (bx, by, _bz, cls_id) in cones_body
        ]
        self.pf.update(meas_body)

        best = self.pf.get_best_particle()
        if best is None:
            return

        # Publish outputs
        self._publish_slam_odom(t_c, best)
        self._publish_map_cones(t_c, best)

    # ---------------------- Publishers ----------------------

    def _publish_slam_odom(self, t: float, best_particle):
        """
        Publish best-particle pose as nav_msgs/Odometry on topics.slam_odom_out.
        Twist is left zeroed for now.
        """
        od = Odometry()
        od.header.stamp = rclpy.time.Time(seconds=t).to_msg()
        od.header.frame_id = "map"
        od.child_frame_id = "base_link"

        od.pose.pose.position.x = float(best_particle.x)
        od.pose.pose.position.y = float(best_particle.y)
        od.pose.pose.position.z = 0.0

        half_yaw = 0.5 * float(best_particle.yaw)
        od.pose.pose.orientation.x = 0.0
        od.pose.pose.orientation.y = 0.0
        od.pose.pose.orientation.z = math.sin(half_yaw)
        od.pose.pose.orientation.w = math.cos(half_yaw)

        # Covariances / twist left at defaults (zero) for now
        self.pub_slam_odom.publish(od)

    def _publish_map_cones(self, t: float, best_particle):
        """
        Publish landmarks from best particle as ConeArrayWithCovariance in 'map' frame.
        Only 2D coordinates + colour are used; covariance left at defaults.
        cls mapping:
          0 -> blue_cones
          1 -> yellow_cones
          2 -> orange_cones
          3 -> big_orange_cones
        """
        msg = ConeArrayWithCovariance()
        msg.header.stamp = rclpy.time.Time(seconds=t).to_msg()
        msg.header.frame_id = "map"

        for lm in best_particle.landmarks:
            x = float(lm.mean[0])
            y = float(lm.mean[1])
            cls_id = int(lm.cls)

            cone = ConeWithCovariance()
            cone.point.x = x
            cone.point.y = y
            cone.point.z = 0.0
            # cone.covariance: leave default; we don't use it anywhere

            if cls_id == 0:
                msg.blue_cones.append(cone)
            elif cls_id == 1:
                msg.yellow_cones.append(cone)
            elif cls_id == 2:
                msg.orange_cones.append(cone)
            elif cls_id == 3:
                msg.big_orange_cones.append(cone)
            else:
                # If ConeArrayWithCovariance has unknown_color_cones, you can push here.
                # If not, just ignore or treat as blue.
                msg.unknown_color_cones.append(cone)

        self.pub_map_cones.publish(msg)


def main():
    rclpy.init()
    node = PFSlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
