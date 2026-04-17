#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import PointCloud2, PointField
from eufs_msgs.msg import ConeArrayWithCovariance


class GroundTruthConeAdapter(Node):
    """
    Ground-truth cone adapter.

    Input:
      - /ground_truth/cones   (ConeArrayWithCovariance)

    Output:
      - /perception/cones_fused (PointCloud2)
        fields: x, y, z, class_id

    Keeps the same downstream format as the old fusion node, but without
    depending on camera/lidar feeds.
    """

    def __init__(self):
        super().__init__("ground_truth_cone_adapter")

        # Topics
        self.declare_parameter("ground_truth_topic", "/ground_truth/cones")
        self.declare_parameter("output_topic", "/perception/cones_fused")
        self.declare_parameter("output_frame", "")

        # IMPORTANT:
        # Set these IDs to whatever your downstream stack expects.
        # These defaults are a common convention, but if your RViz / planner /
        # mapper uses another one, change only these five params.
        self.declare_parameter("class_ids.yellow", 0)
        self.declare_parameter("class_ids.blue", 1)
        self.declare_parameter("class_ids.orange", 2)
        self.declare_parameter("class_ids.big_orange", 3)
        self.declare_parameter("class_ids.unknown", 4)

        self.declare_parameter("log.enable", True)

        P = lambda k: self.get_parameter(k).value

        self.gt_topic = str(P("ground_truth_topic"))
        self.output_topic = str(P("output_topic"))
        self.output_frame = str(P("output_frame"))
        self.log_enable = bool(P("log.enable"))

        self.class_id_yellow = int(P("class_ids.yellow"))
        self.class_id_blue = int(P("class_ids.blue"))
        self.class_id_orange = int(P("class_ids.orange"))
        self.class_id_big_orange = int(P("class_ids.big_orange"))
        self.class_id_unknown = int(P("class_ids.unknown"))

        self.sub = self.create_subscription(
            ConeArrayWithCovariance,
            self.gt_topic,
            self.cb_cones,
            qos_profile_sensor_data,
        )

        self.pub = self.create_publisher(
            PointCloud2,
            self.output_topic,
            qos_profile_sensor_data,
        )

        self.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="class_id", offset=12, datatype=PointField.UINT32, count=1),
        ]

        if self.log_enable:
            self.get_logger().info(
                f"[GT-CONE-ADAPTER] {self.gt_topic} -> {self.output_topic} | "
                f"IDs: yellow={self.class_id_yellow}, blue={self.class_id_blue}, "
                f"orange={self.class_id_orange}, big_orange={self.class_id_big_orange}, "
                f"unknown={self.class_id_unknown}"
            )

    @staticmethod
    def cone_xyz(cone):
        return (
            float(cone.point.x),
            float(cone.point.y),
            float(cone.point.z),
        )

    def cone_to_xyzcls(self, cone, class_id: int):
        x, y, z = self.cone_xyz(cone)
        return (x, y, z, int(class_id))

    def build_output_points(self, msg: ConeArrayWithCovariance):
        pts = []

        # Preserve categories explicitly. No guessing in code path beyond the param values.
        if hasattr(msg, "yellow_cones"):
            for c in msg.yellow_cones:
                pts.append(self.cone_to_xyzcls(c, self.class_id_yellow))

        if hasattr(msg, "blue_cones"):
            for c in msg.blue_cones:
                pts.append(self.cone_to_xyzcls(c, self.class_id_blue))

        if hasattr(msg, "orange_cones"):
            for c in msg.orange_cones:
                pts.append(self.cone_to_xyzcls(c, self.class_id_orange))

        if hasattr(msg, "big_orange_cones"):
            for c in msg.big_orange_cones:
                pts.append(self.cone_to_xyzcls(c, self.class_id_big_orange))

        if hasattr(msg, "unknown_color_cones"):
            for c in msg.unknown_color_cones:
                pts.append(self.cone_to_xyzcls(c, self.class_id_unknown))

        return pts

    def publish_cloud(self, header_in, frame_id: str, points):
        K = len(points)
        point_step = 16
        data = bytearray(point_step * K)

        for i, (x, y, z, cid) in enumerate(points):
            base = i * point_step
            data[base + 0: base + 4] = np.float32(x).tobytes()
            data[base + 4: base + 8] = np.float32(y).tobytes()
            data[base + 8: base + 12] = np.float32(z).tobytes()
            data[base + 12: base + 16] = np.uint32(cid).tobytes()

        msg = PointCloud2()
        msg.header = header_in
        if frame_id:
            msg.header.frame_id = frame_id

        msg.height = 1
        msg.width = K
        msg.fields = self.fields
        msg.is_bigendian = False
        msg.point_step = point_step
        msg.row_step = point_step * K
        msg.is_dense = True
        msg.data = bytes(data)

        self.pub.publish(msg)

    def cb_cones(self, msg: ConeArrayWithCovariance):
        points = self.build_output_points(msg)

        frame_id = self.output_frame.strip()
        if not frame_id:
            frame_id = msg.header.frame_id

        self.publish_cloud(msg.header, frame_id, points)


def main(args=None):
    rclpy.init(args=args)
    node = GroundTruthConeAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()