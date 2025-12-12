#!/usr/bin/env python3
# gt_cones_to_fused_pc.py
#
# Minimal adapter:
#   /ground_truth/cones (EUFS ConeArray[WithCovariance])
#     -> /perception/cones_fused (PointCloud2: x,y,z,class_id)
#
# Only coloured cones are kept by default:
#   0: blue, 1: yellow, 2: orange, 3: big_orange
#   (unknown_color_cones are dropped unless include_unknown=true)
#
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2


def _import_cones_msg():
    """Prefer ConeArrayWithCovariance; fall back to ConeArray."""
    try:
        from eufs_msgs.msg import ConeArrayWithCovariance as ConesMsg
        return ConesMsg
    except Exception:
        from eufs_msgs.msg import ConeArray as ConesMsg
        return ConesMsg


def _xyz_from_cone(cone):
    """
    Extract (x,y,z) from different EUFS cone message variants.
    """
    for attr in ("point", "position", "location"):
        if hasattr(cone, attr):
            p = getattr(cone, attr)
            return float(getattr(p, "x", 0.0)), float(getattr(p, "y", 0.0)), float(getattr(p, "z", 0.0))
    if hasattr(cone, "x") and hasattr(cone, "y"):
        return float(cone.x), float(cone.y), float(getattr(cone, "z", 0.0))
    return 0.0, 0.0, 0.0


class GTConesToFusedPC(Node):
    """
    Subscribe: /ground_truth/cones (EUFS)
    Publish : /perception/cones_fused (PointCloud2 with x,y,z,class_id)

    Colour → class_id:
      0: blue_cones
      1: yellow_cones
      2: orange_cones
      3: big_orange_cones
    unknown_color_cones are dropped by default.
    """

    def __init__(self):
        super().__init__("gt_cones_to_fused_pc")

        # Parameters
        self.declare_parameter("input_topic", "/ground_truth/cones")
        self.declare_parameter("output_topic", "/perception/cones_fused")
        self.declare_parameter("include_unknown", False)  # drop unknowns by default

        input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self.include_unknown = bool(self.get_parameter("include_unknown").value)

        ConesMsg = _import_cones_msg()

        # Sub / Pub
        self.sub = self.create_subscription(
            ConesMsg,
            input_topic,
            self.cb_cones,
            qos_profile_sensor_data,
        )

        self.pub = self.create_publisher(
            PointCloud2,
            output_topic,
            qos_profile_sensor_data,
        )

        # Predefine PointCloud2 fields
        self.fields = [
            PointField(name="x",        offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y",        offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z",        offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name="class_id", offset=12, datatype=PointField.UINT32,  count=1),
        ]

        self.get_logger().info(
            f"[GTConesToFusedPC] input={input_topic} -> output={output_topic}, "
            f"include_unknown={self.include_unknown}"
        )

    def cb_cones(self, msg):
        fused_points = []

        # Color→id mapping
        # 0: blue, 1: yellow, 2: orange, 3: big_orange
        def add_cones(cones, cls_id):
            for c in cones:
                x, y, z = _xyz_from_cone(c)
                fused_points.append((float(x), float(y), float(z), int(cls_id)))

        # EUFS standard fields (if some are missing, getattr returns empty list)
        add_cones(getattr(msg, "blue_cones", []),       0)
        add_cones(getattr(msg, "yellow_cones", []),     1)
        add_cones(getattr(msg, "orange_cones", []),     2)
        add_cones(getattr(msg, "big_orange_cones", []), 3)

        # Unknown colour cones are OPTIONAL, dropped by default
        if self.include_unknown:
            unknown = getattr(msg, "unknown_color_cones", [])
            for c in unknown:
                x, y, z = _xyz_from_cone(c)
                fused_points.append((float(x), float(y), float(z), 4))

        if not fused_points:
            return

        cloud = pc2.create_cloud(msg.header, self.fields, fused_points)
        self.pub.publish(cloud)
        # kept silent on purpose


def main(args=None):
    rclpy.init(args=args)
    node = GTConesToFusedPC()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
