#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
)

from nav_msgs.msg import Odometry


class GroundTruthOdomRelay(Node):
    def __init__(self):
        # Keep same node name for compatibility with your launch/setup
        super().__init__(
            "fastslam_localizer",
            automatically_declare_parameters_from_overrides=True,
        )

        # Input = simulator ground truth odometry
        # Output = perfect odometry for SLAM
        self.declare_parameter("topics.gt_odom", "/ground_truth/odom")
        self.declare_parameter("topics.out_odom", "/slam/odom_raw")
        self.declare_parameter("frames.frame_id", "map")
        self.declare_parameter("frames.child_frame_id", "base_link")
        self.declare_parameter("log.enable", True)

        P = lambda k: self.get_parameter(k).value

        self.topic_gt_odom = str(P("topics.gt_odom"))
        self.topic_out_odom = str(P("topics.out_odom"))
        self.frame_id = str(P("frames.frame_id"))
        self.child_frame_id = str(P("frames.child_frame_id"))
        self.log_enable = bool(P("log.enable"))

        qos_fast = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=50,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.sub_gt = self.create_subscription(
            Odometry,
            self.topic_gt_odom,
            self.cb_gt_odom,
            qos_fast,
        )

        self.pub_out = self.create_publisher(Odometry, self.topic_out_odom, 10)

        if self.log_enable:
            self.get_logger().info(
                f"[GT-ODOM-RELAY] Subscribing: {self.topic_gt_odom} -> Publishing: {self.topic_out_odom}"
            )

    def cb_gt_odom(self, msg: Odometry):
        out = Odometry()

        # Keep original timestamp
        out.header.stamp = msg.header.stamp

        # Force frames to what downstream SLAM expects
        out.header.frame_id = self.frame_id
        out.child_frame_id = self.child_frame_id

        # Copy pose and twist directly from simulator ground truth
        out.pose = msg.pose
        out.twist = msg.twist

        self.pub_out.publish(out)


def main():
    rclpy.init()
    node = GroundTruthOdomRelay()
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