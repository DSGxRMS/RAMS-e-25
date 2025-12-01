#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2


class ConeFusionNode(Node):
    """
    Fuses LiDAR cones (/cones) with camera cones (/cones_colour).

    Inputs:
      - /cones:        PointCloud2 with fields x,y,z (LiDAR cones, trusted for position)
      - /cones_colour: PointCloud2 with fields x,y,z,class_id (camera cones, trusted for colour)

    Output:
      - /cones_fused: PointCloud2 with fields x,y,z,class_id
        * Position = LiDAR x,y
        * class_id chosen by:
            1) If matched with camera cone within match_radius:
                 -> take camera class_id.
            2) Else (unmatched LiDAR cone, but camera cones exist):
                 -> take colour of nearest camera cone (no radius limit).
            3) Else (no camera cones at all):
                 -> class_id = 4 (unknown/gray).

    Note:
      - Camera-only cones (without LiDAR association) are currently ignored.
        TODO: consider promoting persistent camera-only cones if LiDAR is missing/not seeing them.
    """

    def __init__(self):
        super().__init__("cone_fusion_node")

        # Parameters
        self.declare_parameter("lidar_topic", "/perception/cones")
        self.declare_parameter("camera_topic", "/perception/cones_colour")
        self.declare_parameter("output_topic", "/perception/cones_fused")
        self.declare_parameter("match_radius", 1.5)  # meters

        self.lidar_topic = self.get_parameter("lidar_topic").get_parameter_value().string_value
        self.camera_topic = self.get_parameter("camera_topic").get_parameter_value().string_value
        self.output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self.match_radius = float(self.get_parameter("match_radius").get_parameter_value().double_value)

        # Shared state
        self._lock = threading.Lock()
        self._lidar_pts = []     # list of (x, y)
        self._lidar_header = None
        self._cam_pts = []       # list of (x, y, class_id)
        self._have_lidar = False
        self._have_cam = False

        # Subscribers
        self.sub_lidar = self.create_subscription(
            PointCloud2,
            self.lidar_topic,
            self.cb_lidar,
            qos_profile_sensor_data,
        )
        self.sub_cam = self.create_subscription(
            PointCloud2,
            self.camera_topic,
            self.cb_camera,
            qos_profile_sensor_data,
        )

        # Publisher
        self.pub_fused = self.create_publisher(
            PointCloud2,
            self.output_topic,
            qos_profile_sensor_data,
        )

        # Timer for fusion (10 Hz)
        self.timer = self.create_timer(0.1, self.fuse_and_publish)

        self.get_logger().info(
            f"[ConeFusion] Subscribing to LiDAR: {self.lidar_topic}, "
            f"Camera: {self.camera_topic}, publishing: {self.output_topic}, "
            f"match_radius={self.match_radius:.2f} m"
        )

    # ----------- Callbacks -----------

    def cb_lidar(self, msg: PointCloud2):
        """Read LiDAR cones: expect fields x,y,z."""
        pts = []
        try:
            for x, y, _ in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                pts.append((float(x), float(y)))
        except Exception as e:
            self.get_logger().warn(f"[ConeFusion] Error parsing LiDAR PointCloud2: {e}")
            return

        with self._lock:
            self._lidar_pts = pts
            self._lidar_header = msg.header
            self._have_lidar = True

    def cb_camera(self, msg: PointCloud2):
        """Read camera cones: expect fields x,y,z,class_id."""
        pts = []
        try:
            for x, y, _, cls_id in pc2.read_points(
                msg, field_names=("x", "y", "z", "class_id"), skip_nans=True
            ):
                pts.append((float(x), float(y), int(cls_id)))
        except Exception as e:
            self.get_logger().warn(f"[ConeFusion] Error parsing camera PointCloud2: {e}")
            return

        with self._lock:
            self._cam_pts = pts
            self._have_cam = True

    # ----------- Fusion -----------

    def fuse_and_publish(self):
        with self._lock:
            lidar_pts = list(self._lidar_pts)
            cam_pts = list(self._cam_pts)
            header = self._lidar_header
            have_lidar = self._have_lidar
            have_cam = self._have_cam

        if not have_lidar or header is None:
            # No LiDAR cones => nothing to publish (we trust LiDAR for geometry)
            return

        nL = len(lidar_pts)
        nC = len(cam_pts)

        if nL == 0 and nC == 0:
            # Nothing at all; we could publish empty cloud if desired
            return

        fused_points = []

        # Pre-convert to arrays for speed if needed; here simple lists are fine
        used_cam = set()
        radius_sq = self.match_radius * self.match_radius

        # 1) First pass: try to match each LiDAR cone with a unique camera cone within match_radius
        for i, (lx, ly) in enumerate(lidar_pts):
            best_j = -1
            best_d2 = None

            for j, (cx, cy, ccls) in enumerate(cam_pts):
                if j in used_cam:
                    continue
                dx = lx - cx
                dy = ly - cy
                d2 = dx * dx + dy * dy
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best_j = j

            if best_j != -1 and best_d2 is not None and best_d2 <= radius_sq:
                # Match found within radius: use camera colour
                _, _, ccls = cam_pts[best_j]
                used_cam.add(best_j)
                fused_points.append((lx, ly, 0.0, ccls))
            else:
                # No camera cone close enough -> will handle in second pass (colour borrowing)
                fused_points.append((lx, ly, 0.0, None))  # colour to be filled

        # 2) Second pass: unmatched LiDAR cones get colour of nearest camera cone (if any)
        if nC > 0:
            for idx, (x, y, z, ccls) in enumerate(fused_points):
                if ccls is not None:
                    continue  # already matched directly
                # Find nearest camera cone (no radius limit)
                best_j = -1
                best_d2 = None
                for j, (cx, cy, cam_cls) in enumerate(cam_pts):
                    dx = x - cx
                    dy = y - cy
                    d2 = dx * dx + dy * dy
                    if best_d2 is None or d2 < best_d2:
                        best_d2 = d2
                        best_j = j
                if best_j != -1:
                    _, _, cam_cls = cam_pts[best_j]
                    fused_points[idx] = (x, y, z, cam_cls)
                else:
                    # Extremely unlikely if nC>0, but just in case
                    fused_points[idx] = (x, y, z, 4)  # unknown/gray
        else:
            # No camera cones at all: mark all as unknown colour
            fused_points = [(x, y, z, 4) for (x, y, z, _) in fused_points]

        # TODO: Handle camera-only cones (cam_pts indices not used in any association).
        # For now, we ignore them because LiDAR is trusted for geometry and
        # camera-only detections are more likely to be false positives.
        # Currently decent matches available - Can sustain cam points if YOLO v8 turns out better.

        # 3) Publish fused cloud
        fields = [
            PointField(name="x",        offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y",        offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z",        offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name="class_id", offset=12, datatype=PointField.UINT32,  count=1),
        ]

        cloud = pc2.create_cloud(header, fields, fused_points)
        self.pub_fused.publish(cloud)

        # self.get_logger().info(
        #     # f"[ConeFusion] LiDAR cones: {nL}, camera cones: {nC}, fused: {len(fused_points)}"
        # )


def main(args=None):
    rclpy.init(args=args)
    node = ConeFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
