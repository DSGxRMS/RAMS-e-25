#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2

from message_filters import Subscriber, ApproximateTimeSynchronizer

# Optional: best 1:1 assignment (Hungarian)
try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


class LidarStereoColourFuseNoTF(Node):
    """
    Fuses:
      - LiDAR centroids: /perception/cones (frame velodyne), fields x,y,z
      - Stereo+YOLO:     /perception/cones_stereo (camera optical), fields x,y,z,class_id

    No TF used.
    We match in a common pseudo vehicle plane:
      LiDAR:   forward=f=x, left=l=y
      Camera optical: forward=f=z, left=l=-x   (since x=right, z=forward)

    Output:
      - /perception/cones_fused in LiDAR frame_id (same as LiDAR msg)
      - points are LiDAR xyz + camera class_id
      - 1:1 matches only; unmatched LiDAR ignored
    """

    def __init__(self):
        super().__init__("lidar_stereo_colour_fuse_notf")

        # Topics
        self.declare_parameter("lidar_topic", "/perception/cones")
        self.declare_parameter("camera_topic", "/perception/cones_stereo")
        self.declare_parameter("output_topic", "/perception/cones_fused")

        # Sync
        self.declare_parameter("sync_slop", 0.08)
        self.declare_parameter("sync_queue", 10)

        # Matching gate in meters (in forward/left plane)
        self.declare_parameter("max_match_dist", 2.0)

        # Use Hungarian (best) if SciPy exists; else greedy
        self.declare_parameter("use_hungarian", True)

        # Camera frame interpretation:
        # "optical"  => camera points are (x=right, y=down, z=forward) [your current cones_stereo]
        # "ros"      => camera points are (x=forward, y=left, z=up)   [if you ever publish that]
        self.declare_parameter("camera_frame_mode", "optical")  # optical | ros

        self.lidar_topic = self.get_parameter("lidar_topic").value
        self.camera_topic = self.get_parameter("camera_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.maxd = float(self.get_parameter("max_match_dist").value)
        self.use_h = bool(self.get_parameter("use_hungarian").value) and SCIPY_OK
        self.cam_mode = str(self.get_parameter("camera_frame_mode").value).lower()

        # Subscribers (sync)
        self.sub_lidar = Subscriber(self, PointCloud2, self.lidar_topic, qos_profile=qos_profile_sensor_data)
        self.sub_cam = Subscriber(self, PointCloud2, self.camera_topic, qos_profile=qos_profile_sensor_data)
        self.sync = ApproximateTimeSynchronizer(
            [self.sub_lidar, self.sub_cam],
            queue_size=int(self.get_parameter("sync_queue").value),
            slop=float(self.get_parameter("sync_slop").value),
        )
        self.sync.registerCallback(self.cb_sync)

        # Publisher
        self.pub = self.create_publisher(PointCloud2, self.output_topic, qos_profile_sensor_data)

        self.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="class_id", offset=12, datatype=PointField.UINT32, count=1),
        ]

        self.get_logger().info(
            f"[FuseNoTF] lidar={self.lidar_topic} cam={self.camera_topic} -> {self.output_topic} | "
            f"cam_mode={self.cam_mode} | hungarian={self.use_h} (scipy={SCIPY_OK}) | gate={self.maxd}m"
        )

    @staticmethod
    def read_xyz(msg: PointCloud2) -> np.ndarray:
        pts = []
        for x, y, z in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            pts.append((float(x), float(y), float(z)))
        return np.asarray(pts, dtype=np.float64)

    @staticmethod
    def read_xyz_cls(msg: PointCloud2):
        pts = []
        cls = []
        for x, y, z, cid in pc2.read_points(msg, field_names=("x", "y", "z", "class_id"), skip_nans=True):
            pts.append((float(x), float(y), float(z)))
            cls.append(int(cid))
        return np.asarray(pts, dtype=np.float64), np.asarray(cls, dtype=np.int32)

    def to_forward_left_lidar(self, P_l: np.ndarray) -> np.ndarray:
        # LiDAR assumed: x forward, y left
        if P_l.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float64)
        return P_l[:, [0, 1]]  # (forward, left)

    def to_forward_left_cam(self, P_c: np.ndarray) -> np.ndarray:
        if P_c.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float64)

        if self.cam_mode == "ros":
            # camera points already x forward, y left
            f = P_c[:, 0]
            l = P_c[:, 1]
        else:
            # optical: x right, y down, z forward  => forward=z, left=-x
            f = P_c[:, 2]
            l = -P_c[:, 0]
        return np.stack([f, l], axis=1)

    def associate_1to1(self, A: np.ndarray, B: np.ndarray):
        """
        A: Nx2 (lidar forward/left)
        B: Mx2 (cam forward/left)
        returns list of (i, j) matches (1:1) within gate
        """
        n, m = A.shape[0], B.shape[0]
        if n == 0 or m == 0:
            return []

        D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)  # NxM
        BIG = 1e6
        C = np.where(D <= self.maxd, D, BIG)

        matches = []
        if self.use_h:
            rows, cols = linear_sum_assignment(C)
            for i, j in zip(rows, cols):
                if C[i, j] < BIG:
                    matches.append((int(i), int(j)))
        else:
            # greedy
            Cg = C.copy()
            used_i = set()
            used_j = set()
            while True:
                i, j = np.unravel_index(np.argmin(Cg), Cg.shape)
                if Cg[i, j] >= BIG:
                    break
                if i in used_i or j in used_j:
                    Cg[i, j] = BIG
                    continue
                matches.append((int(i), int(j)))
                used_i.add(i)
                used_j.add(j)
                Cg[i, :] = BIG
                Cg[:, j] = BIG

        return matches

    def publish_cloud(self, header_in, frame_id: str, pts_xyz: np.ndarray, pts_cls: np.ndarray):
        K = pts_xyz.shape[0]
        point_step = 16
        data = bytearray(point_step * K)

        for i in range(K):
            base = i * point_step
            x, y, z = pts_xyz[i]
            cid = int(pts_cls[i])
            data[base + 0: base + 4] = np.float32(x).tobytes()
            data[base + 4: base + 8] = np.float32(y).tobytes()
            data[base + 8: base + 12] = np.float32(z).tobytes()
            data[base + 12: base + 16] = np.uint32(cid).tobytes()

        msg = PointCloud2()
        msg.header = header_in
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

    def cb_sync(self, lidar_msg: PointCloud2, cam_msg: PointCloud2):
        P_l = self.read_xyz(lidar_msg)
        P_c, C_c = self.read_xyz_cls(cam_msg)

        # empty publish
        if P_l.shape[0] == 0 or P_c.shape[0] == 0:
            self.publish_cloud(
                lidar_msg.header,
                lidar_msg.header.frame_id,
                np.empty((0, 3), dtype=np.float64),
                np.empty((0,), dtype=np.int32),
            )
            return

        A = self.to_forward_left_lidar(P_l)
        B = self.to_forward_left_cam(P_c)

        matches = self.associate_1to1(A, B)

        if len(matches) == 0:
            self.publish_cloud(
                lidar_msg.header,
                lidar_msg.header.frame_id,
                np.empty((0, 3), dtype=np.float64),
                np.empty((0,), dtype=np.int32),
            )
            return

        out_xyz = np.array([P_l[i] for i, _ in matches], dtype=np.float64)  # LiDAR xyz
        out_cls = np.array([C_c[j] for _, j in matches], dtype=np.int32)    # camera class_id

        self.publish_cloud(lidar_msg.header, lidar_msg.header.frame_id, out_xyz, out_cls)


def main(args=None):
    rclpy.init(args=args)
    node = LidarStereoColourFuseNoTF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
