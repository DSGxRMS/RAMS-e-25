#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LiDAR crop + ground removal + DBSCAN clustering (2D output, best-effort QoS).

Subscribes:
  - /velodyne_points (sensor_msgs/PointCloud2)

Pipeline per frame:
  - PC2 -> Nx3 float32
  - Ground removal selected with ``ground_removal_method``:
      - Patchwork++ (default; full cloud before crop)
      - Vectorized RANSAC (crop before plane fitting)
  - Box crop:
      X: [0.2, 30] m  (forward)
      Y: [-8, +8] m   (lateral)
      Z: [-2, 2] m    (height band)
  - 2D DBSCAN clustering on XY of non-ground points
  - Publish one point per cluster centroid as 2D:
      - x, y from cluster centroid
      - z forced to 0

Publishes:
  - /cones (sensor_msgs/PointCloud2, xyz32, best-effort)
"""

import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from sklearn.cluster import DBSCAN

from perception.ground_removal import (
    GroundRemovalMethod,
    PatchworkGroundRemover,
    default_cluster_eps,
    vectorized_ransac,
)


class LidarCropGround2DNode(Node):
    def __init__(self):
        super().__init__("lidar_crop_ground_2d")

        # -------- Parameters --------
        self.declare_parameter("input_topic", "/velodyne_points")
        self.declare_parameter("output_topic", "/perception/cones")
        self.declare_parameter("ground_removal_method", "patchwork")

        # Box limits (meters)
        # X forward, Y left/right, Z up from ground
        self.declare_parameter("x_min", 0.2)
        self.declare_parameter("x_max", 15.0)
        self.declare_parameter("y_abs_max", 8.0)
        self.declare_parameter("z_min", -2.0)
        self.declare_parameter("z_max", 2.0)

        # RANSAC ground plane parameters
        self.declare_parameter("ransac_iters", 50)
        self.declare_parameter("ground_thresh", 0.05)  # meters
        self.declare_parameter("min_ground_inliers", 30)

        # Patchwork++ parameters validated on the EUFS single-lap benchmark.
        self.declare_parameter("patchwork_sensor_height", 1.08)
        self.declare_parameter("patchwork_min_range", 0.2)
        self.declare_parameter("patchwork_max_range", 80.0)
        self.declare_parameter("patchwork_distance_threshold", 0.04)
        self.declare_parameter("patchwork_enable_rnr", False)

        # DBSCAN clustering parameters (2D on XY)
        # <= 0 selects 0.50 m for Patchwork++ and 1.00 m for RANSAC.
        self.declare_parameter("cluster_eps", 0.0)          # meters
        self.declare_parameter("cluster_min_points", 1)
        self.declare_parameter("cluster_max_points", 50)    # reject huge blobs

        # -------- Read parameters --------
        self.input_topic = (
            self.get_parameter("input_topic")
            .get_parameter_value().string_value
        )
        self.output_topic = (
            self.get_parameter("output_topic")
            .get_parameter_value().string_value
        )
        method_value = self.get_parameter(
            "ground_removal_method"
        ).get_parameter_value().string_value
        self.ground_removal_method = GroundRemovalMethod.parse(method_value)

        self.x_min = (
            self.get_parameter("x_min").get_parameter_value().double_value
        )
        self.x_max = (
            self.get_parameter("x_max").get_parameter_value().double_value
        )
        self.y_abs_max = (
            self.get_parameter("y_abs_max").get_parameter_value().double_value
        )
        self.z_min = (
            self.get_parameter("z_min").get_parameter_value().double_value
        )
        self.z_max = (
            self.get_parameter("z_max").get_parameter_value().double_value
        )

        self.ransac_iters = (
            self.get_parameter("ransac_iters")
            .get_parameter_value().integer_value
        )
        self.ground_thresh = (
            self.get_parameter("ground_thresh")
            .get_parameter_value().double_value
        )
        self.min_ground_inliers = self.get_parameter(
            "min_ground_inliers"
        ).get_parameter_value().integer_value

        self.patchwork_sensor_height = self.get_parameter(
            "patchwork_sensor_height"
        ).get_parameter_value().double_value
        self.patchwork_min_range = self.get_parameter(
            "patchwork_min_range"
        ).get_parameter_value().double_value
        self.patchwork_max_range = self.get_parameter(
            "patchwork_max_range"
        ).get_parameter_value().double_value
        self.patchwork_distance_threshold = self.get_parameter(
            "patchwork_distance_threshold"
        ).get_parameter_value().double_value
        self.patchwork_enable_rnr = self.get_parameter(
            "patchwork_enable_rnr"
        ).get_parameter_value().bool_value

        self.cluster_eps = (
            self.get_parameter("cluster_eps")
            .get_parameter_value().double_value
        )
        if self.cluster_eps <= 0.0:
            self.cluster_eps = default_cluster_eps(self.ground_removal_method)
        self.cluster_min_points = self.get_parameter(
            "cluster_min_points"
        ).get_parameter_value().integer_value
        self.cluster_max_points = self.get_parameter(
            "cluster_max_points"
        ).get_parameter_value().integer_value

        self.patchwork = None
        if self.ground_removal_method is GroundRemovalMethod.PATCHWORK:
            self.patchwork = PatchworkGroundRemover(
                sensor_height=self.patchwork_sensor_height,
                minimum_range=self.patchwork_min_range,
                maximum_range=self.patchwork_max_range,
                distance_threshold=self.patchwork_distance_threshold,
                enable_rnr=self.patchwork_enable_rnr,
            )

        # -------- ROS I/O (best-effort QoS) --------
        self.sub = self.create_subscription(
            PointCloud2,
            self.input_topic,
            self.pc_callback,
            qos_profile_sensor_data,
        )

        self.pub = self.create_publisher(
            PointCloud2,
            self.output_topic,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            f"[LidarCropGround2D] Subscribing to: {self.input_topic} "
            "(best-effort)"
        )
        self.get_logger().info(
            f"[LidarCropGround2D] Publishing to: {self.output_topic} "
            "(best-effort)"
        )
        self.get_logger().info(
            f"[LidarCropGround2D] Crop box: X∈[{self.x_min},{self.x_max}] m, "
            f"|Y|≤{self.y_abs_max} m, "
            f"Z∈[{self.z_min},{self.z_max}] m"
        )
        self.get_logger().info(
            f"[LidarCropGround2D] Ground removal: "
            f"{self.ground_removal_method.value}"
        )
        if self.ground_removal_method is GroundRemovalMethod.RANSAC:
            self.get_logger().info(
                f"[LidarCropGround2D] RANSAC: iters={self.ransac_iters}, "
                f"ground_thresh={self.ground_thresh} m, "
                f"min_ground_inliers={self.min_ground_inliers}"
            )
        else:
            self.get_logger().info(
                f"[LidarCropGround2D] Patchwork++: "
                f"sensor_height={self.patchwork_sensor_height} m, "
                f"distance_threshold={self.patchwork_distance_threshold} m, "
                f"RNR={self.patchwork_enable_rnr}"
            )
        self.get_logger().info(
            f"[LidarCropGround2D] DBSCAN: eps={self.cluster_eps} m, "
            f"min_points={self.cluster_min_points}, "
            f"max_points={self.cluster_max_points}"
        )

        self._frames = 0
        self._last_log = time.time()

    # ---------- ROS callback ----------
    def pc_callback(self, msg: PointCloud2):
        t0 = time.time()

        try:
            # 1) Convert to Nx3
            P = self._pc2_to_xyz_array(msg)
            n_in = P.shape[0]

            if n_in == 0:
                # publish empty cones
                out_msg = pc2.create_cloud_xyz32(
                    msg.header, np.empty((0, 3), dtype=np.float32)
                )
                self.pub.publish(out_msg)
                self._log_stats(n_in, 0, 0, 0, time.time() - t0)
                return

            # 2) Crop statistics and selectable ground removal. Patchwork++
            # needs full-scan spatial context; RANSAC keeps the old crop-first
            # behaviour.
            P_crop = self._box_crop(P)
            n_crop = P_crop.shape[0]

            if self.ground_removal_method is GroundRemovalMethod.PATCHWORK:
                P_ng = self.patchwork.remove(P)
                P_ng = self._box_crop(P_ng)
            else:
                if n_crop == 0:
                    P_ng = P_crop
                else:
                    P_ng = self._remove_ground_ransac(P_crop)
                    # Preserve the previous RANSAC fallback.
                    if P_ng is None or P_ng.shape[0] == 0:
                        P_ng = P_crop

            n_ng = P_ng.shape[0]

            # 4) 2D DBSCAN clustering on XY
            centroids_xy, n_clusters = self._cluster_2d(P_ng)

            # 5) Build 2D output cloud (one point per centroid)
            if centroids_xy.shape[0] == 0:
                pts2d = np.empty((0, 3), dtype=np.float32)
            else:
                xy = centroids_xy
                z_zero = np.zeros((xy.shape[0], 1), dtype=np.float32)
                pts2d = np.hstack([xy.astype(np.float32), z_zero])  # [x, y, 0]

            out_msg = pc2.create_cloud_xyz32(msg.header, pts2d)
            self.pub.publish(out_msg)

            # 6) Logging
            self._log_stats(n_in, n_crop, n_ng, n_clusters, time.time() - t0)

        except Exception as e:
            self.get_logger().error(f"[LidarCropGround2D] Error: {e}")

    # ---------- Helpers: PC2 <-> NumPy ----------
    @staticmethod
    def _pc2_to_xyz_array(msg: PointCloud2) -> np.ndarray:
        """Convert PointCloud2 to Nx3 float32 array (x, y, z)."""
        gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        pts = np.fromiter((c for p in gen for c in p), dtype=np.float32)
        if pts.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        return pts.reshape(-1, 3)

    # ---------- Box crop ----------
    def _box_crop(self, P: np.ndarray) -> np.ndarray:
        """
        Keep points in:
            X ∈ [x_min, x_max]
            |Y| ≤ y_abs_max
            Z ∈ [z_min, z_max]
        """
        x = P[:, 0]
        y = P[:, 1]
        z = P[:, 2]

        mask = (
            (x >= self.x_min) & (x <= self.x_max) &
            (np.abs(y) <= self.y_abs_max) &
            (z >= self.z_min) & (z <= self.z_max)
        )
        return P[mask]

    # ---------- Vectorized RANSAC ground removal ----------
    def _remove_ground_ransac(self, P: np.ndarray) -> np.ndarray:
        """
        RANSAC plane fit, remove inliers (ground).

        Returns:
          nonground_points: Mx3, or None on failure
        """
        retained = vectorized_ransac(
            P,
            iterations=self.ransac_iters,
            threshold=self.ground_thresh,
            minimum_inliers=self.min_ground_inliers,
        )
        if retained is None:
            self.get_logger().warn(
                "[LidarCropGround2D] RANSAC failed or found too few ground "
                "inliers. "
                f"Skipping ground removal for this frame."
            )
        return retained

    # ---------- 2D DBSCAN clustering ----------
    def _cluster_2d(self, P: np.ndarray):
        """
        Run DBSCAN on XY only.
        Returns:
          - centroids_xy: Mx2
          - n_clusters: number of valid clusters
        """
        if P.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float32), 0

        xy = P[:, :2]  # Nx2

        db = DBSCAN(
            eps=float(self.cluster_eps),
            min_samples=int(self.cluster_min_points),
            n_jobs=-1,
        )

        labels = db.fit_predict(xy)  # -1 is noise
        labels = labels.astype(np.int32)

        unique_labels = np.unique(labels)
        centroids = []
        n_clusters = 0

        for lbl in unique_labels:
            if lbl < 0:
                continue  # skip noise

            idxs = np.where(labels == lbl)[0]
            count = idxs.size
            if (
                count < self.cluster_min_points
                or count > self.cluster_max_points
            ):
                continue

            cluster_xy = xy[idxs]
            c = cluster_xy.mean(axis=0)
            centroids.append(c)
            n_clusters += 1

        if not centroids:
            return np.empty((0, 2), dtype=np.float32), 0

        return np.vstack(centroids).astype(np.float32), n_clusters

    # ---------- Logging ----------
    def _log_stats(
        self, n_in: int, n_crop: int, n_ng: int,
        n_clusters: int, dt: float
    ):
        self._frames += 1
        now = time.time()
        if now - self._last_log > 1.5:
            self._last_log = now
            # self.get_logger().info(
            #     f"[LidarCropGround2D] Frame {self._frames}: "
            #     f"in={n_in} pts, crop={n_crop} pts, nonground={n_ng} pts, "
            #     f"clusters={n_clusters}, "
            #     f"{dt*1000.0:.1f} ms."
            # )


def main(args=None):
    rclpy.init(args=args)
    node = LidarCropGround2DNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
