#!/usr/bin/env python3
# slam_path_points_publisher.py
#
# Publishes a sequential path to /path_points (nav_msgs/Path):
#   [current car pose] -> [matched centroid 1] -> [matched centroid 2] -> ...
#
# Logic is adapted from slam_path_csv_visualiser.py, but WITHOUT matplotlib.
# It keeps the same sequential constraints used for plotting the red polyline.

import math
import threading
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
)

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from eufs_msgs.msg import ConeArrayWithCovariance

from pathlib import Path as SysPath

# Optional SciPy-based Delaunay
try:
    from scipy.spatial import Delaunay
    _HAS_SCIPY = True
except ImportError:
    Delaunay = None
    _HAS_SCIPY = False


# Default CSV under the package dir
CSVPATH = SysPath(__file__).parent / "pp_utils" / "skidpad_path.csv"


def yaw_from_quat(qx, qy, qz, qw) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def quat_from_yaw(yaw: float):
    # geometry_msgs Quaternion fields: x,y,z,w
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


class SlamPathPointsPublisher(Node):
    def __init__(self):
        super().__init__("slam_path_points_publisher")

        # ---- Parameters ----
        # Topics
        self.declare_parameter("topics.odom_in", "/slam/odom")
        self.declare_parameter("topics.map_in", "/slam/map_cones")
        self.declare_parameter("topics.path_out", "/path_points")

        # Publish rate (Hz)
        self.declare_parameter("publish_hz", 20.0)

        # Sector FOV parameters
        self.declare_parameter("fov.vertex_offset_m", -5.0)   # vertex 5 m behind car
        self.declare_parameter("fov.radius_m", 30.0)          # reaches ~25 m ahead
        self.declare_parameter("fov.angle_deg", 60.0)         # ±30° around heading

        # QoS
        self.declare_parameter("qos.best_effort", True)
        self.declare_parameter("qos.depth", 50)

        # CSV reference path + sequential tracking behaviour
        self.declare_parameter("path_file", CSVPATH.as_posix())
        self.declare_parameter("ref.loop", False)
        self.declare_parameter("ref.forward_band", 50)
        self.declare_parameter("ref.search_window", 200)
        self.declare_parameter("ref.lost_threshold_m", 5.0)
        self.declare_parameter("match.radius_m", 1.0)

        # Centroid filters
        self.declare_parameter("centroid.min_car_dist_m", 2.0)
        self.declare_parameter("centroid.min_spacing_m", 1.0)

        gp = self.get_parameter
        odom_topic = str(gp("topics.odom_in").value)
        map_topic = str(gp("topics.map_in").value)
        path_topic = str(gp("topics.path_out").value)

        best_effort = bool(gp("qos.best_effort").value)
        depth = int(gp("qos.depth").value)

        self.publish_hz = float(gp("publish_hz").value)
        self.publish_period = 1.0 / max(1e-6, self.publish_hz)

        self.vertex_offset = float(gp("fov.vertex_offset_m").value)
        self.fov_radius = float(gp("fov.radius_m").value)
        self.fov_angle_deg = float(gp("fov.angle_deg").value)
        self.fov_half_rad = math.radians(self.fov_angle_deg * 0.5)

        self.ref_loop = bool(gp("ref.loop").value)
        self.ref_forward_band = int(gp("ref.forward_band").value)
        self.ref_search_window = int(gp("ref.search_window").value)
        self.ref_lost_threshold = float(gp("ref.lost_threshold_m").value)
        self.match_radius = float(gp("match.radius_m").value)

        self.centroid_min_car_dist = float(gp("centroid.min_car_dist_m").value)
        self.centroid_min_spacing = float(gp("centroid.min_spacing_m").value)

        # Resolve CSV path via pathlib
        path_param = SysPath(str(gp("path_file").value))
        if not path_param.is_absolute():
            path_param = SysPath(__file__).parent / path_param

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

        # ---- Load CSV reference path ----
        try:
            df = pd.read_csv(path_param)
            self.global_rx = df["x"].to_numpy(dtype=float)
            self.global_ry = df["y"].to_numpy(dtype=float)
            self.get_logger().info(
                f"[path_pub] Loaded reference path {path_param.as_posix()} "
                f"with {len(self.global_rx)} points"
            )
        except Exception as e:
            self.get_logger().error(f"[path_pub] Failed to load path {path_param}: {e}")
            self.global_rx = np.array([], dtype=float)
            self.global_ry = np.array([], dtype=float)

        # Reference path tracking state (forward-only)
        self.cur_idx = 0
        self.path_initialized = False

        # ---- Subscriptions ----
        self.create_subscription(Odometry, odom_topic, self.cb_odom, qos)
        self.create_subscription(ConeArrayWithCovariance, map_topic, self.cb_map, qos)

        # ---- Publisher ----
        self.path_pub = self.create_publisher(Path, path_topic, qos)

        # ---- State ----
        self.car_x: Optional[float] = None
        self.car_y: Optional[float] = None
        self.car_yaw: Optional[float] = None  # radians
        self.last_frame_id: str = "map"

        # Global map cones in map frame
        self.blue_global: List[Tuple[float, float]] = []
        self.yellow_global: List[Tuple[float, float]] = []
        self.orange_global: List[Tuple[float, float]] = []
        self.big_global: List[Tuple[float, float]] = []

        self.data_lock = threading.Lock()

        if not _HAS_SCIPY:
            self.get_logger().warn(
                "[path_pub] SciPy not found; using k-NN graph instead of true Delaunay."
            )

        self.get_logger().info(
            f"[path_pub] odom={odom_topic}, map={map_topic}, out={path_topic}, "
            f"publish_hz={self.publish_hz}, "
            f"vertex_offset={self.vertex_offset}m, radius={self.fov_radius}m, "
            f"angle={self.fov_angle_deg}°"
        )

        # Timer loop for publishing
        self.create_timer(self.publish_period, self.timer_cb)

    # --------------------- Core helpers ---------------------

    def _has_reference(self) -> bool:
        return self.global_rx.size > 0

    def _dist_sq_to_ref(self, idx: int, cx: float, cy: float) -> float:
        dx = cx - self.global_rx[idx]
        dy = cy - self.global_ry[idx]
        return dx * dx + dy * dy

    def _update_reference_progress(self, cx: float, cy: float):
        """
        Forward-only index tracking along CSV with a forward band and
        "lost" re-localisation.
        """
        if not self._has_reference():
            return

        n = len(self.global_rx)
        if n == 0:
            return

        # --- First initialisation: global nearest search ---
        if not self.path_initialized:
            best_idx = 0
            min_d2 = float("inf")
            for i in range(n):
                d2 = self._dist_sq_to_ref(i, cx, cy)
                if d2 < min_d2:
                    min_d2 = d2
                    best_idx = i
            self.cur_idx = best_idx
            self.path_initialized = True
            return

        # --- Normal forward-band search ---
        forward_band = max(1, min(self.ref_forward_band, n - 1))
        best_idx = None
        best_d2 = float("inf")

        if self.ref_loop:
            for step in range(forward_band + 1):
                idx = (self.cur_idx + step) % n
                d2 = self._dist_sq_to_ref(idx, cx, cy)
                if d2 < best_d2:
                    best_d2 = d2
                    best_idx = idx
        else:
            start = self.cur_idx
            end = min(n - 1, self.cur_idx + forward_band)
            for idx in range(start, end + 1):
                d2 = self._dist_sq_to_ref(idx, cx, cy)
                if d2 < best_d2:
                    best_d2 = d2
                    best_idx = idx

        if best_idx is not None:
            if self.ref_loop:
                self.cur_idx = best_idx
            else:
                self.cur_idx = max(self.cur_idx, best_idx)

        # --- Lost detection + global re-localisation ---
        if self.ref_lost_threshold > 0.0 and math.isfinite(best_d2):
            lost_thresh2 = self.ref_lost_threshold * self.ref_lost_threshold
            if best_d2 > lost_thresh2:
                global_best_idx = self.cur_idx
                global_best_d2 = float("inf")
                for i in range(n):
                    d2 = self._dist_sq_to_ref(i, cx, cy)
                    if d2 < global_best_d2:
                        global_best_d2 = d2
                        global_best_idx = i
                if global_best_d2 < best_d2:
                    self.cur_idx = global_best_idx

    # --------------------- Callbacks ---------------------

    def cb_odom(self, msg: Odometry):
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

        frame_id = msg.header.frame_id.strip() or "map"

        with self.data_lock:
            self.car_x = x
            self.car_y = y
            self.car_yaw = yaw
            self.last_frame_id = frame_id

    def cb_map(self, msg: ConeArrayWithCovariance):
        blue, yellow, orange, big = [], [], [], []

        for c in msg.blue_cones:
            blue.append((float(c.point.x), float(c.point.y)))
        for c in msg.yellow_cones:
            yellow.append((float(c.point.x), float(c.point.y)))
        for c in msg.orange_cones:
            orange.append((float(c.point.x), float(c.point.y)))
        for c in msg.big_orange_cones:
            big.append((float(c.point.x), float(c.point.y)))

        with self.data_lock:
            self.blue_global = blue
            self.yellow_global = yellow
            self.orange_global = orange
            self.big_global = big

    # --------------------- FOV / graph helpers ---------------------

    def _world_to_local(self, car_x, car_y, car_yaw, px, py):
        """Transform world -> car frame (x forward, y left)."""
        c = math.cos(car_yaw)
        s = math.sin(car_yaw)
        dx = px - car_x
        dy = py - car_y
        lx = c * dx + s * dy
        ly = -s * dx + c * dy
        return lx, ly

    def _compute_fov_points(self, car_x, car_y, car_yaw):
        """
        Select cones within the circular sector FOV, and in front of the car (local x >= 0).
        """
        vertex_offset = self.vertex_offset
        R = self.fov_radius
        half_angle = self.fov_half_rad

        points_window: List[Tuple[float, float]] = []
        classes_window: List[str] = []

        def add_list(cones: List[Tuple[float, float]], cls_name: str):
            nonlocal points_window, classes_window
            for (px, py) in cones:
                lx, ly = self._world_to_local(car_x, car_y, car_yaw, px, py)

                # ignore behind-car points
                if lx < 0.0:
                    continue

                # relative to vertex in local frame
                vx = lx - vertex_offset
                vy = ly
                r = math.hypot(vx, vy)
                if r > R or r < 1e-3:
                    continue
                theta = math.atan2(vy, vx)
                if abs(theta) <= half_angle:
                    points_window.append((px, py))
                    classes_window.append(cls_name)

        add_list(self.blue_global, "blue")
        add_list(self.yellow_global, "yellow")
        add_list(self.orange_global, "orange")
        add_list(self.big_global, "big")

        return points_window, classes_window

    def _build_edges(self, points_window, classes_window):
        """
        Build graph edges using Delaunay (if available) or k-NN.

        Constraints:
          - edge length < 6 m
          - Drop blue-blue edges
          - Drop yellow-yellow edges
          - Drop orange-like ↔ (blue or yellow)
          - Keep orange-like ↔ orange-like and blue-yellow
        """
        N = len(points_window)
        if N < 2:
            return []

        pts = np.asarray(points_window, dtype=float)
        edges_set = set()

        if _HAS_SCIPY and N >= 3:
            tri = Delaunay(pts)
            for simplex in tri.simplices:
                i, j, k = int(simplex[0]), int(simplex[1]), int(simplex[2])
                for a, b in ((i, j), (j, k), (k, i)):
                    if a > b:
                        a, b = b, a
                    edges_set.add((a, b))
        else:
            k = min(3, N - 1)
            for i in range(N):
                d2 = np.sum((pts - pts[i]) ** 2, axis=1)
                d2[i] = float("inf")
                nn_idx = np.argsort(d2)[:k]
                for j in nn_idx:
                    a, b = i, int(j)
                    if a > b:
                        a, b = b, a
                    edges_set.add((a, b))

        orange_like = {"orange", "big"}
        blue_yellow = {"blue", "yellow"}

        edges = []
        max_len2 = 6.0 * 6.0

        for (i, j) in edges_set:
            ci = classes_window[i]
            cj = classes_window[j]

            x1, y1 = points_window[i]
            x2, y2 = points_window[j]
            d2 = (x1 - x2) ** 2 + (y1 - y2) ** 2
            if d2 >= max_len2:
                continue

            if (ci == "blue" and cj == "blue") or (ci == "yellow" and cj == "yellow"):
                continue

            if ((ci in orange_like and cj in blue_yellow) or
               (cj in orange_like and ci in blue_yellow)):
                continue

            edges.append((i, j))

        return edges

    def _filter_centroids(
        self,
        centroids_raw: List[Tuple[float, float, bool]],
        car_x: float,
        car_y: float,
    ) -> List[Tuple[float, float, bool]]:
        """
        Enforce:
          - no centroid within centroid_min_car_dist of car
          - no two centroids closer than centroid_min_spacing
        """
        if not centroids_raw:
            return []

        pts = np.array([[c[0], c[1]] for c in centroids_raw], dtype=float)
        car = np.array([car_x, car_y], dtype=float)
        d2_car = np.sum((pts - car) ** 2, axis=1)

        order = np.argsort(-d2_car)  # farthest first

        kept_pts = []
        kept = []

        for idx in order:
            p = pts[idx]

            if np.linalg.norm(p - car) < self.centroid_min_car_dist:
                continue

            if not kept_pts:
                kept_pts.append(p)
                kept.append((float(p[0]), float(p[1]), bool(centroids_raw[idx][2])))
                continue

            too_close = False
            for kp in kept_pts:
                if np.linalg.norm(p - kp) < self.centroid_min_spacing:
                    too_close = True
                    break
            if too_close:
                continue

            kept_pts.append(p)
            kept.append((float(p[0]), float(p[1]), bool(centroids_raw[idx][2])))

        return kept

    def _match_centroids_to_csv_window(
        self,
        centroids: List[Tuple[float, float, bool]],
    ) -> List[Tuple[float, float]]:
        """
        For CSV indices in [cur_idx, cur_idx+ref_search_window], in order,
        match nearest centroid within match_radius (unused after matched).
        """
        if not self._has_reference() or not centroids:
            return []

        n_ref = len(self.global_rx)
        if n_ref == 0:
            return []

        pts = np.array([[c[0], c[1]] for c in centroids], dtype=float)
        used = np.zeros(len(centroids), dtype=bool)

        matched: List[Tuple[float, float]] = []
        max_r2 = self.match_radius * self.match_radius

        max_steps = min(self.ref_search_window, n_ref)
        for step in range(max_steps + 1):
            idx = self.cur_idx + step
            if self.ref_loop:
                idx %= n_ref
            elif idx >= n_ref:
                break

            px = self.global_rx[idx]
            py = self.global_ry[idx]

            diff = pts - np.array([px, py])
            d2 = np.sum(diff * diff, axis=1)
            d2[used] = float("inf")

            j = int(np.argmin(d2))
            if not math.isfinite(d2[j]) or d2[j] > max_r2:
                continue

            used[j] = True
            matched.append((float(pts[j, 0]), float(pts[j, 1])))

        return matched

    # --------------------- Publishing loop ---------------------

    def timer_cb(self):
        with self.data_lock:
            car_x = self.car_x
            car_y = self.car_y
            car_yaw = self.car_yaw
            frame_id = self.last_frame_id

            blue_global = list(self.blue_global)
            yellow_global = list(self.yellow_global)
            orange_global = list(self.orange_global)
            big_global = list(self.big_global)

        if car_x is None or car_y is None or car_yaw is None:
            return

        # Update cone lists locally (avoid lock usage deeper)
        self.blue_global = blue_global
        self.yellow_global = yellow_global
        self.orange_global = orange_global
        self.big_global = big_global

        # Preserve sequential progression along CSV
        self._update_reference_progress(car_x, car_y)

        # Find centroids from FOV graph
        points_window, classes_window = self._compute_fov_points(car_x, car_y, car_yaw)
        edges = self._build_edges(points_window, classes_window)

        orange_like = {"orange", "big"}
        centroids_raw: List[Tuple[float, float, bool]] = []

        for (i, j) in edges:
            (x1e, y1e) = points_window[i]
            (x2e, y2e) = points_window[j]
            ci = classes_window[i]
            cj = classes_window[j]
            is_big_big = (ci == "big" and cj == "big")

            mx = 0.5 * (x1e + x2e)
            my = 0.5 * (y1e + y2e)
            centroids_raw.append((mx, my, is_big_big))

        centroids = self._filter_centroids(centroids_raw, car_x, car_y)

        # Sequential match in CSV order (this was your red polyline)
        matched = self._match_centroids_to_csv_window(centroids)

        # Build and publish nav_msgs/Path: [car pose] + matched points
        now = self.get_clock().now().to_msg()
        path_msg = Path()
        path_msg.header.stamp = now
        path_msg.header.frame_id = frame_id if frame_id else "map"

        # First pose: current car pose
        car_pose = PoseStamped()
        car_pose.header.stamp = now
        car_pose.header.frame_id = path_msg.header.frame_id
        car_pose.pose.position.x = float(car_x)
        car_pose.pose.position.y = float(car_y)
        car_pose.pose.position.z = 0.0
        qx, qy, qz, qw = quat_from_yaw(float(car_yaw))
        car_pose.pose.orientation.x = qx
        car_pose.pose.orientation.y = qy
        car_pose.pose.orientation.z = qz
        car_pose.pose.orientation.w = qw
        path_msg.poses.append(car_pose)

        # Subsequent poses: matched points in order
        for (mx, my) in matched:
            ps = PoseStamped()
            ps.header.stamp = now
            ps.header.frame_id = path_msg.header.frame_id
            ps.pose.position.x = float(mx)
            ps.pose.position.y = float(my)
            ps.pose.position.z = 0.0
            # Waypoint orientation not essential; set identity.
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)

        self.path_pub.publish(path_msg)


def main():
    rclpy.init()
    node = SlamPathPointsPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()