#!/usr/bin/env python3
# slam_path_sector_visualiser.py
#
# Sector-FOV path generator for SLAM cone map (NO PLOTTER).
#   - Subscribes to /slam/odom (nav_msgs/Odometry)
#   - Subscribes to /slam/map_cones (eufs_msgs/ConeArrayWithCovariance)
#   - Inside the FOV:
#       * Build Delaunay (or k-NN) edges on cones
#       * Apply constraints:
#           - edge length < 6 m
#           - drop blue-blue and yellow-yellow edges
#           - drop any orange-like ↔ (blue or yellow) edges
#           - keep orange-like ↔ orange-like and blue-yellow
#       * For each allowed edge (EXCEPT big-big), compute midpoint as candidate
#       * BIG ORANGE SPECIAL:
#           - If exactly 2 big-orange cones in FOV: output ONLY their midpoint (as a special candidate)
#           - If >= 3 big-orange cones in FOV: output centroid of ALL big-orange cones (as a special candidate)
#       * Enforce ≥ 1 m spacing between candidates, keeping farther-first
#       * Drop any candidate within min_car_dist (default 2.0 m) of the car pose
#       * Build a greedy NN path from car through kept candidates,
#         limited by max hop distance (default 7 m)
#   - Publishes greedy path to /path_points (nav_msgs/Path) BEST_EFFORT QoS
#
import math
import threading
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

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from eufs_msgs.msg import ConeArrayWithCovariance

# Optional SciPy-based Delaunay
try:
    from scipy.spatial import Delaunay
    _HAS_SCIPY = True
except ImportError:
    Delaunay = None
    _HAS_SCIPY = False


def yaw_from_quat(qx, qy, qz, qw) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class SlamPathSectorVisualizer(Node):
    def __init__(self):
        super().__init__("slam_path_sector_visualiser")

        # ---- Parameters ----
        self.declare_parameter("topics.odom_in", "/slam/odom")
        self.declare_parameter("topics.map_in", "/slam/map_cones")
        self.declare_parameter("topics.path_out", "/path_points")

        self.declare_parameter("qos.best_effort", True)
        self.declare_parameter("qos.depth", 50)

        # Sector FOV parameters
        self.declare_parameter("fov.vertex_offset_m", -5.0)
        self.declare_parameter("fov.radius_m", 30.0)
        self.declare_parameter("fov.angle_deg", 60.0)

        # Candidate filtering
        self.declare_parameter("candidates.min_spacing_m", 1.0)
        self.declare_parameter("candidates.min_car_dist_m", 2.0)

        # Greedy path
        self.declare_parameter("greedy.max_hop_m", 7.0)

        # Publish rate (Hz)
        self.declare_parameter("publish.rate_hz", 20.0)

        gp = self.get_parameter
        odom_topic = str(gp("topics.odom_in").value)
        map_topic = str(gp("topics.map_in").value)
        path_topic = str(gp("topics.path_out").value)

        best_effort = bool(gp("qos.best_effort").value)
        depth = int(gp("qos.depth").value)

        self.vertex_offset = float(gp("fov.vertex_offset_m").value)
        self.fov_radius = float(gp("fov.radius_m").value)
        self.fov_angle_deg = float(gp("fov.angle_deg").value)
        self.fov_half_rad = math.radians(self.fov_angle_deg * 0.5)

        self.min_spacing = float(gp("candidates.min_spacing_m").value)
        self.min_car_dist = float(gp("candidates.min_car_dist_m").value)

        self.max_hop = float(gp("greedy.max_hop_m").value)
        self.publish_rate_hz = float(gp("publish.rate_hz").value)

        # Subscriber QoS
        sub_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=depth,
            reliability=(
                QoSReliabilityPolicy.BEST_EFFORT
                if best_effort
                else QoSReliabilityPolicy.RELIABLE
            ),
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # Publisher QoS (FORCED BEST_EFFORT)
        pub_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=depth,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # ---- Subscriptions ----
        self.create_subscription(Odometry, odom_topic, self.cb_odom, sub_qos)
        self.create_subscription(ConeArrayWithCovariance, map_topic, self.cb_map, sub_qos)

        # ---- Publisher ----
        self.path_pub = self.create_publisher(Path, path_topic, pub_qos)

        # ---- State ----
        self.data_lock = threading.Lock()

        self.car_x = None
        self.car_y = None
        self.car_yaw = None  # radians

        self.blue_global: List[Tuple[float, float]] = []
        self.yellow_global: List[Tuple[float, float]] = []
        self.orange_global: List[Tuple[float, float]] = []
        self.big_global: List[Tuple[float, float]] = []

        if not _HAS_SCIPY:
            self.get_logger().warn(
                "[slam_path_sector_visualiser] SciPy not found; falling back to k-NN graph instead of Delaunay."
            )

        self.get_logger().info(
            f"[slam_path_sector_visualiser] odom={odom_topic}, map={map_topic}, path_out={path_topic} | "
            f"vertex_offset={self.vertex_offset}m radius={self.fov_radius}m angle={self.fov_angle_deg}deg | "
            f"min_spacing={self.min_spacing}m min_car_dist={self.min_car_dist}m max_hop={self.max_hop}m"
        )

        # ---- Timer loop ----
        period = 1.0 / max(self.publish_rate_hz, 1e-6)
        self.timer = self.create_timer(period, self._tick)

    # --------------------- Callbacks ---------------------

    def cb_odom(self, msg: Odometry):
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

        with self.data_lock:
            self.car_x = x
            self.car_y = y
            self.car_yaw = yaw

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

    # --------------------- Helpers ---------------------

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
        Select cones within the circular sector FOV.
        Returns (points_window, classes_window).
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
        Build graph edges using Delaunay (if available) or k-NN, then apply constraints.
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

    def _big_cone_candidate(self, points_window, classes_window):
        """
        BIG ORANGE SPECIAL:
          - If exactly 2 big cones in FOV: return their midpoint (single candidate)
          - If >= 3 big cones in FOV: return centroid of all big cones (single candidate)
        Returns: List[(x, y, is_big_special)]
        """
        big_pts = [
            points_window[i]
            for i, cls in enumerate(classes_window)
            if cls == "big"
        ]
        n = len(big_pts)
        if n < 2:
            return []

        if n == 2:
            (x1, y1), (x2, y2) = big_pts
            mx = 0.5 * (x1 + x2)
            my = 0.5 * (y1 + y2)
            return [(mx, my, True)]

        # n >= 3
        xs = [p[0] for p in big_pts]
        ys = [p[1] for p in big_pts]
        cx = sum(xs) / n
        cy = sum(ys) / n
        return [(cx, cy, True)]

    def _filter_centroids_min_spacing(
        self,
        centroids_raw: List[Tuple[float, float, bool]],
        car_x: float,
        car_y: float,
        min_dist: float,
    ) -> List[Tuple[float, float, bool]]:
        """
        Strategy:
          - sort candidates by distance from car DESC (farthest-first)
          - keep if:
              * >= min_car_dist from car
              * >= min_dist from all already kept
        """
        if not centroids_raw:
            return []

        pts = np.array([[c[0], c[1]] for c in centroids_raw], dtype=float)
        car = np.array([car_x, car_y], dtype=float)
        d2_car = np.sum((pts - car) ** 2, axis=1)
        order = np.argsort(-d2_car)

        kept_pts = []
        kept = []

        for idx in order:
            p = pts[idx]

            if np.linalg.norm(p - car) < self.min_car_dist:
                continue

            if not kept_pts:
                kept_pts.append(p)
                kept.append((float(p[0]), float(p[1]), bool(centroids_raw[idx][2])))
                continue

            too_close = False
            for kp in kept_pts:
                if np.linalg.norm(p - kp) < min_dist:
                    too_close = True
                    break
            if too_close:
                continue

            kept_pts.append(p)
            kept.append((float(p[0]), float(p[1]), bool(centroids_raw[idx][2])))

        return kept

    def _build_greedy_path(self, car_x, car_y, candidate_points: List[Tuple[float, float]]):
        """Greedy NN chain with max hop length self.max_hop."""
        if not candidate_points:
            return []

        pts = np.asarray(candidate_points, dtype=float)
        N = pts.shape[0]
        used = np.zeros(N, dtype=bool)

        current = np.array([car_x, car_y], dtype=float)
        max_step2 = self.max_hop * self.max_hop

        order: List[int] = []

        for _ in range(N):
            diff = pts - current
            d2 = np.sum(diff * diff, axis=1)
            d2[used] = float("inf")
            d2[d2 > max_step2] = float("inf")

            idx = int(np.argmin(d2))
            if not math.isfinite(float(d2[idx])) or float(d2[idx]) == float("inf"):
                break

            used[idx] = True
            order.append(idx)
            current = pts[idx]

        return [tuple(pts[i]) for i in order]

    def _publish_path(self, car_x: float, car_y: float, path_points: List[Tuple[float, float]]):
        """Publish nav_msgs/Path (car pose first, then greedy points)."""
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"  # adjust if needed

        poses: List[PoseStamped] = []

        car_ps = PoseStamped()
        car_ps.header = msg.header
        car_ps.pose.position.x = float(car_x)
        car_ps.pose.position.y = float(car_y)
        car_ps.pose.position.z = 0.0
        car_ps.pose.orientation.w = 1.0
        poses.append(car_ps)

        for (x, y) in path_points:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            poses.append(ps)

        msg.poses = poses
        self.path_pub.publish(msg)

    # --------------------- Main loop tick ---------------------

    def _tick(self):
        with self.data_lock:
            car_x = self.car_x
            car_y = self.car_y
            car_yaw = self.car_yaw

            blue_global = list(self.blue_global)
            yellow_global = list(self.yellow_global)
            orange_global = list(self.orange_global)
            big_global = list(self.big_global)

        if car_x is None or car_y is None or car_yaw is None:
            return

        # stash into class fields for FOV helper usage (so we reuse your exact logic style)
        with self.data_lock:
            self.blue_global = blue_global
            self.yellow_global = yellow_global
            self.orange_global = orange_global
            self.big_global = big_global

        points_window, classes_window = self._compute_fov_points(car_x, car_y, car_yaw)

        if len(points_window) < 2:
            self._publish_path(car_x, car_y, [])
            return

        edges = self._build_edges(points_window, classes_window)

        orange_like = {"orange", "big"}

        # BIG ORANGE SPECIAL candidate
        big_special = self._big_cone_candidate(points_window, classes_window)

        # Collect candidates from edges (midpoints), excluding big-big edges
        centroids_raw: List[Tuple[float, float, bool]] = []

        for (i, j) in edges:
            (x1, y1) = points_window[i]
            (x2, y2) = points_window[j]
            ci = classes_window[i]
            cj = classes_window[j]

            # skip big-big midpoint; handled by special logic instead
            if ci == "big" and cj == "big":
                continue

            mx = 0.5 * (x1 + x2)
            my = 0.5 * (y1 + y2)
            centroids_raw.append((mx, my, False))

        # Add the big special point (if present)
        for (bx, by, _) in big_special:
            centroids_raw.append((bx, by, True))

        # Spacing + car-distance filtering
        centroids = self._filter_centroids_min_spacing(
            centroids_raw, car_x, car_y, min_dist=self.min_spacing
        )

        candidate_points = [(x, y) for (x, y, _) in centroids]

        path_points = self._build_greedy_path(car_x, car_y, candidate_points)

        self._publish_path(car_x, car_y, path_points)


def main():
    rclpy.init()
    node = SlamPathSectorVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
