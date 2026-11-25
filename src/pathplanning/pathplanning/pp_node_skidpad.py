# #!/usr/bin/env python3
# # slam_path_sector_visualiser.py
# #
# # Sector-FOV path generator for SLAM cone map (NO PLOTTER).
# #   - Subscribes to /slam/odom (nav_msgs/Odometry)
# #   - Subscribes to /slam/map_cones (eufs_msgs/ConeArrayWithCovariance)
# #   - Inside the FOV:
# #       * Build Delaunay (or k-NN) edges on cones
# #       * Apply constraints:
# #           - edge length < 6 m
# #           - drop blue-blue and yellow-yellow edges
# #           - drop any orange-like ↔ (blue or yellow) edges
# #           - keep orange-like ↔ orange-like and blue-yellow
# #       * For each allowed edge (EXCEPT big-big), compute midpoint as candidate
# #       * BIG ORANGE SPECIAL:
# #           - If exactly 2 big-orange cones in FOV: output ONLY their midpoint (as a special candidate)
# #           - If >= 3 big-orange cones in FOV: output centroid of ALL big-orange cones (as a special candidate)
# #       * Enforce ≥ 1 m spacing between candidates, keeping farther-first
# #       * Drop any candidate within min_car_dist (default 2.0 m) of the car pose
# #       * Build a greedy NN path from car through kept candidates,
# #         limited by max hop distance (default 7 m)
# #   - Publishes greedy path to /path_points (nav_msgs/Path) BEST_EFFORT QoS
# #
# import math
# import threading
# from typing import List, Tuple

# import numpy as np

# import rclpy
# from rclpy.node import Node
# from rclpy.qos import (
#     QoSProfile,
#     QoSHistoryPolicy,
#     QoSReliabilityPolicy,
#     QoSDurabilityPolicy,
# )

# from nav_msgs.msg import Odometry, Path
# from geometry_msgs.msg import PoseStamped
# from eufs_msgs.msg import ConeArrayWithCovariance

# # Optional SciPy-based Delaunay
# try:
#     from scipy.spatial import Delaunay
#     _HAS_SCIPY = True
# except ImportError:
#     Delaunay = None
#     _HAS_SCIPY = False


# def yaw_from_quat(qx, qy, qz, qw) -> float:
#     siny_cosp = 2.0 * (qw * qz + qx * qy)
#     cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
#     return math.atan2(siny_cosp, cosy_cosp)


# class SlamPathSectorVisualizer(Node):
#     def __init__(self):
#         super().__init__("slam_path_sector_visualiser")

#         # ---- Parameters ----
#         self.declare_parameter("topics.odom_in", "/slam/odom")
#         self.declare_parameter("topics.map_in", "/slam/map_cones")
#         self.declare_parameter("topics.path_out", "/path_points")

#         self.declare_parameter("qos.best_effort", True)
#         self.declare_parameter("qos.depth", 50)

#         # Sector FOV parameters
#         self.declare_parameter("fov.vertex_offset_m", -5.0)
#         self.declare_parameter("fov.radius_m", 30.0)
#         self.declare_parameter("fov.angle_deg", 60.0)

#         # Candidate filtering
#         self.declare_parameter("candidates.min_spacing_m", 1.0)
#         self.declare_parameter("candidates.min_car_dist_m", 2.0)

#         # Greedy path
#         self.declare_parameter("greedy.max_hop_m", 7.0)

#         # Publish rate (Hz)
#         self.declare_parameter("publish.rate_hz", 20.0)

#         gp = self.get_parameter
#         odom_topic = str(gp("topics.odom_in").value)
#         map_topic = str(gp("topics.map_in").value)
#         path_topic = str(gp("topics.path_out").value)

#         best_effort = bool(gp("qos.best_effort").value)
#         depth = int(gp("qos.depth").value)

#         self.vertex_offset = float(gp("fov.vertex_offset_m").value)
#         self.fov_radius = float(gp("fov.radius_m").value)
#         self.fov_angle_deg = float(gp("fov.angle_deg").value)
#         self.fov_half_rad = math.radians(self.fov_angle_deg * 0.5)

#         self.min_spacing = float(gp("candidates.min_spacing_m").value)
#         self.min_car_dist = float(gp("candidates.min_car_dist_m").value)

#         self.max_hop = float(gp("greedy.max_hop_m").value)
#         self.publish_rate_hz = float(gp("publish.rate_hz").value)

#         # Subscriber QoS
#         sub_qos = QoSProfile(
#             history=QoSHistoryPolicy.KEEP_LAST,
#             depth=depth,
#             reliability=(
#                 QoSReliabilityPolicy.BEST_EFFORT
#                 if best_effort
#                 else QoSReliabilityPolicy.RELIABLE
#             ),
#             durability=QoSDurabilityPolicy.VOLATILE,
#         )

#         # Publisher QoS (FORCED BEST_EFFORT)
#         pub_qos = QoSProfile(
#             history=QoSHistoryPolicy.KEEP_LAST,
#             depth=depth,
#             reliability=QoSReliabilityPolicy.BEST_EFFORT,
#             durability=QoSDurabilityPolicy.VOLATILE,
#         )

#         # ---- Subscriptions ----
#         self.create_subscription(Odometry, odom_topic, self.cb_odom, sub_qos)
#         self.create_subscription(ConeArrayWithCovariance, map_topic, self.cb_map, sub_qos)

#         # ---- Publisher ----
#         self.path_pub = self.create_publisher(Path, path_topic, pub_qos)

#         # ---- State ----
#         self.data_lock = threading.Lock()

#         self.car_x = None
#         self.car_y = None
#         self.car_yaw = None  # radians

#         self.blue_global: List[Tuple[float, float]] = []
#         self.yellow_global: List[Tuple[float, float]] = []
#         self.orange_global: List[Tuple[float, float]] = []
#         self.big_global: List[Tuple[float, float]] = []

#         if not _HAS_SCIPY:
#             self.get_logger().warn(
#                 "[slam_path_sector_visualiser] SciPy not found; falling back to k-NN graph instead of Delaunay."
#             )

#         self.get_logger().info(
#             f"[slam_path_sector_visualiser] odom={odom_topic}, map={map_topic}, path_out={path_topic} | "
#             f"vertex_offset={self.vertex_offset}m radius={self.fov_radius}m angle={self.fov_angle_deg}deg | "
#             f"min_spacing={self.min_spacing}m min_car_dist={self.min_car_dist}m max_hop={self.max_hop}m"
#         )

#         # ---- Timer loop ----
#         period = 1.0 / max(self.publish_rate_hz, 1e-6)
#         self.timer = self.create_timer(period, self._tick)

#     # --------------------- Callbacks ---------------------

#     def cb_odom(self, msg: Odometry):
#         x = float(msg.pose.pose.position.x)
#         y = float(msg.pose.pose.position.y)
#         q = msg.pose.pose.orientation
#         yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

#         with self.data_lock:
#             self.car_x = x
#             self.car_y = y
#             self.car_yaw = yaw

#     def cb_map(self, msg: ConeArrayWithCovariance):
#         blue, yellow, orange, big = [], [], [], []

#         for c in msg.blue_cones:
#             blue.append((float(c.point.x), float(c.point.y)))
#         for c in msg.yellow_cones:
#             yellow.append((float(c.point.x), float(c.point.y)))
#         for c in msg.orange_cones:
#             orange.append((float(c.point.x), float(c.point.y)))
#         for c in msg.big_orange_cones:
#             big.append((float(c.point.x), float(c.point.y)))

#         with self.data_lock:
#             self.blue_global = blue
#             self.yellow_global = yellow
#             self.orange_global = orange
#             self.big_global = big

#     # --------------------- Helpers ---------------------

#     def _world_to_local(self, car_x, car_y, car_yaw, px, py):
#         """Transform world -> car frame (x forward, y left)."""
#         c = math.cos(car_yaw)
#         s = math.sin(car_yaw)
#         dx = px - car_x
#         dy = py - car_y
#         lx = c * dx + s * dy
#         ly = -s * dx + c * dy
#         return lx, ly

#     def _compute_fov_points(self, car_x, car_y, car_yaw):
#         """
#         Select cones within the circular sector FOV.
#         Returns (points_window, classes_window).
#         """
#         vertex_offset = self.vertex_offset
#         R = self.fov_radius
#         half_angle = self.fov_half_rad

#         points_window: List[Tuple[float, float]] = []
#         classes_window: List[str] = []

#         def add_list(cones: List[Tuple[float, float]], cls_name: str):
#             nonlocal points_window, classes_window
#             for (px, py) in cones:
#                 lx, ly = self._world_to_local(car_x, car_y, car_yaw, px, py)
#                 vx = lx - vertex_offset
#                 vy = ly
#                 r = math.hypot(vx, vy)
#                 if r > R or r < 1e-3:
#                     continue
#                 theta = math.atan2(vy, vx)
#                 if abs(theta) <= half_angle:
#                     points_window.append((px, py))
#                     classes_window.append(cls_name)

#         add_list(self.blue_global, "blue")
#         add_list(self.yellow_global, "yellow")
#         add_list(self.orange_global, "orange")
#         add_list(self.big_global, "big")

#         return points_window, classes_window

#     def _build_edges(self, points_window, classes_window):
#         """
#         Build graph edges using Delaunay (if available) or k-NN, then apply constraints.
#         """
#         N = len(points_window)
#         if N < 2:
#             return []

#         pts = np.asarray(points_window, dtype=float)
#         edges_set = set()

#         if _HAS_SCIPY and N >= 3:
#             tri = Delaunay(pts)
#             for simplex in tri.simplices:
#                 i, j, k = int(simplex[0]), int(simplex[1]), int(simplex[2])
#                 for a, b in ((i, j), (j, k), (k, i)):
#                     if a > b:
#                         a, b = b, a
#                     edges_set.add((a, b))
#         else:
#             k = min(3, N - 1)
#             for i in range(N):
#                 d2 = np.sum((pts - pts[i]) ** 2, axis=1)
#                 d2[i] = float("inf")
#                 nn_idx = np.argsort(d2)[:k]
#                 for j in nn_idx:
#                     a, b = i, int(j)
#                     if a > b:
#                         a, b = b, a
#                     edges_set.add((a, b))

#         orange_like = {"orange", "big"}
#         blue_yellow = {"blue", "yellow"}

#         edges = []
#         max_len2 = 6.0 * 6.0

#         for (i, j) in edges_set:
#             ci = classes_window[i]
#             cj = classes_window[j]

#             x1, y1 = points_window[i]
#             x2, y2 = points_window[j]
#             d2 = (x1 - x2) ** 2 + (y1 - y2) ** 2
#             if d2 >= max_len2:
#                 continue

#             if (ci == "blue" and cj == "blue") or (ci == "yellow" and cj == "yellow"):
#                 continue

#             if ((ci in orange_like and cj in blue_yellow) or
#                (cj in orange_like and ci in blue_yellow)):
#                 continue

#             edges.append((i, j))

#         return edges

#     def _big_cone_candidate(self, points_window, classes_window):
#         """
#         BIG ORANGE SPECIAL:
#           - If exactly 2 big cones in FOV: return their midpoint (single candidate)
#           - If >= 3 big cones in FOV: return centroid of all big cones (single candidate)
#         Returns: List[(x, y, is_big_special)]
#         """
#         big_pts = [
#             points_window[i]
#             for i, cls in enumerate(classes_window)
#             if cls == "big"
#         ]
#         n = len(big_pts)
#         if n < 2:
#             return []

#         if n == 2:
#             (x1, y1), (x2, y2) = big_pts
#             mx = 0.5 * (x1 + x2)
#             my = 0.5 * (y1 + y2)
#             return [(mx, my, True)]

#         # n >= 3
#         xs = [p[0] for p in big_pts]
#         ys = [p[1] for p in big_pts]
#         cx = sum(xs) / n
#         cy = sum(ys) / n
#         return [(cx, cy, True)]

#     def _filter_centroids_min_spacing(
#         self,
#         centroids_raw: List[Tuple[float, float, bool]],
#         car_x: float,
#         car_y: float,
#         min_dist: float,
#     ) -> List[Tuple[float, float, bool]]:
#         """
#         Strategy:
#           - sort candidates by distance from car DESC (farthest-first)
#           - keep if:
#               * >= min_car_dist from car
#               * >= min_dist from all already kept
#         """
#         if not centroids_raw:
#             return []

#         pts = np.array([[c[0], c[1]] for c in centroids_raw], dtype=float)
#         car = np.array([car_x, car_y], dtype=float)
#         d2_car = np.sum((pts - car) ** 2, axis=1)
#         order = np.argsort(-d2_car)

#         kept_pts = []
#         kept = []

#         for idx in order:
#             p = pts[idx]

#             if np.linalg.norm(p - car) < self.min_car_dist:
#                 continue

#             if not kept_pts:
#                 kept_pts.append(p)
#                 kept.append((float(p[0]), float(p[1]), bool(centroids_raw[idx][2])))
#                 continue

#             too_close = False
#             for kp in kept_pts:
#                 if np.linalg.norm(p - kp) < min_dist:
#                     too_close = True
#                     break
#             if too_close:
#                 continue

#             kept_pts.append(p)
#             kept.append((float(p[0]), float(p[1]), bool(centroids_raw[idx][2])))

#         return kept

#     def _build_greedy_path(self, car_x, car_y, candidate_points: List[Tuple[float, float]]):
#         """Greedy NN chain with max hop length self.max_hop."""
#         if not candidate_points:
#             return []

#         pts = np.asarray(candidate_points, dtype=float)
#         N = pts.shape[0]
#         used = np.zeros(N, dtype=bool)

#         current = np.array([car_x, car_y], dtype=float)
#         max_step2 = self.max_hop * self.max_hop

#         order: List[int] = []

#         for _ in range(N):
#             diff = pts - current
#             d2 = np.sum(diff * diff, axis=1)
#             d2[used] = float("inf")
#             d2[d2 > max_step2] = float("inf")

#             idx = int(np.argmin(d2))
#             if not math.isfinite(float(d2[idx])) or float(d2[idx]) == float("inf"):
#                 break

#             used[idx] = True
#             order.append(idx)
#             current = pts[idx]

#         return [tuple(pts[i]) for i in order]

#     def _publish_path(self, car_x: float, car_y: float, path_points: List[Tuple[float, float]]):
#         """Publish nav_msgs/Path (car pose first, then greedy points)."""
#         msg = Path()
#         msg.header.stamp = self.get_clock().now().to_msg()
#         msg.header.frame_id = "map"  # adjust if needed

#         poses: List[PoseStamped] = []

#         car_ps = PoseStamped()
#         car_ps.header = msg.header
#         car_ps.pose.position.x = float(car_x)
#         car_ps.pose.position.y = float(car_y)
#         car_ps.pose.position.z = 0.0
#         car_ps.pose.orientation.w = 1.0
#         poses.append(car_ps)

#         for (x, y) in path_points:
#             ps = PoseStamped()
#             ps.header = msg.header
#             ps.pose.position.x = float(x)
#             ps.pose.position.y = float(y)
#             ps.pose.position.z = 0.0
#             ps.pose.orientation.w = 1.0
#             poses.append(ps)

#         msg.poses = poses
#         self.path_pub.publish(msg)

#     # --------------------- Main loop tick ---------------------

#     def _tick(self):
#         with self.data_lock:
#             car_x = self.car_x
#             car_y = self.car_y
#             car_yaw = self.car_yaw

#             blue_global = list(self.blue_global)
#             yellow_global = list(self.yellow_global)
#             orange_global = list(self.orange_global)
#             big_global = list(self.big_global)

#         if car_x is None or car_y is None or car_yaw is None:
#             return

#         # stash into class fields for FOV helper usage (so we reuse your exact logic style)
#         with self.data_lock:
#             self.blue_global = blue_global
#             self.yellow_global = yellow_global
#             self.orange_global = orange_global
#             self.big_global = big_global

#         points_window, classes_window = self._compute_fov_points(car_x, car_y, car_yaw)

#         if len(points_window) < 2:
#             self._publish_path(car_x, car_y, [])
#             return

#         edges = self._build_edges(points_window, classes_window)

#         orange_like = {"orange", "big"}

#         # BIG ORANGE SPECIAL candidate
#         big_special = self._big_cone_candidate(points_window, classes_window)

#         # Collect candidates from edges (midpoints), excluding big-big edges
#         centroids_raw: List[Tuple[float, float, bool]] = []

#         for (i, j) in edges:
#             (x1, y1) = points_window[i]
#             (x2, y2) = points_window[j]
#             ci = classes_window[i]
#             cj = classes_window[j]

#             # skip big-big midpoint; handled by special logic instead
#             if ci == "big" and cj == "big":
#                 continue

#             mx = 0.5 * (x1 + x2)
#             my = 0.5 * (y1 + y2)
#             centroids_raw.append((mx, my, False))

#         # Add the big special point (if present)
#         for (bx, by, _) in big_special:
#             centroids_raw.append((bx, by, True))

#         # Spacing + car-distance filtering
#         centroids = self._filter_centroids_min_spacing(
#             centroids_raw, car_x, car_y, min_dist=self.min_spacing
#         )

#         candidate_points = [(x, y) for (x, y, _) in centroids]

#         path_points = self._build_greedy_path(car_x, car_y, candidate_points)

#         self._publish_path(car_x, car_y, path_points)


# def main():
#     rclpy.init()
#     node = SlamPathSectorVisualizer()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == "__main__":
#     main()


#another program (fov rotation)

#another program 


#!/usr/bin/env python3


# og code where only one virtal infront of car.
# challenges: overlap the boundary when too close.

# ============================================================
# pp_node_skidpad.py
#
# SLAM-based skidpad path planner with live visualization.
#
# NORMAL MODE:
#   If BLUE > 1 AND YELLOW > 1:
#       - Normal Delaunay on all FOV cones
#       - Original cone-type constraints
#       - Edge midpoints
#       - Big-orange special candidate
#       - Candidate filtering
#       - Greedy path
#
# FALLBACK MODE:
#   If BLUE <= 1 OR YELLOW <= 1:
#       - Ignore the side having <= 1 cone
#       - Select the side having more cones
#       - Require selected side to have >= 2 cones
#       - Add virtual colourless cone in front of car
#       - Delaunay(selected side + virtual cone)
#       - Use edge midpoints as candidates
#       - Candidate filtering
#       - Greedy path
#
# LIVE VISUALIZATION:
#   - Blue cones
#   - Yellow cones
#   - Orange cones
#   - Big-orange cones
#   - Virtual colourless cone
#   - FOV sector
#   - Delaunay edges
#   - Candidate points
#   - Final greedy path
#   - Car position + heading
#   - NORMAL / FALLBACK status
#
# ============================================================


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


# ============================================================
# Matplotlib
# ============================================================

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge
except ImportError:
    plt = None


# ============================================================
# Optional SciPy Delaunay
# ============================================================

try:

    from scipy.spatial import Delaunay, QhullError

    _HAS_SCIPY = True

except ImportError:

    Delaunay = None
    _HAS_SCIPY = False


# ============================================================
# Quaternion -> yaw
# ============================================================

def yaw_from_quat(
    qx,
    qy,
    qz,
    qw
) -> float:

    siny_cosp = (
        2.0 *
        (
            qw * qz
            +
            qx * qy
        )
    )

    cosy_cosp = (
        1.0
        -
        2.0 *
        (
            qy * qy
            +
            qz * qz
        )
    )

    return math.atan2(
        siny_cosp,
        cosy_cosp
    )


# ============================================================
# MAIN NODE
# ============================================================

class SlamPathSectorVisualizer(Node):

    def __init__(self):

        super().__init__(
            "slam_path_sector_visualiser"
        )

        # ====================================================
        # PARAMETERS
        # ====================================================

        self.declare_parameter(
            "topics.odom_in",
            "/slam/odom"
        )

        self.declare_parameter(
            "topics.map_in",
            "/slam/map_cones"
        )

        self.declare_parameter(
            "topics.path_out",
            "/path_points"
        )

        # ----------------------------------------------------
        # QoS
        # ----------------------------------------------------

        self.declare_parameter(
            "qos.best_effort",
            True
        )

        self.declare_parameter(
            "qos.depth",
            50
        )

        # ----------------------------------------------------
        # FOV
        # ----------------------------------------------------

        self.declare_parameter(
            "fov.vertex_offset_m",
            -5.0
        )

        self.declare_parameter(
            "fov.radius_m",
            30.0
        )

        self.declare_parameter(
            "fov.angle_deg",
            60.0
        )

        # ----------------------------------------------------
        # Candidate filtering
        # ----------------------------------------------------

        self.declare_parameter(
            "candidates.min_spacing_m",
            1.0
        )

        self.declare_parameter(
            "candidates.min_car_dist_m",
            2.0
        )

        # ----------------------------------------------------
        # Greedy path
        # ----------------------------------------------------

        self.declare_parameter(
            "greedy.max_hop_m",
            7.0
        )

        # ----------------------------------------------------
        # Publish
        # ----------------------------------------------------

        self.declare_parameter(
            "publish.rate_hz",
            20.0
        )

        # ----------------------------------------------------
        # Virtual colourless cone
        # ----------------------------------------------------

        self.declare_parameter(
            "fallback.virtual_cone_distance_m",
            1.0
        )

        # ====================================================
        # READ PARAMETERS
        # ====================================================

        gp = self.get_parameter

        odom_topic = str(
            gp(
                "topics.odom_in"
            ).value
        )

        map_topic = str(
            gp(
                "topics.map_in"
            ).value
        )

        path_topic = str(
            gp(
                "topics.path_out"
            ).value
        )

        best_effort = bool(
            gp(
                "qos.best_effort"
            ).value
        )

        depth = int(
            gp(
                "qos.depth"
            ).value
        )

        self.vertex_offset = float(
            gp(
                "fov.vertex_offset_m"
            ).value
        )

        self.fov_radius = float(
            gp(
                "fov.radius_m"
            ).value
        )

        self.fov_angle_deg = float(
            gp(
                "fov.angle_deg"
            ).value
        )

        self.fov_half_rad = math.radians(
            self.fov_angle_deg * 0.5
        )

        self.min_spacing = float(
            gp(
                "candidates.min_spacing_m"
            ).value
        )

        self.min_car_dist = float(
            gp(
                "candidates.min_car_dist_m"
            ).value
        )

        self.max_hop = float(
            gp(
                "greedy.max_hop_m"
            ).value
        )

        self.publish_rate_hz = float(
            gp(
                "publish.rate_hz"
            ).value
        )

        self.virtual_cone_distance = float(
            gp(
                "fallback.virtual_cone_distance_m"
            ).value
        )

        # ====================================================
        # SUBSCRIBER QoS
        # ====================================================

        sub_qos = QoSProfile(

            history=QoSHistoryPolicy.KEEP_LAST,

            depth=depth,

            reliability=(
                QoSReliabilityPolicy.BEST_EFFORT
                if best_effort
                else QoSReliabilityPolicy.RELIABLE
            ),

            durability=(
                QoSDurabilityPolicy.VOLATILE
            ),
        )

        # ====================================================
        # PUBLISHER QoS
        # ====================================================

        pub_qos = QoSProfile(

            history=QoSHistoryPolicy.KEEP_LAST,

            depth=depth,

            reliability=(
                QoSReliabilityPolicy.BEST_EFFORT
            ),

            durability=(
                QoSDurabilityPolicy.VOLATILE
            ),
        )

        # ====================================================
        # SUBSCRIBERS
        # ====================================================

        self.create_subscription(
            Odometry,
            odom_topic,
            self.cb_odom,
            sub_qos
        )

        self.create_subscription(
            ConeArrayWithCovariance,
            map_topic,
            self.cb_map,
            sub_qos
        )

        # ====================================================
        # PUBLISHER
        # ====================================================

        self.path_pub = (
            self.create_publisher(
                Path,
                path_topic,
                pub_qos
            )
        )

        # ====================================================
        # STATE
        # ====================================================

        self.data_lock = threading.Lock()

        self.car_x = None
        self.car_y = None
        self.car_yaw = None

        # Global cone positions
        self.blue_global = []
        self.yellow_global = []
        self.orange_global = []
        self.big_global = []

        # ====================================================
        # LIVE VISUALIZATION STATE
        # ====================================================

        self.plot_lock = threading.Lock()

        self.plot_points_window = []
        self.plot_classes_window = []

        self.plot_edges = []

        self.plot_edge_points = []

        self.plot_candidates = []

        self.plot_path = []

        self.plot_virtual_cone = None

        self.plot_mode = "WAITING"

        self.plot_selected_side = "None"

        self.plot_blue_count = 0
        self.plot_yellow_count = 0

        # ====================================================
        # INITIALIZE PLOT
        # ====================================================

        self.fig = None
        if plt is not None and plt.get_backend().lower() != "agg":
            try:
                self._init_live_plot()
            except Exception as exc:
                self.get_logger().warn(f"Live plot unavailable: {exc}")
                if self.fig is not None:
                    plt.close(self.fig)
                self.fig = None

        # ====================================================
        # SCIpy warning
        # ====================================================

        if not _HAS_SCIPY:

            self.get_logger().warn(
                "SciPy not found. "
                "Delaunay will use k-NN fallback."
            )

        # ====================================================
        # LOG
        # ====================================================

        self.get_logger().info(
            f"odom={odom_topic}, "
            f"map={map_topic}, "
            f"path_out={path_topic}"
        )

        self.get_logger().info(
            f"FOV: radius={self.fov_radius} m, "
            f"angle={self.fov_angle_deg} deg"
        )

        self.get_logger().info(
            f"Virtual cone distance="
            f"{self.virtual_cone_distance} m"
        )

        # ====================================================
        # TIMER
        # ====================================================

        period = (
            1.0 /
            max(
                self.publish_rate_hz,
                1e-6
            )
        )

        self.timer = self.create_timer(
            period,
            self._tick
        )

    # ========================================================
    # LIVE PLOT INITIALIZATION
    # ========================================================

    def _init_live_plot(self):

        """
        Initialize the live Matplotlib visualization.

        This function does NOT affect the planner.
        """

        plt.ion()

        self.fig, self.ax = plt.subplots(
            figsize=(10, 8)
        )

        self.fig.canvas.manager.set_window_title(
            "SLAM Skidpad Path Planner - Live Visualization"
        )

        self.ax.set_aspect(
            "equal",
            adjustable="box"
        )

        self.ax.set_xlabel(
            "X [m]"
        )

        self.ax.set_ylabel(
            "Y [m]"
        )

        self.ax.set_title(
            "SLAM Skidpad Path Planner"
        )

        self.ax.grid(
            True,
            alpha=0.25
        )

        # ----------------------------------------------------
        # Plot handles
        # ----------------------------------------------------

        self.scatter_blue = (
            self.ax.scatter(
                [],
                [],
                s=45,
                marker="o",
                label="Blue"
            )
        )

        self.scatter_yellow = (
            self.ax.scatter(
                [],
                [],
                s=45,
                marker="o",
                label="Yellow"
            )
        )

        self.scatter_orange = (
            self.ax.scatter(
                [],
                [],
                s=50,
                marker="o",
                label="Orange"
            )
        )

        self.scatter_big = (
            self.ax.scatter(
                [],
                [],
                s=90,
                marker="s",
                label="Big Orange"
            )
        )

        # ----------------------------------------------------
        # Virtual cone
        # ----------------------------------------------------

        self.scatter_virtual = (
            self.ax.scatter(
                [],
                [],
                s=110,
                marker="X",
                label="Virtual Colourless"
            )
        )

        # ----------------------------------------------------
        # Candidate points
        # ----------------------------------------------------

        self.scatter_candidates = (
            self.ax.scatter(
                [],
                [],
                s=60,
                marker="x",
                label="Candidates"
            )
        )

        # ----------------------------------------------------
        # Car
        # ----------------------------------------------------

        self.scatter_car = (
            self.ax.scatter(
                [],
                [],
                s=120,
                marker="^",
                label="Car"
            )
        )

        # ----------------------------------------------------
        # Final path
        # ----------------------------------------------------

        self.path_line, = (
            self.ax.plot(
                [],
                [],
                linewidth=2.5,
                label="Greedy Path"
            )
        )

        # ----------------------------------------------------
        # Car heading
        # ----------------------------------------------------

        self.heading_line, = (
            self.ax.plot(
                [],
                [],
                linewidth=2.0
            )
        )

        # ----------------------------------------------------
        # FOV wedge
        # ----------------------------------------------------

        self.fov_patch = None

        # ----------------------------------------------------
        # Delaunay edge lines
        # ----------------------------------------------------

        self.edge_lines = []

        # ----------------------------------------------------
        # Status text
        # ----------------------------------------------------

        self.status_text = (
            self.ax.text(
                0.02,
                0.98,
                "",
                transform=self.ax.transAxes,
                verticalalignment="top",
                fontsize=11,
                bbox=dict(
                    boxstyle="round",
                    facecolor="white",
                    alpha=0.85
                )
            )
        )

        self.ax.legend(
            loc="upper right"
        )

        plt.show(
            block=False
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # ========================================================
    # ODOMETRY CALLBACK
    # ========================================================

    def cb_odom(
        self,
        msg: Odometry
    ):

        x = float(
            msg.pose.pose.position.x
        )

        y = float(
            msg.pose.pose.position.y
        )

        q = (
            msg.pose.pose.orientation
        )

        yaw = yaw_from_quat(
            q.x,
            q.y,
            q.z,
            q.w
        )

        with self.data_lock:

            self.car_x = x
            self.car_y = y
            self.car_yaw = yaw

    # ========================================================
    # MAP CALLBACK
    # ========================================================

    def cb_map(
        self,
        msg: ConeArrayWithCovariance
    ):

        blue = []
        yellow = []
        orange = []
        big = []

        # ----------------------------------------------------
        # Blue
        # ----------------------------------------------------

        for c in msg.blue_cones:

            blue.append(
                (
                    float(c.point.x),
                    float(c.point.y)
                )
            )

        # ----------------------------------------------------
        # Yellow
        # ----------------------------------------------------

        for c in msg.yellow_cones:

            yellow.append(
                (
                    float(c.point.x),
                    float(c.point.y)
                )
            )

        # ----------------------------------------------------
        # Orange
        # ----------------------------------------------------

        for c in msg.orange_cones:

            orange.append(
                (
                    float(c.point.x),
                    float(c.point.y)
                )
            )

        # ----------------------------------------------------
        # Big orange
        # ----------------------------------------------------

        for c in msg.big_orange_cones:

            big.append(
                (
                    float(c.point.x),
                    float(c.point.y)
                )
            )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        with self.data_lock:

            self.blue_global = blue
            self.yellow_global = yellow
            self.orange_global = orange
            self.big_global = big

    # ========================================================
    # WORLD -> LOCAL
    # ========================================================

    def _world_to_local(
        self,
        car_x,
        car_y,
        car_yaw,
        px,
        py
    ):

        """
        World/map frame -> car frame.

        Car frame:

            +X = forward
            +Y = left
        """

        c = math.cos(
            car_yaw
        )

        s = math.sin(
            car_yaw
        )

        dx = px - car_x
        dy = py - car_y

        lx = (
            c * dx
            +
            s * dy
        )

        ly = (
            -s * dx
            +
            c * dy
        )

        return (
            lx,
            ly
        )

    # ========================================================
    # COMPUTE FOV
    # ========================================================

    def _compute_fov_points(
        self,
        car_x,
        car_y,
        car_yaw
    ):

        """
        Select cones inside the FOV sector.
        """

        vertex_offset = (
            self.vertex_offset
        )

        R = self.fov_radius

        half_angle = (
            self.fov_half_rad
        )

        points_window = []
        classes_window = []

        # ----------------------------------------------------
        # Add list
        # ----------------------------------------------------

        def add_list(
            cones,
            cls_name
        ):

            for px, py in cones:

                lx, ly = (
                    self._world_to_local(
                        car_x,
                        car_y,
                        car_yaw,
                        px,
                        py
                    )
                )

                vx = (
                    lx -
                    vertex_offset
                )

                vy = ly

                r = math.hypot(
                    vx,
                    vy
                )

                if (
                    r > R
                    or
                    r < 1e-3
                ):

                    continue

                theta = math.atan2(
                    vy,
                    vx
                )

                if (
                    abs(theta)
                    <=
                    half_angle
                ):

                    points_window.append(
                        (
                            px,
                            py
                        )
                    )

                    classes_window.append(
                        cls_name
                    )

        # ----------------------------------------------------
        # All cone types
        # ----------------------------------------------------

        add_list(
            self.blue_global,
            "blue"
        )

        add_list(
            self.yellow_global,
            "yellow"
        )

        add_list(
            self.orange_global,
            "orange"
        )

        add_list(
            self.big_global,
            "big"
        )

        return (
            points_window,
            classes_window
        )

    # ========================================================
    # VIRTUAL COLOURLESS CONE
    # ========================================================

    def _get_virtual_cone(
        self,
        car_x,
        car_y,
        car_yaw
    ):

        """
        Create a virtual cone in front of the car.

        Returned in GLOBAL coordinates.
        """

        d = (
            self.virtual_cone_distance
        )

        virtual_x = (
            car_x
            +
            d * math.cos(
                car_yaw
            )
        )

        virtual_y = (
            car_y
            +
            d * math.sin(
                car_yaw
            )
        )

        return (
            virtual_x,
            virtual_y
        )

    # ========================================================
    # FALLBACK DELAUNAY
    # ========================================================

    def _build_fallback_edges(
        self,
        points_window,
        classes_window,
        car_x,
        car_y,
        car_yaw
    ):

        """
        Fallback mode:

            selected colour cones
                    +
            virtual colourless cone
                    ↓
                Delaunay
                    ↓
                  edges
        """

        # ----------------------------------------------------
        # Separate blue and yellow
        # ----------------------------------------------------

        blue_points = [
            points_window[i]
            for i, cls
            in enumerate(
                classes_window
            )
            if cls == "blue"
        ]

        yellow_points = [
            points_window[i]
            for i, cls
            in enumerate(
                classes_window
            )
            if cls == "yellow"
        ]

        blue_count = len(
            blue_points
        )

        yellow_count = len(
            yellow_points
        )

        # ----------------------------------------------------
        # Select side with MORE cones
        # ----------------------------------------------------

        if blue_count > yellow_count:

            selected_points = (
                list(
                    blue_points
                )
            )

            selected_colour = "blue"

        elif yellow_count > blue_count:

            selected_points = (
                list(
                    yellow_points
                )
            )

            selected_colour = "yellow"

        else:

            # Equal and neither side has enough
            if (
                blue_count < 2
                and
                yellow_count < 2
            ):

                return (
                    [],
                    [],
                    [],
                    None
                )

            return (
                [],
                [],
                [],
                None
            )

        # ----------------------------------------------------
        # Need >= 2 cones
        # ----------------------------------------------------

        if len(
            selected_points
        ) < 2:

            return (
                [],
                [],
                [],
                None
            )

        # ----------------------------------------------------
        # Virtual cone
        # ----------------------------------------------------

        virtual_cone = (
            self._get_virtual_cone(
                car_x,
                car_y,
                car_yaw
            )
        )

        # ----------------------------------------------------
        # Build working set
        # ----------------------------------------------------

        fallback_points = (
            list(
                selected_points
            )
        )

        fallback_classes = [
            selected_colour
            for _ in selected_points
        ]

        fallback_points.append(
            virtual_cone
        )

        fallback_classes.append(
            "colorless"
        )

        # ----------------------------------------------------
        # Delaunay
        # ----------------------------------------------------

        N = len(
            fallback_points
        )

        if N < 2:

            return (
                [],
                fallback_points,
                fallback_classes,
                selected_colour
            )

        pts = np.asarray(
            fallback_points,
            dtype=float
        )

        edges_set = set()

        # ----------------------------------------------------
        # Delaunay
        # ----------------------------------------------------

        if (
            _HAS_SCIPY
            and
            N >= 3
        ):

            try:

                tri = Delaunay(
                    pts
                )

                for simplex in (
                    tri.simplices
                ):

                    i = int(
                        simplex[0]
                    )

                    j = int(
                        simplex[1]
                    )

                    k = int(
                        simplex[2]
                    )

                    for a, b in (
                        (i, j),
                        (j, k),
                        (k, i)
                    ):

                        if a > b:

                            a, b = (
                                b,
                                a
                            )

                        edges_set.add(
                            (a, b)
                        )

            except Exception as e:

                self.get_logger().warn(
                    f"Fallback Delaunay failed: {e}"
                )

                # --------------------------------------------
                # k-NN fallback
                # --------------------------------------------

                k = min(
                    3,
                    N - 1
                )

                for i in range(N):

                    d2 = np.sum(
                        (
                            pts
                            -
                            pts[i]
                        ) ** 2,
                        axis=1
                    )

                    d2[i] = (
                        float("inf")
                    )

                    nn_idx = (
                        np.argsort(
                            d2
                        )[:k]
                    )

                    for j in nn_idx:

                        a = i
                        b = int(j)

                        if a > b:

                            a, b = (
                                b,
                                a
                            )

                        edges_set.add(
                            (a, b)
                        )

        else:

            # ------------------------------------------------
            # k-NN
            # ------------------------------------------------

            k = min(
                3,
                N - 1
            )

            for i in range(N):

                d2 = np.sum(
                    (
                        pts
                        -
                        pts[i]
                    ) ** 2,
                    axis=1
                )

                d2[i] = (
                    float("inf")
                )

                nn_idx = (
                    np.argsort(
                        d2
                    )[:k]
                )

                for j in nn_idx:

                    a = i
                    b = int(j)

                    if a > b:

                        a, b = (
                            b,
                            a
                        )

                    edges_set.add(
                        (a, b)
                    )

        # ----------------------------------------------------
        # Edge length filtering
        # ----------------------------------------------------

        edges = []

        max_len2 = (
            6.0 *
            6.0
        )

        for i, j in edges_set:

            x1, y1 = (
                fallback_points[i]
            )

            x2, y2 = (
                fallback_points[j]
            )

            d2 = (
                (x1 - x2) ** 2
                +
                (y1 - y2) ** 2
            )

            if (
                d2 >=
                max_len2
            ):

                continue

            edges.append(
                (i, j)
            )

        return (
            edges,
            fallback_points,
            fallback_classes,
            selected_colour
        )

    # ========================================================
    # NORMAL DELAUNAY
    # ========================================================

    def _build_edges(
        self,
        points_window,
        classes_window
    ):

        """
        Original Delaunay algorithm.
        """

        N = len(
            points_window
        )

        if N < 2:

            return []

        pts = np.asarray(
            points_window,
            dtype=float
        )

        edges_set = set()

        # ----------------------------------------------------
        # Delaunay
        # ----------------------------------------------------

        if (
            _HAS_SCIPY
            and
            N >= 3
        ):

            try:
                simplices = Delaunay(pts).simplices
            except QhullError:
                simplices = ()

            for simplex in simplices:

                i = int(
                    simplex[0]
                )

                j = int(
                    simplex[1]
                )

                k = int(
                    simplex[2]
                )

                for a, b in (
                    (i, j),
                    (j, k),
                    (k, i)
                ):

                    if a > b:

                        a, b = (
                            b,
                            a
                        )

                    edges_set.add(
                        (a, b)
                    )

        if not edges_set:

            # ------------------------------------------------
            # k-NN
            # ------------------------------------------------

            k = min(
                3,
                N - 1
            )

            for i in range(N):

                d2 = np.sum(
                    (
                        pts
                        -
                        pts[i]
                    ) ** 2,
                    axis=1
                )

                d2[i] = (
                    float("inf")
                )

                nn_idx = (
                    np.argsort(
                        d2
                    )[:k]
                )

                for j in nn_idx:

                    a = i
                    b = int(j)

                    if a > b:

                        a, b = (
                            b,
                            a
                        )

                    edges_set.add(
                        (a, b)
                    )

        # ----------------------------------------------------
        # Original constraints
        # ----------------------------------------------------

        orange_like = {
            "orange",
            "big"
        }

        blue_yellow = {
            "blue",
            "yellow"
        }

        edges = []

        max_len2 = (
            6.0 *
            6.0
        )

        for i, j in edges_set:

            ci = (
                classes_window[i]
            )

            cj = (
                classes_window[j]
            )

            x1, y1 = (
                points_window[i]
            )

            x2, y2 = (
                points_window[j]
            )

            d2 = (
                (x1 - x2) ** 2
                +
                (y1 - y2) ** 2
            )

            # ------------------------------------------------
            # Maximum edge length
            # ------------------------------------------------

            if (
                d2 >=
                max_len2
            ):

                continue

            # ------------------------------------------------
            # Blue-blue
            # ------------------------------------------------

            if (
                ci == "blue"
                and
                cj == "blue"
            ):

                continue

            # ------------------------------------------------
            # Yellow-yellow
            # ------------------------------------------------

            if (
                ci == "yellow"
                and
                cj == "yellow"
            ):

                continue

            # ------------------------------------------------
            # Orange <-> blue/yellow
            # ------------------------------------------------

            if (
                (
                    ci in orange_like
                    and
                    cj in blue_yellow
                )
                or
                (
                    cj in orange_like
                    and
                    ci in blue_yellow
                )
            ):

                continue

            edges.append(
                (i, j)
            )

        return edges

    # ========================================================
    # BIG ORANGE SPECIAL
    # ========================================================

    def _big_cone_candidate(
        self,
        points_window,
        classes_window
    ):

        """
        Big-orange logic:

            2 big cones:
                midpoint

            >=3 big cones:
                centroid
        """

        big_pts = [
            points_window[i]
            for i, cls
            in enumerate(
                classes_window
            )
            if cls == "big"
        ]

        n = len(
            big_pts
        )

        if n < 2:

            return []

        # ----------------------------------------------------
        # Exactly 2
        # ----------------------------------------------------

        if n == 2:

            (
                x1,
                y1
            ), (
                x2,
                y2
            ) = big_pts

            mx = (
                0.5 *
                (
                    x1 + x2
                )
            )

            my = (
                0.5 *
                (
                    y1 + y2
                )
            )

            return [
                (
                    mx,
                    my,
                    True
                )
            ]

        # ----------------------------------------------------
        # >= 3
        # ----------------------------------------------------

        xs = [
            p[0]
            for p in big_pts
        ]

        ys = [
            p[1]
            for p in big_pts
        ]

        cx = (
            sum(xs) / n
        )

        cy = (
            sum(ys) / n
        )

        return [
            (
                cx,
                cy,
                True
            )
        ]

    # ========================================================
    # FILTER CANDIDATES
    # ========================================================

    def _filter_centroids_min_spacing(
        self,
        centroids_raw,
        car_x,
        car_y,
        min_dist
    ):

        """
        Candidate filtering:

        1. Farthest-first.
        2. Reject candidates too close to car.
        3. Reject candidates too close to an already
           accepted candidate.
        """

        if not centroids_raw:

            return []

        pts = np.array(
            [
                [
                    c[0],
                    c[1]
                ]
                for c in centroids_raw
            ],
            dtype=float
        )

        car = np.array(
            [
                car_x,
                car_y
            ],
            dtype=float
        )

        d2_car = np.sum(
            (
                pts - car
            ) ** 2,
            axis=1
        )

        order = np.argsort(
            -d2_car
        )

        kept_pts = []
        kept = []

        for idx in order:

            p = pts[idx]

            # ------------------------------------------------
            # Car distance
            # ------------------------------------------------

            if (
                np.linalg.norm(
                    p - car
                )
                <
                self.min_car_dist
            ):

                continue

            # ------------------------------------------------
            # First candidate
            # ------------------------------------------------

            if not kept_pts:

                kept_pts.append(
                    p
                )

                kept.append(
                    (
                        float(
                            p[0]
                        ),
                        float(
                            p[1]
                        ),
                        bool(
                            centroids_raw[
                                idx
                            ][2]
                        )
                    )
                )

                continue

            # ------------------------------------------------
            # Spacing
            # ------------------------------------------------

            too_close = False

            for kp in kept_pts:

                if (
                    np.linalg.norm(
                        p - kp
                    )
                    <
                    min_dist
                ):

                    too_close = True
                    break

            if too_close:

                continue

            kept_pts.append(
                p
            )

            kept.append(
                (
                    float(
                        p[0]
                    ),
                    float(
                        p[1]
                    ),
                    bool(
                        centroids_raw[
                            idx
                        ][2]
                    )
                )
            )

        return kept

    # ========================================================
    # GREEDY PATH
    # ========================================================

    def _build_greedy_path(
        self,
        car_x,
        car_y,
        candidate_points
    ):

        """
        Greedy nearest-neighbour path.
        """

        if not candidate_points:

            return []

        pts = np.asarray(
            candidate_points,
            dtype=float
        )

        N = (
            pts.shape[0]
        )

        used = np.zeros(
            N,
            dtype=bool
        )

        current = np.array(
            [
                car_x,
                car_y
            ],
            dtype=float
        )

        max_step2 = (
            self.max_hop
            *
            self.max_hop
        )

        order = []

        for _ in range(N):

            diff = (
                pts -
                current
            )

            d2 = np.sum(
                diff * diff,
                axis=1
            )

            # Already used
            d2[
                used
            ] = float(
                "inf"
            )

            # Too far
            d2[
                d2 > max_step2
            ] = float(
                "inf"
            )

            idx = int(
                np.argmin(
                    d2
                )
            )

            if (
                not math.isfinite(
                    float(
                        d2[idx]
                    )
                )
                or
                float(
                    d2[idx]
                )
                ==
                float("inf")
            ):

                break

            used[idx] = True

            order.append(
                idx
            )

            current = pts[idx]

        return [
            tuple(
                pts[i]
            )
            for i in order
        ]

    # ========================================================
    # PUBLISH PATH
    # ========================================================

    def _publish_path(
        self,
        car_x,
        car_y,
        path_points
    ):

        msg = Path()

        msg.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        msg.header.frame_id = (
            "map"
        )

        poses = []

        # ----------------------------------------------------
        # Car pose
        # ----------------------------------------------------

        car_ps = PoseStamped()

        car_ps.header = (
            msg.header
        )

        car_ps.pose.position.x = (
            float(car_x)
        )

        car_ps.pose.position.y = (
            float(car_y)
        )

        car_ps.pose.position.z = 0.0

        car_ps.pose.orientation.w = 1.0

        poses.append(
            car_ps
        )

        # ----------------------------------------------------
        # Path points
        # ----------------------------------------------------

        for x, y in path_points:

            ps = PoseStamped()

            ps.header = (
                msg.header
            )

            ps.pose.position.x = (
                float(x)
            )

            ps.pose.position.y = (
                float(y)
            )

            ps.pose.position.z = 0.0

            ps.pose.orientation.w = 1.0

            poses.append(
                ps
            )

        msg.poses = poses

        self.path_pub.publish(
            msg
        )

    # ========================================================
    # UPDATE LIVE VISUALIZATION
    # ========================================================

    def _update_live_plot(
        self,
        car_x,
        car_y,
        car_yaw
    ):

        """
        Update the live Matplotlib visualization.

        This function ONLY visualizes already-computed
        algorithm data.
        """

        if self.fig is None or not plt.fignum_exists(
            self.fig.number
        ):

            return

        with self.plot_lock:

            points_window = list(
                self.plot_points_window
            )

            classes_window = list(
                self.plot_classes_window
            )

            edges = list(
                self.plot_edges
            )

            edge_points = list(
                self.plot_edge_points
            )

            candidates = list(
                self.plot_candidates
            )

            path = list(
                self.plot_path
            )

            virtual_cone = (
                self.plot_virtual_cone
            )

            mode = (
                self.plot_mode
            )

            selected_side = (
                self.plot_selected_side
            )

            blue_count = (
                self.plot_blue_count
            )

            yellow_count = (
                self.plot_yellow_count
            )

        # ====================================================
        # UPDATE CONE SCATTERS
        # ====================================================

        def get_points(
            cls_name
        ):

            return [
                p
                for p, c
                in zip(
                    points_window,
                    classes_window
                )
                if c == cls_name
            ]

        blue_pts = get_points(
            "blue"
        )

        yellow_pts = get_points(
            "yellow"
        )

        orange_pts = get_points(
            "orange"
        )

        big_pts = get_points(
            "big"
        )

        # ----------------------------------------------------
        # Helper for scatter
        # ----------------------------------------------------

        def set_scatter(
            scatter,
            points
        ):

            if points:

                scatter.set_offsets(
                    np.asarray(
                        points
                    )
                )

            else:

                scatter.set_offsets(
                    np.empty(
                        (0, 2)
                    )
                )

        set_scatter(
            self.scatter_blue,
            blue_pts
        )

        set_scatter(
            self.scatter_yellow,
            yellow_pts
        )

        set_scatter(
            self.scatter_orange,
            orange_pts
        )

        set_scatter(
            self.scatter_big,
            big_pts
        )

        # ====================================================
        # VIRTUAL CONE
        # ====================================================

        if virtual_cone is not None:

            self.scatter_virtual.set_offsets(
                np.asarray(
                    [virtual_cone]
                )
            )

        else:

            self.scatter_virtual.set_offsets(
                np.empty(
                    (0, 2)
                )
            )

        # ====================================================
        # CANDIDATES
        # ====================================================

        set_scatter(
            self.scatter_candidates,
            candidates
        )

        # ====================================================
        # CAR
        # ====================================================

        self.scatter_car.set_offsets(
            np.asarray(
                [
                    [
                        car_x,
                        car_y
                    ]
                ]
            )
        )

        # ====================================================
        # FINAL PATH
        # ====================================================

        if path:

            path_arr = np.asarray(
                path
            )

            self.path_line.set_data(
                path_arr[:, 0],
                path_arr[:, 1]
            )

        else:

            self.path_line.set_data(
                [],
                []
            )

        # ====================================================
        # CAR HEADING
        # ====================================================

        heading_length = 2.0

        hx = (
            car_x
            +
            heading_length *
            math.cos(
                car_yaw
            )
        )

        hy = (
            car_y
            +
            heading_length *
            math.sin(
                car_yaw
            )
        )

        self.heading_line.set_data(
            [
                car_x,
                hx
            ],
            [
                car_y,
                hy
            ]
        )

        # ====================================================
        # REMOVE OLD DELAUNAY LINES
        # ====================================================

        for line in (
            self.edge_lines
        ):

            try:

                line.remove()

            except Exception:

                pass

        self.edge_lines = []

        # ====================================================
        # DRAW CURRENT DELAUNAY EDGES
        # ====================================================

        for i, j in edges:

            try:

                x1, y1 = (
                    edge_points[i]
                )

                x2, y2 = (
                    edge_points[j]
                )

            except (
                IndexError,
                ValueError
            ):

                continue

            line, = self.ax.plot(
                [
                    x1,
                    x2
                ],
                [
                    y1,
                    y2
                ],
                linewidth=1.0,
                alpha=0.55
            )

            self.edge_lines.append(
                line
            )

        # ====================================================
        # REMOVE OLD FOV
        # ====================================================

        if self.fov_patch is not None:

            try:

                self.fov_patch.remove()

            except Exception:

                pass

            self.fov_patch = None

        # ====================================================
        # DRAW FOV SECTOR
        # ====================================================

        # FOV vertex is behind the car by vertex_offset.
        #
        # In global frame:
        #
        # vertex = car - vertex_offset * heading
        #
        # Since vertex_offset is normally -5:
        # vertex is 5 m behind the car.

        vertex_x = (
            car_x
            +
            self.vertex_offset *
            math.cos(
                car_yaw
            )
        )

        vertex_y = (
            car_y
            +
            self.vertex_offset *
            math.sin(
                car_yaw
            )
        )

        start_angle = (
            math.degrees(
                car_yaw
            )
            -
            self.fov_angle_deg / 2.0
        )

        self.fov_patch = Wedge(
            (
                vertex_x,
                vertex_y
            ),
            self.fov_radius,
            start_angle,
            start_angle
            +
            self.fov_angle_deg,
            fill=False,
            linewidth=1.2,
            alpha=0.35
        )

        self.ax.add_patch(
            self.fov_patch
        )

        # ====================================================
        # STATUS
        # ====================================================

        if mode == "FALLBACK":

            status = (
                f"MODE: FALLBACK\n"
                f"Blue in FOV: {blue_count}\n"
                f"Yellow in FOV: {yellow_count}\n"
                f"Selected side: "
                f"{selected_side}\n"
                f"Virtual cone: "
                f"{self.virtual_cone_distance:.1f} m"
            )

        elif mode == "NORMAL":

            status = (
                f"MODE: NORMAL\n"
                f"Blue in FOV: {blue_count}\n"
                f"Yellow in FOV: {yellow_count}\n"
                f"Delaunay: all FOV cones"
            )

        else:

            status = (
                f"MODE: {mode}\n"
                f"Blue in FOV: {blue_count}\n"
                f"Yellow in FOV: {yellow_count}"
            )

        self.status_text.set_text(
            status
        )

        # ====================================================
        # AUTO VIEW
        # ====================================================

        # Keep car visible with a reasonable local window.
        view_radius = max(
            15.0,
            min(
                self.fov_radius + 5.0,
                35.0
            )
        )

        self.ax.set_xlim(
            car_x - view_radius,
            car_x + view_radius
        )

        self.ax.set_ylim(
            car_y - view_radius,
            car_y + view_radius
        )

        # ====================================================
        # TITLE
        # ====================================================

        self.ax.set_title(
            "SLAM Skidpad Path Planner — Live"
        )

        # ====================================================
        # REDRAW
        # ====================================================

        self.fig.canvas.draw_idle()

        self.fig.canvas.flush_events()

        plt.pause(
            0.001
        )

    # ========================================================
    # MAIN TICK
    # ========================================================

    def _tick(self):

        # ====================================================
        # GET CURRENT STATE
        # ====================================================

        with self.data_lock:

            car_x = self.car_x
            car_y = self.car_y
            car_yaw = self.car_yaw

            blue_global = list(
                self.blue_global
            )

            yellow_global = list(
                self.yellow_global
            )

            orange_global = list(
                self.orange_global
            )

            big_global = list(
                self.big_global
            )

        # ----------------------------------------------------
        # No odometry yet
        # ----------------------------------------------------

        if (
            car_x is None
            or
            car_y is None
            or
            car_yaw is None
        ):

            return

        # ====================================================
        # KEEP LATEST DATA
        # ====================================================

        with self.data_lock:

            self.blue_global = (
                blue_global
            )

            self.yellow_global = (
                yellow_global
            )

            self.orange_global = (
                orange_global
            )

            self.big_global = (
                big_global
            )

        # ====================================================
        # STEP 1:
        # COMPUTE FOV
        # ====================================================

        (
            points_window,
            classes_window
        ) = self._compute_fov_points(
            car_x,
            car_y,
            car_yaw
        )

        # ====================================================
        # COUNTS
        # ====================================================

        blue_count = sum(
            1
            for cls
            in classes_window
            if cls == "blue"
        )

        yellow_count = sum(
            1
            for cls
            in classes_window
            if cls == "yellow"
        )

        # ====================================================
        # VIRTUAL CONE
        #
        # Only visualized/used in fallback.
        # ====================================================

        virtual_cone = None

        # ====================================================
        # TOO FEW TOTAL POINTS
        # ====================================================

        if len(
            points_window
        ) < 2:

            with self.plot_lock:

                self.plot_points_window = (
                    points_window
                )

                self.plot_classes_window = (
                    classes_window
                )

                self.plot_edges = []
                self.plot_edge_points = []
                self.plot_candidates = []
                self.plot_path = []

                self.plot_virtual_cone = None

                self.plot_mode = (
                    "INSUFFICIENT FOV POINTS"
                )

                self.plot_selected_side = (
                    "None"
                )

                self.plot_blue_count = (
                    blue_count
                )

                self.plot_yellow_count = (
                    yellow_count
                )

            self._publish_path(
                car_x,
                car_y,
                []
            )

            self._update_live_plot(
                car_x,
                car_y,
                car_yaw
            )

            return

        # ====================================================
        # STEP 2:
        # SELECT MODE
        # ====================================================

        fallback_mode = (
            blue_count <= 1
            or
            yellow_count <= 1
        )

        # ====================================================
        # FALLBACK MODE
        # ====================================================

        if fallback_mode:

            (
                edges,
                working_points,
                working_classes,
                selected_side
            ) = (
                self._build_fallback_edges(
                    points_window,
                    classes_window,
                    car_x,
                    car_y,
                    car_yaw
                )
            )

            # ------------------------------------------------
            # Virtual cone for visualization
            # ------------------------------------------------

            if selected_side is not None:

                virtual_cone = (
                    self._get_virtual_cone(
                        car_x,
                        car_y,
                        car_yaw
                    )
                )

            # ------------------------------------------------
            # Edge midpoint candidates
            # ------------------------------------------------

            centroids_raw = []

            for i, j in edges:

                x1, y1 = (
                    working_points[i]
                )

                x2, y2 = (
                    working_points[j]
                )

                mx = (
                    0.5 *
                    (
                        x1 + x2
                    )
                )

                my = (
                    0.5 *
                    (
                        y1 + y2
                    )
                )

                centroids_raw.append(
                    (
                        mx,
                        my,
                        False
                    )
                )

            # ------------------------------------------------
            # Filtering
            # ------------------------------------------------

            centroids = (
                self._filter_centroids_min_spacing(
                    centroids_raw,
                    car_x,
                    car_y,
                    min_dist=(
                        self.min_spacing
                    )
                )
            )

            candidate_points = [
                (
                    x,
                    y
                )
                for x, y, _
                in centroids
            ]

            # ------------------------------------------------
            # Greedy path
            # ------------------------------------------------

            path_points = (
                self._build_greedy_path(
                    car_x,
                    car_y,
                    candidate_points
                )
            )

            # ------------------------------------------------
            # Publish
            # ------------------------------------------------

            self._publish_path(
                car_x,
                car_y,
                path_points
            )

            # ------------------------------------------------
            # Visualization state
            # ------------------------------------------------

            with self.plot_lock:

                self.plot_points_window = (
                    points_window
                )

                self.plot_classes_window = (
                    classes_window
                )

                self.plot_edges = (
                    edges
                )

                self.plot_edge_points = (
                    working_points
                )

                self.plot_candidates = (
                    candidate_points
                )

                self.plot_path = (
                    path_points
                )

                self.plot_virtual_cone = (
                    virtual_cone
                )

                self.plot_mode = (
                    "FALLBACK"
                )

                self.plot_selected_side = (
                    selected_side
                    if selected_side
                    is not None
                    else "None"
                )

                self.plot_blue_count = (
                    blue_count
                )

                self.plot_yellow_count = (
                    yellow_count
                )

        # ====================================================
        # NORMAL MODE
        # ====================================================

        else:

            # ------------------------------------------------
            # Normal Delaunay
            # ------------------------------------------------

            edges = (
                self._build_edges(
                    points_window,
                    classes_window
                )
            )

            # ------------------------------------------------
            # Big-orange special
            # ------------------------------------------------

            big_special = (
                self._big_cone_candidate(
                    points_window,
                    classes_window
                )
            )

            # ------------------------------------------------
            # Midpoint candidates
            # ------------------------------------------------

            centroids_raw = []

            for i, j in edges:

                x1, y1 = (
                    points_window[i]
                )

                x2, y2 = (
                    points_window[j]
                )

                ci = (
                    classes_window[i]
                )

                cj = (
                    classes_window[j]
                )

                # Big-big handled separately
                if (
                    ci == "big"
                    and
                    cj == "big"
                ):

                    continue

                mx = (
                    0.5 *
                    (
                        x1 + x2
                    )
                )

                my = (
                    0.5 *
                    (
                        y1 + y2
                    )
                )

                centroids_raw.append(
                    (
                        mx,
                        my,
                        False
                    )
                )

            # ------------------------------------------------
            # Add big-orange candidate
            # ------------------------------------------------

            for bx, by, _ in (
                big_special
            ):

                centroids_raw.append(
                    (
                        bx,
                        by,
                        True
                    )
                )

            # ------------------------------------------------
            # Candidate filtering
            # ------------------------------------------------

            centroids = (
                self._filter_centroids_min_spacing(
                    centroids_raw,
                    car_x,
                    car_y,
                    min_dist=(
                        self.min_spacing
                    )
                )
            )

            candidate_points = [
                (
                    x,
                    y
                )
                for x, y, _
                in centroids
            ]

            # ------------------------------------------------
            # Greedy path
            # ------------------------------------------------

            path_points = (
                self._build_greedy_path(
                    car_x,
                    car_y,
                    candidate_points
                )
            )

            # ------------------------------------------------
            # Publish
            # ------------------------------------------------

            self._publish_path(
                car_x,
                car_y,
                path_points
            )

            # ------------------------------------------------
            # Visualization state
            # ------------------------------------------------

            with self.plot_lock:

                self.plot_points_window = (
                    points_window
                )

                self.plot_classes_window = (
                    classes_window
                )

                self.plot_edges = (
                    edges
                )

                self.plot_edge_points = (
                    points_window
                )

                self.plot_candidates = (
                    candidate_points
                )

                self.plot_path = (
                    path_points
                )

                self.plot_virtual_cone = (
                    None
                )

                self.plot_mode = (
                    "NORMAL"
                )

                self.plot_selected_side = (
                    "None"
                )

                self.plot_blue_count = (
                    blue_count
                )

                self.plot_yellow_count = (
                    yellow_count
                )

        # ====================================================
        # UPDATE VISUALIZATION
        # ====================================================

        self._update_live_plot(
            car_x,
            car_y,
            car_yaw
        )


# ============================================================
# MAIN
# ============================================================

def main():

    rclpy.init()

    node = (
        SlamPathSectorVisualizer()
    )

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    finally:

        if node.fig is not None:
            plt.close(node.fig)

        node.destroy_node()

        rclpy.shutdown()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()