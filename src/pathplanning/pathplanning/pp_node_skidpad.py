#!/usr/bin/env python3
# slam_path_csv_visualiser.py
#
# Visualiser that:
#   - Subscribes to /slam/odom (nav_msgs/Odometry)
#   - Subscribes to /slam/map_cones (eufs_msgs/ConeArrayWithCovariance)
#   - Loads a CSV reference path (x,y) in map frame
#   - Tracks a *sequential* index along the CSV (forward-only, like PathPublisher)
#   - Builds FOV sector around the car and Delaunay/k-NN edges between cones
#   - Computes centroids of valid edges (black X’s)
#   - For CSV points in a limited window [cur_idx, cur_idx+N_window],
#       finds nearest centroid within 1 m and marks those centroids as red dots
#   - Draws a red polyline from the car through the matched centroids
#   - Everything else stays as in the original sector visualiser.

import math
import threading
from typing import List, Tuple

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

from nav_msgs.msg import Odometry
from eufs_msgs.msg import ConeArrayWithCovariance

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
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


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class SlamPathCsvVisualizer(Node):
    def __init__(self):
        super().__init__("slam_path_csv_visualiser")

        # ---- Parameters ----
        # Topics
        self.declare_parameter("topics.odom_in", "/slam/odom")
        self.declare_parameter("topics.map_in", "/slam/map_cones")

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
        self.declare_parameter("ref.search_window", 200)   # how far ahead in CSV to look
        self.declare_parameter("match.radius_m", 1.0)      # max distance cone↔CSV

        # Centroid filters
        self.declare_parameter("centroid.min_car_dist_m", 2.0)   # reject < 2 m from car
        self.declare_parameter("centroid.min_spacing_m", 1.0)    # spacing between centroids

        gp = self.get_parameter
        odom_topic = str(gp("topics.odom_in").value)
        map_topic = str(gp("topics.map_in").value)

        best_effort = bool(gp("qos.best_effort").value)
        depth = int(gp("qos.depth").value)

        self.vertex_offset = float(gp("fov.vertex_offset_m").value)
        self.fov_radius = float(gp("fov.radius_m").value)
        self.fov_angle_deg = float(gp("fov.angle_deg").value)
        self.fov_half_rad = math.radians(self.fov_angle_deg * 0.5)

        self.ref_loop = bool(gp("ref.loop").value)
        self.ref_search_window = int(gp("ref.search_window").value)
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
                f"[csv_vis] Loaded reference path {path_param.as_posix()} "
                f"with {len(self.global_rx)} points"
            )
        except Exception as e:
            self.get_logger().error(f"[csv_vis] Failed to load path {path_param}: {e}")
            self.global_rx = np.array([], dtype=float)
            self.global_ry = np.array([], dtype=float)

        # Reference path tracking state (forward-only)
        self.cur_idx = 0
        self.path_initialized = False

        # ---- Subscriptions ----
        self.create_subscription(Odometry, odom_topic, self.cb_odom, qos)
        self.create_subscription(ConeArrayWithCovariance, map_topic, self.cb_map, qos)

        # ---- State ----
        self.car_x = None
        self.car_y = None
        self.car_yaw = None  # radians

        # Global map cones in map frame
        self.blue_global: List[Tuple[float, float]] = []
        self.yellow_global: List[Tuple[float, float]] = []
        self.orange_global: List[Tuple[float, float]] = []
        self.big_global: List[Tuple[float, float]] = []

        self.data_lock = threading.Lock()

        # ---- Matplotlib figure ----
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.grid(True)
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_title("CSV-anchored path visualiser (sector FOV + cone centroids)")

        if not _HAS_SCIPY:
            self.get_logger().warn(
                "[csv_vis] SciPy not found; using k-NN graph instead of true Delaunay."
            )

        self.get_logger().info(
            f"[csv_vis] odom={odom_topic}, map={map_topic}, "
            f"vertex_offset={self.vertex_offset}m, radius={self.fov_radius}m, "
            f"angle={self.fov_angle_deg}°"
        )

    # --------------------- Core helpers ---------------------

    def _has_reference(self) -> bool:
        return self.global_rx.size > 0

    def _dist_sq_to_ref(self, idx: int, cx: float, cy: float) -> float:
        dx = cx - self.global_rx[idx]
        dy = cy - self.global_ry[idx]
        return dx * dx + dy * dy

    def _update_reference_progress(self, cx: float, cy: float):
        """
        Forward-only index tracking along CSV:
          - First call: global search for nearest CSV point.
          - Later: only walk forward while next point is closer.
        """
        if not self._has_reference():
            return

        n = len(self.global_rx)
        if n == 0:
            return

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

        search_window = min(self.ref_search_window, n - 1)
        for _ in range(search_window):
            cur_d2 = self._dist_sq_to_ref(self.cur_idx, cx, cy)
            next_idx = self.cur_idx + 1
            if self.ref_loop:
                next_idx %= n
            elif next_idx >= n:
                break

            next_d2 = self._dist_sq_to_ref(next_idx, cx, cy)
            if next_d2 < cur_d2:
                self.cur_idx = next_idx
            else:
                break

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

    # --------------------- FOV / graph helpers ---------------------

    def _world_to_local(self, car_x, car_y, car_yaw, px, py):
        """
        Transform world -> car frame (x forward, y left).
        """
        c = math.cos(car_yaw)
        s = math.sin(car_yaw)
        dx = px - car_x
        dy = py - car_y
        # rotate by -yaw
        lx =  c * dx + s * dy
        ly = -s * dx + c * dy
        return lx, ly

    def _compute_fov_points(self, car_x, car_y, car_yaw):
        """
        Select cones within the circular sector FOV, and *in front* of the
        perpendicular line through the car (local x >= 0).
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

                # Cut by perpendicular line at car: ignore behind-car points
                if lx < 0.0:
                    continue

                # Position relative to vertex in local frame
                vx = lx - vertex_offset
                vy = ly
                r = math.hypot(vx, vy)
                if r > R or r < 1e-3:
                    continue
                theta = math.atan2(vy, vx)  # angle w.r.t +x from vertex
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
        Build graph edges on FOV points using Delaunay (if available) or k-NN.

        Constraints:
          - edge length < 6 m
          - Drop blue-blue edges
          - Drop yellow-yellow edges
          - Drop any orange-like ↔ (blue or yellow) edge
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
            # Fallback: simple k-NN (k up to 3)
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
        max_len2 = 6.0 * 6.0  # 6 m

        for (i, j) in edges_set:
            ci = classes_window[i]
            cj = classes_window[j]

            # Length constraint
            x1, y1 = points_window[i]
            x2, y2 = points_window[j]
            d2 = (x1 - x2) ** 2 + (y1 - y2) ** 2
            if d2 >= max_len2:
                continue

            # Drop blue-blue and yellow-yellow
            if (ci == "blue" and cj == "blue") or (ci == "yellow" and cj == "yellow"):
                continue

            # Drop any orange-like ↔ (blue or yellow) edge
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
        centroids_raw: list of (x, y, is_big_big)
        Enforce:
          - no centroid within centroid_min_car_dist of car
          - no two centroids closer than centroid_min_spacing to each other
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

            # reject near car
            if np.linalg.norm(p - car) < self.centroid_min_car_dist:
                continue

            if not kept_pts:
                kept_pts.append(p)
                kept.append(
                    (float(p[0]), float(p[1]), bool(centroids_raw[idx][2]))
                )
                continue

            # spacing vs existing kept
            too_close = False
            for kp in kept_pts:
                if np.linalg.norm(p - kp) < self.centroid_min_spacing:
                    too_close = True
                    break
            if too_close:
                continue

            kept_pts.append(p)
            kept.append(
                (float(p[0]), float(p[1]), bool(centroids_raw[idx][2]))
            )

        return kept

    def _match_centroids_to_csv_window(
        self,
        centroids: List[Tuple[float, float, bool]],
    ) -> List[Tuple[float, float]]:
        """
        For CSV indices in [cur_idx, cur_idx+ref_search_window], in *order*,
        find nearest centroid within match_radius and mark those centroids as used.

        Returns list of matched centroid positions [(mx, my), ...] in CSV order.
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

        # Walk forward along CSV starting at cur_idx
        max_steps = min(self.ref_search_window, n_ref)
        for step in range(max_steps + 1):
            idx = self.cur_idx + step
            if self.ref_loop:
                idx %= n_ref
            elif idx >= n_ref:
                break

            px = self.global_rx[idx]
            py = self.global_ry[idx]

            # nearest unused centroid to this CSV point
            diff = pts - np.array([px, py])
            d2 = np.sum(diff * diff, axis=1)
            d2[used] = float("inf")

            j = int(np.argmin(d2))
            if not math.isfinite(d2[j]) or d2[j] > max_r2:
                continue  # no centroid close enough to this CSV point

            used[j] = True
            matched.append((float(pts[j, 0]), float(pts[j, 1])))

        return matched

    # --------------------- Plot update ---------------------

    def update_plot(self):
        with self.data_lock:
            car_x = self.car_x
            car_y = self.car_y
            car_yaw = self.car_yaw

            blue_global = list(self.blue_global)
            yellow_global = list(self.yellow_global)
            orange_global = list(self.orange_global)
            big_global = list(self.big_global)

        self.ax.clear()
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.grid(True)
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_title("CSV-anchored path visualiser (sector FOV + cone centroids)")

        all_x = []
        all_y = []

        # Plot reference path (dashed grey), if present
        if self._has_reference():
            self.ax.plot(
                self.global_rx,
                self.global_ry,
                linestyle="--",
                color="0.7",
                linewidth=1.0,
                label="CSV reference",
            )
            all_x.extend(self.global_rx.tolist())
            all_y.extend(self.global_ry.tolist())

        # Plot cones
        if blue_global:
            bx, by = zip(*blue_global)
            self.ax.scatter(bx, by, s=20, c="b", marker="o", label="blue cones")
            all_x.extend(bx); all_y.extend(by)

        if yellow_global:
            yx, yy = zip(*yellow_global)
            self.ax.scatter(
                yx, yy, s=20, c="y", marker="o", edgecolors="k", label="yellow cones"
            )
            all_x.extend(yx); all_y.extend(yy)

        if orange_global:
            ox, oy = zip(*orange_global)
            self.ax.scatter(ox, oy, s=20, c="orange", marker="o", label="orange cones")
            all_x.extend(ox); all_y.extend(oy)

        if big_global:
            gx, gy = zip(*big_global)
            self.ax.scatter(gx, gy, s=40, c="magenta", marker="^", label="big orange")
            all_x.extend(gx); all_y.extend(gy)

        # If car pose missing, just scale axes and bail
        if (car_x is None) or (car_y is None) or (car_yaw is None):
            if all_x and all_y:
                xmin, xmax = min(all_x), max(all_x)
                ymin, ymax = min(all_y), max(all_y)
                pad = 3.0
                if xmax - xmin < 1e-3: xmax = xmin + 1.0
                if ymax - ymin < 1e-3: ymax = ymin + 1.0
                self.ax.set_xlim(xmin - pad, xmax + pad)
                self.ax.set_ylim(ymin - pad, ymax + pad)
            return

        # Update CSV index based on car pose (sequence preservation)
        self._update_reference_progress(car_x, car_y)

        # Car triangle
        tri_len = 1.0
        tri_width = 0.5
        pts_local = np.array([
            [tri_len, 0.0],
            [-tri_len * 0.5, -tri_width],
            [-tri_len * 0.5, +tri_width],
        ])
        c = math.cos(car_yaw); s = math.sin(car_yaw)
        R = np.array([[c, -s], [s, c]])
        pts_world = (R @ pts_local.T).T + np.array([car_x, car_y])
        car_tri = Polygon(
            pts_world,
            closed=True,
            facecolor="black",
            edgecolor="white",
            linewidth=1.0,
            label="car",
        )
        self.ax.add_patch(car_tri)
        all_x.append(car_x); all_y.append(car_y)

        # Perpendicular cut line through car
        perp_yaw = car_yaw + math.pi * 0.5
        cp = math.cos(perp_yaw); sp = math.sin(perp_yaw)
        L = 100.0
        x1 = car_x - L * cp; y1 = car_y - L * sp
        x2 = car_x + L * cp; y2 = car_y + L * sp
        self.ax.plot(
            [x1, x2], [y1, y2],
            linestyle="--", color="0.5", linewidth=0.8, label="perp cut"
        )

        # FOV sector (dotted)
        vertex_local = np.array([self.vertex_offset, 0.0])
        vertex_world = (R @ vertex_local) + np.array([car_x, car_y])
        vx, vy = vertex_world[0], vertex_world[1]

        R_arc = self.fov_radius
        arc_points = []
        for k in range(0, 61):
            theta = -self.fov_half_rad + (2.0 * self.fov_half_rad) * (k / 60.0)
            px_local = self.vertex_offset + R_arc * math.cos(theta)
            py_local = 0.0 + R_arc * math.sin(theta)
            p_world = (R @ np.array([px_local, py_local])) + np.array([car_x, car_y])
            arc_points.append(p_world)
        arc_points = np.asarray(arc_points)
        self.ax.plot(arc_points[:, 0], arc_points[:, 1], linestyle=":", color="gray")

        left_theta = -self.fov_half_rad
        right_theta = self.fov_half_rad
        for theta in (left_theta, right_theta):
            end_local = np.array([
                self.vertex_offset + R_arc * math.cos(theta),
                0.0 + R_arc * math.sin(theta),
            ])
            end_world = (R @ end_local) + np.array([car_x, car_y])
            self.ax.plot(
                [vx, end_world[0]],
                [vy, end_world[1]],
                linestyle=":",
                color="gray",
            )

        # Cones in FOV
        points_window, classes_window = self._compute_fov_points(
            car_x, car_y, car_yaw
        )

        # Graph edges
        edges = self._build_edges(points_window, classes_window)

        orange_like = {"orange", "big"}
        centroids_raw: List[Tuple[float, float, bool]] = []

        # Draw edges & accumulate raw centroids
        for (i, j) in edges:
            (x1e, y1e) = points_window[i]
            (x2e, y2e) = points_window[j]
            ci = classes_window[i]
            cj = classes_window[j]

            is_big_big = (ci == "big" and cj == "big")

            color = "orange" if (ci in orange_like and cj in orange_like) else "k"
            self.ax.plot([x1e, x2e], [y1e, y2e], color=color, linewidth=1.0)

            mx = 0.5 * (x1e + x2e)
            my = 0.5 * (y1e + y2e)
            centroids_raw.append((mx, my, is_big_big))

        # Filter centroids -> candidate centroids (black X)
        centroids = self._filter_centroids(centroids_raw, car_x, car_y)

        candidate_points: List[Tuple[float, float]] = []
        for (mx, my, _is_big_big) in centroids:
            self.ax.scatter([mx], [my], marker="x", c="k", s=25)  # black X
            candidate_points.append((mx, my))
            all_x.append(mx); all_y.append(my)

        # Match candidate centroids to CSV window (1 m gate, sequential CSV)
        matched = self._match_centroids_to_csv_window(centroids)

        # Red dots + red polyline from car through matched centroids
        if matched:
            mpx = [car_x]
            mpy = [car_y]
            for (mx, my) in matched:
                # overlay red dot on top of black X
                self.ax.scatter([mx], [my], c="red", s=30, marker="o", zorder=5)
                mpx.append(mx); mpy.append(my)
            self.ax.plot(mpx, mpy, color="red", linewidth=2.0, label="CSV-matched path")
            all_x.extend(mpx); all_y.extend(mpy)

        # Auto-fit axes
        if all_x and all_y:
            xmin, xmax = min(all_x), max(all_x)
            ymin, ymax = min(all_y), max(all_y)
            pad = 3.0
            if xmax - xmin < 1e-3: xmax = xmin + 1.0
            if ymax - ymin < 1e-3: ymax = ymin + 1.0
            self.ax.set_xlim(xmin - pad, xmax + pad)
            self.ax.set_ylim(ymin - pad, ymax + pad)

        # De-duplicate legend
        handles, labels = self.ax.get_legend_handles_labels()
        uniq = {}
        for h, l in zip(handles, labels):
            uniq[l] = h
        if uniq:
            self.ax.legend(
                uniq.values(),
                uniq.keys(),
                loc="upper right",
                fontsize=8,
            )


def main():
    rclpy.init()
    node = SlamPathCsvVisualizer()
    plt.ion()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            node.update_plot()
            plt.pause(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
