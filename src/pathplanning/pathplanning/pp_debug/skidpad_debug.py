#!/usr/bin/env python3
# slam_path_sector_visualiser.py
#
# Visualiser for SLAM cone map using a sector FOV:
#   - Subscribes to /slam/odom (nav_msgs/Odometry)
#   - Subscribes to /slam/map_cones (eufs_msgs/ConeArrayWithCovariance)
#   - FOV is a circular sector:
#       * vertex 5 m behind car along heading
#       * radius 30 m (from vertex) => about -5..+25 m along heading
#       * total angle 60° (±30° around heading)
#   - Inside the FOV:
#       * Build Delaunay (or k-NN) edges on cones
#       * Apply constraints:
#           - edge length < 6 m (as currently configured)
#           - drop blue-blue and yellow-yellow edges
#           - drop any orange-like ↔ (blue or yellow) edges
#           - keep orange-like ↔ orange-like and blue-yellow
#       * For each edge, compute centroid; then:
#           - enforce ≥ 1 m spacing between centroids, keeping the farther
#             one from the car in each cluster
#           - drop any centroid within 0.5 m of the car pose
#           - plot an "X" at each kept centroid
#           - additionally, if edge is big-orange ↔ big-orange, mark that
#             centroid with a green square
#       * Build a greedy NN path from car through all kept centroids,
#         draw as cyan polyline.
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

from nav_msgs.msg import Odometry
from eufs_msgs.msg import ConeArrayWithCovariance

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

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


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class SlamPathSectorVisualizer(Node):
    def __init__(self):
        super().__init__("slam_path_sector_visualiser")

        # ---- Parameters ----
        self.declare_parameter("topics.odom_in", "/slam/odom")
        self.declare_parameter("topics.map_in", "/slam/map_cones")
        self.declare_parameter("qos.best_effort", True)
        self.declare_parameter("qos.depth", 50)

        # Sector FOV parameters
        # Vertex is at -5 m along heading from car
        self.declare_parameter("fov.vertex_offset_m", -5.0)
        # From vertex, radius so that we reach 25 m ahead of car: 30 m
        self.declare_parameter("fov.radius_m", 30.0)
        # Total FOV angle in degrees (around heading)
        self.declare_parameter("fov.angle_deg", 60.0)

        gp = self.get_parameter
        odom_topic = str(gp("topics.odom_in").value)
        map_topic = str(gp("topics.map_in").value)
        best_effort = bool(gp("qos.best_effort").value)
        depth = int(gp("qos.depth").value)

        self.vertex_offset = float(gp("fov.vertex_offset_m").value)
        self.fov_radius = float(gp("fov.radius_m").value)
        self.fov_angle_deg = float(gp("fov.angle_deg").value)
        self.fov_half_rad = math.radians(self.fov_angle_deg * 0.5)

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
        self.ax.set_title("Path-planning sector FOV (Delaunay + greedy NN)")

        if not _HAS_SCIPY:
            self.get_logger().warn(
                "[slam_path_sector_visualiser] SciPy not found; "
                "falling back to k-NN graph instead of true Delaunay."
            )

        self.get_logger().info(
            f"[slam_path_sector_visualiser] odom={odom_topic}, map={map_topic}, "
            f"vertex_offset={self.vertex_offset}m, radius={self.fov_radius}m, "
            f"angle={self.fov_angle_deg}°"
        )

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
        Select cones within the circular sector FOV.

        Sector is defined in car-local coordinates with:
          - vertex at (vertex_offset, 0)  [vertex_offset is negative => behind car]
          - radius = self.fov_radius
          - central direction = +x axis (car heading)
          - FOV half-angle = self.fov_half_rad around +x, measured from vertex
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

    def _filter_centroids_min_spacing(
        self,
        centroids_raw: List[Tuple[float, float, bool]],
        car_x: float,
        car_y: float,
        min_dist: float = 1.0,
    ) -> List[Tuple[float, float, bool]]:
        """
        Enforce that:
          - no two centroids are closer than min_dist to each other
          - no centroid is within 0.5 m of the car pose

        centroids_raw: list of (x, y, is_big_big)
        Strategy:
          - compute distance from car
          - sort by distance DESC (farthest first)
          - greedily keep centroids that are:
              * ≥ 0.5 m from the car, AND
              * ≥ min_dist from all already kept centroids

        This matches: "No 2 centroid points should be obtained in less than 1m
        circle distance - connect to the farther one from the chain" and
        "ensure a point around 0.5m of the car pose is not registered".
        """
        if not centroids_raw:
            return []

        pts = np.array([[c[0], c[1]] for c in centroids_raw], dtype=float)
        car = np.array([car_x, car_y], dtype=float)
        d2_car = np.sum((pts - car) ** 2, axis=1)

        # indices sorted by distance from car (farthest first)
        order = np.argsort(-d2_car)

        kept_pts = []
        kept = []

        min_car_dist = 2  # [m] reject centroids inside this radius around car

        for idx in order:
            p = pts[idx]

            # Reject centroids too close to the car pose
            if np.linalg.norm(p - car) < min_car_dist:
                continue

            if not kept_pts:
                kept_pts.append(p)
                kept.append(
                    (float(p[0]), float(p[1]), bool(centroids_raw[idx][2]))
                )
                continue

            # Check spacing to all kept
            too_close = False
            for kp in kept_pts:
                if np.linalg.norm(p - kp) < min_dist:
                    too_close = True
                    break
            if too_close:
                continue

            kept_pts.append(p)
            kept.append(
                (float(p[0]), float(p[1]), bool(centroids_raw[idx][2]))
            )

        return kept

    def _build_greedy_path(self, car_x, car_y, candidate_points: List[Tuple[float, float]]):
        """
        Greedy NN path with max step length:

          - start at car (car_x, car_y)
          - repeatedly go to nearest unused candidate
          - BUT only if the hop distance <= 7 m

        Returns list of world-frame points [p1, p2, ...] in visitation order.
        """
        if not candidate_points:
            return []

        pts = np.asarray(candidate_points, dtype=float)
        N = pts.shape[0]
        used = np.zeros(N, dtype=bool)
        path_order: List[int] = []

        current = np.array([car_x, car_y], dtype=float)
        max_step2 = 7.0 * 7.0  # 7 m squared

        for _ in range(N):
            diff = pts - current
            d2 = np.sum(diff * diff, axis=1)

            # Mask out already-used points
            d2[used] = float("inf")

            # Enforce max step length: reject anything beyond 7 m
            d2[d2 > max_step2] = float("inf")

            idx = int(np.argmin(d2))
            if not math.isfinite(d2[idx]) or d2[idx] == float("inf"):
                # No remaining candidate within 7 m – stop the chain
                break

            used[idx] = True
            path_order.append(idx)
            current = pts[idx]

        return [tuple(pts[i]) for i in path_order]


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
        self.ax.set_title("Path-planning sector FOV (Delaunay + greedy NN)")

        all_x = []
        all_y = []

        # Plot all cones (global map)
        if blue_global:
            bx, by = zip(*blue_global)
            self.ax.scatter(bx, by, s=20, c="b", marker="o", label="blue cones")
            all_x.extend(bx)
            all_y.extend(by)

        if yellow_global:
            yx, yy = zip(*yellow_global)
            self.ax.scatter(
                yx, yy, s=20, c="y", marker="o", edgecolors="k", label="yellow cones"
            )
            all_x.extend(yx)
            all_y.extend(yy)

        if orange_global:
            ox, oy = zip(*orange_global)
            self.ax.scatter(ox, oy, s=20, c="orange", marker="o", label="orange cones")
            all_x.extend(ox)
            all_y.extend(oy)

        if big_global:
            gx, gy = zip(*big_global)
            self.ax.scatter(
                gx, gy, s=40, c="magenta", marker="^", label="big orange cones"
            )
            all_x.extend(gx)
            all_y.extend(gy)

        # If we don't have car pose yet, stop here
        if (car_x is None) or (car_y is None) or (car_yaw is None):
            if all_x and all_y:
                xmin, xmax = min(all_x), max(all_x)
                ymin, ymax = min(all_y), max(all_y)
                pad = 3.0
                if xmax - xmin < 1e-3:
                    xmax = xmin + 1.0
                if ymax - ymin < 1e-3:
                    ymax = ymin + 1.0
                self.ax.set_xlim(xmin - pad, xmax + pad)
                self.ax.set_ylim(ymin - pad, ymax + pad)
            return

        # Car pose triangle
        tri_len = 1.0
        tri_width = 0.5
        pts_local = np.array([
            [tri_len, 0.0],
            [-tri_len * 0.5, -tri_width],
            [-tri_len * 0.5, +tri_width],
        ])
        c = math.cos(car_yaw)
        s = math.sin(car_yaw)
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
        all_x.append(car_x)
        all_y.append(car_y)

        # Draw the FOV sector (dotted)
        vertex_local = np.array([self.vertex_offset, 0.0])  # in car frame
        vertex_world = (R @ vertex_local) + np.array([car_x, car_y])
        vx, vy = vertex_world[0], vertex_world[1]

        # Outer arc
        arc_points = []
        R_arc = self.fov_radius
        for k in range(0, 61):  # 60 segments
            theta = -self.fov_half_rad + (2.0 * self.fov_half_rad) * (k / 60.0)
            # point in local frame relative to vertex
            px_local = self.vertex_offset + R_arc * math.cos(theta)
            py_local = 0.0 + R_arc * math.sin(theta)
            p_world = (R @ np.array([px_local, py_local])) + np.array([car_x, car_y])
            arc_points.append(p_world)
        arc_points = np.asarray(arc_points)
        self.ax.plot(arc_points[:, 0], arc_points[:, 1], linestyle=":", color="gray")

        # Two radial edges (vertex -> arc ends)
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

        # Compute cones inside FOV
        points_window, classes_window = self._compute_fov_points(
            car_x, car_y, car_yaw
        )

        # Build graph edges inside FOV
        edges = self._build_edges(points_window, classes_window)

        orange_like = {"orange", "big"}

        # First pass: draw edges, collect raw centroids
        centroids_raw: List[Tuple[float, float, bool]] = []  # (mx, my, is_big_big)

        for (i, j) in edges:
            (x1, y1) = points_window[i]
            (x2, y2) = points_window[j]
            ci = classes_window[i]
            cj = classes_window[j]

            is_big_big = (ci == "big" and cj == "big")

            if ci in orange_like and cj in orange_like:
                color = "orange"
            else:
                color = "k"

            # edge
            self.ax.plot([x1, x2], [y1, y2], color=color, linewidth=1.0)

            # centroid
            mx = 0.5 * (x1 + x2)
            my = 0.5 * (y1 + y2)
            centroids_raw.append((mx, my, is_big_big))

        # Enforce ≥1 m spacing between centroids, preferring farther from car,
        # and reject centroids in a 0.5 m radius around the car
        centroids = self._filter_centroids_min_spacing(
            centroids_raw, car_x, car_y, min_dist=1.0
        )

        # Plot final centroids and prepare candidate points for greedy path
        candidate_points: List[Tuple[float, float]] = []

        for (mx, my, is_big_big) in centroids:
            # X at centroid
            self.ax.scatter([mx], [my], marker="x", c="k", s=25)
            # Green square if big-big edge
            if is_big_big:
                self.ax.scatter([mx], [my], marker="s", c="g", s=40)
            candidate_points.append((mx, my))
            all_x.append(mx)
            all_y.append(my)

        # Build greedy NN path over visible centroids
        path_points = self._build_greedy_path(car_x, car_y, candidate_points)

        if path_points:
            px = [car_x]
            py = [car_y]
            for (x, y) in path_points:
                px.append(x)
                py.append(y)
            self.ax.plot(px, py, color="cyan", linewidth=1.5, label="greedy path")

        # Auto-fit axes
        if all_x and all_y:
            xmin, xmax = min(all_x), max(all_x)
            ymin, ymax = min(all_y), max(all_y)
            pad = 3.0
            if xmax - xmin < 1e-3:
                xmax = xmin + 1.0
            if ymax - ymin < 1e-3:
                ymax = ymin + 1.0
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
    node = SlamPathSectorVisualizer()
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
