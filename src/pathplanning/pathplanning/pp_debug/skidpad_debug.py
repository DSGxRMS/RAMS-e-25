#!/usr/bin/env python3
# pp_debug.py
#
# Live debug plotter for /path_points (nav_msgs/Path) + /slam/map_cones.
# - Plots cones and car pose.
# - Plots received path points as scatter (NOT joined).
# - Fits and plots a B-spline through the received poses.
#
# Notes:
# - Requires: matplotlib, scipy
# - If SciPy is missing, this node will log an error and plot only the scatter points.

import math
import threading
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from eufs_msgs.msg import ConeArrayWithCovariance

import numpy as np

# Matplotlib
import matplotlib.pyplot as plt

# SciPy spline fitting (preferred)
try:
    from scipy.interpolate import splprep, splev
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def yaw_from_quat(qx, qy, qz, qw) -> float:
    """Return yaw (rad) from quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class PPDebug(Node):
    def __init__(self):
        super().__init__("pp_debug")

        # ---------------- Parameters ----------------
        self.declare_parameter("topics.path_in", "/path_points")
        self.declare_parameter("topics.map_in", "/slam/map_cones")

        self.declare_parameter("plot.hz", 20.0)

        # Spline params
        self.declare_parameter("spline.include_car_pose", True)   # include pose[0] in spline fit
        self.declare_parameter("spline.smoothing", 0.5)           # s in splprep (0 = interpolate)
        self.declare_parameter("spline.num_samples", 200)

        # Plotting params
        self.declare_parameter("plot.window_lock", False)         # if True, do not auto-rescale
        self.declare_parameter("plot.padding_m", 5.0)

        # QoS
        self.declare_parameter("qos.best_effort", True)
        self.declare_parameter("qos.depth", 50)

        gp = self.get_parameter
        self.path_topic = str(gp("topics.path_in").value)
        self.map_topic = str(gp("topics.map_in").value)

        self.plot_hz = float(gp("plot.hz").value)
        self.plot_period = 1.0 / max(1e-6, self.plot_hz)

        self.include_car_pose = bool(gp("spline.include_car_pose").value)
        self.spline_s = float(gp("spline.smoothing").value)
        self.spline_num = int(gp("spline.num_samples").value)

        self.window_lock = bool(gp("plot.window_lock").value)
        self.padding_m = float(gp("plot.padding_m").value)

        best_effort = bool(gp("qos.best_effort").value)
        depth = int(gp("qos.depth").value)

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=depth,
            reliability=QoSReliabilityPolicy.BEST_EFFORT if best_effort else QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # ---------------- State ----------------
        self.lock = threading.Lock()

        # Latest path message info
        self.frame_id: str = "map"
        self.car_pose: Optional[Tuple[float, float, float]] = None  # (x,y,yaw)
        self.path_points: List[Tuple[float, float]] = []            # remaining points (excluding car pose)

        # Cones
        self.blue: List[Tuple[float, float]] = []
        self.yellow: List[Tuple[float, float]] = []
        self.orange: List[Tuple[float, float]] = []
        self.big: List[Tuple[float, float]] = []

        # ---------------- Subscriptions ----------------
        self.create_subscription(Path, self.path_topic, self.cb_path, qos)
        self.create_subscription(ConeArrayWithCovariance, self.map_topic, self.cb_map, qos)

        # ---------------- Plot setup ----------------
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("pp_debug: /path_points B-spline + cones + car pose")
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True)

        # Persistent artists (initialized empty)
        self.scat_blue = self.ax.scatter([], [], label="blue cones")
        self.scat_yellow = self.ax.scatter([], [], label="yellow cones")
        self.scat_orange = self.ax.scatter([], [], label="orange cones")
        self.scat_big = self.ax.scatter([], [], label="big orange cones")

        self.scat_pts = self.ax.scatter([], [], label="path points (raw)")

        self.car_dot, = self.ax.plot([], [], marker="o", linestyle="", label="car")
        self.car_arrow = None

        self.spline_line, = self.ax.plot([], [], linewidth=2.0, label="B-spline")

        self.ax.legend(loc="upper right")

        if not _HAS_SCIPY:
            self.get_logger().error(
                "[pp_debug] SciPy not available. Install python3-scipy / pip scipy. "
                "Spline will NOT be drawn."
            )

        self.get_logger().info(
            f"[pp_debug] Subscribing path={self.path_topic}, cones={self.map_topic}, plot_hz={self.plot_hz}"
        )

        # Timer to refresh plot
        self.create_timer(self.plot_period, self.timer_cb)

    # ---------------- Callbacks ----------------
    def cb_path(self, msg: Path):
        poses = msg.poses
        if not poses:
            return

        with self.lock:
            self.frame_id = msg.header.frame_id.strip() or "map"

            # Pose[0] is current car pose (as published by your upstream node)
            p0 = poses[0].pose
            x0 = float(p0.position.x)
            y0 = float(p0.position.y)
            q = p0.orientation
            yaw0 = yaw_from_quat(q.x, q.y, q.z, q.w)
            self.car_pose = (x0, y0, yaw0)

            # Remaining poses are future points
            pts = []
            for ps in poses[1:]:
                pts.append((float(ps.pose.position.x), float(ps.pose.position.y)))
            self.path_points = pts

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

        with self.lock:
            self.blue = blue
            self.yellow = yellow
            self.orange = orange
            self.big = big

    # ---------------- Spline helpers ----------------
    def _fit_bspline(self, pts: np.ndarray) -> Optional[np.ndarray]:
        """
        Fit a B-spline through pts (N,2). Returns sampled curve (M,2) or None.
        Uses SciPy splprep/splev.

        - For small N, automatically reduces spline degree.
        """
        if not _HAS_SCIPY:
            return None

        n = pts.shape[0]
        if n < 2:
            return None

        # Reduce degree if too few points (splprep requires m > k)
        k = min(3, n - 1)
        if k < 1:
            return None

        # Parameterize and fit
        x = pts[:, 0]
        y = pts[:, 1]

        # s controls smoothing; s=0 -> interpolating spline
        try:
            tck, _u = splprep([x, y], s=self.spline_s, k=k)
            uu = np.linspace(0.0, 1.0, max(20, self.spline_num))
            out = splev(uu, tck)
            curve = np.vstack(out).T  # (M,2)
            return curve
        except Exception as e:
            self.get_logger().warn(f"[pp_debug] Spline fit failed: {e}")
            return None

    # ---------------- Plot refresh ----------------
    def timer_cb(self):
        with self.lock:
            car = self.car_pose
            pts = list(self.path_points)

            blue = list(self.blue)
            yellow = list(self.yellow)
            orange = list(self.orange)
            big = list(self.big)

        # Update scatter data for cones
        def _set_scatter(sc, data: List[Tuple[float, float]]):
            if not data:
                sc.set_offsets(np.empty((0, 2)))
            else:
                sc.set_offsets(np.array(data, dtype=float))

        _set_scatter(self.scat_blue, blue)
        _set_scatter(self.scat_yellow, yellow)
        _set_scatter(self.scat_orange, orange)
        _set_scatter(self.scat_big, big)

        # Car pose
        if car is not None:
            cx, cy, cyaw = car
            self.car_dot.set_data([cx], [cy])

            # Update heading arrow
            if self.car_arrow is not None:
                try:
                    self.car_arrow.remove()
                except Exception:
                    pass

            arrow_len = 2.0
            hx = cx + arrow_len * math.cos(cyaw)
            hy = cy + arrow_len * math.sin(cyaw)
            self.car_arrow = self.ax.annotate(
                "",
                xy=(hx, hy),
                xytext=(cx, cy),
                arrowprops=dict(arrowstyle="->", linewidth=2.0),
            )
        else:
            self.car_dot.set_data([], [])

        # Raw path points scatter (NOT joined)
        if pts:
            self.scat_pts.set_offsets(np.array(pts, dtype=float))
        else:
            self.scat_pts.set_offsets(np.empty((0, 2)))

        # Build spline input points (car + path points) or just path points
        spline_pts_list: List[Tuple[float, float]] = []
        if self.include_car_pose and car is not None:
            spline_pts_list.append((car[0], car[1]))
        spline_pts_list.extend(pts)

        spline_curve = None
        if len(spline_pts_list) >= 2:
            spline_curve = self._fit_bspline(np.array(spline_pts_list, dtype=float))

        if spline_curve is not None and spline_curve.shape[0] >= 2:
            self.spline_line.set_data(spline_curve[:, 0], spline_curve[:, 1])
        else:
            self.spline_line.set_data([], [])

        # Auto-scale view (unless locked)
        if not self.window_lock:
            all_xy = []
            if car is not None:
                all_xy.append((car[0], car[1]))
            all_xy.extend(pts)
            all_xy.extend(blue)
            all_xy.extend(yellow)
            all_xy.extend(orange)
            all_xy.extend(big)

            if all_xy:
                arr = np.array(all_xy, dtype=float)
                xmin, ymin = arr.min(axis=0)
                xmax, ymax = arr.max(axis=0)
                pad = self.padding_m
                self.ax.set_xlim(xmin - pad, xmax + pad)
                self.ax.set_ylim(ymin - pad, ymax + pad)

        # Refresh
        self.fig.canvas.draw_idle()
        plt.pause(0.001)


def main():
    rclpy.init()
    node = PPDebug()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            plt.close("all")
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
