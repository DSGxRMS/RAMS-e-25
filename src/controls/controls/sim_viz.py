#!/usr/bin/env python3
"""
sim_viz.py — live MATLAB-style visualisation of the BASELINE planner in EUFS Sim.

Built for the working three-node setup:
    ~/run_eufsim.sh
    eufs_adapter.py  -p planner:=baseline
    pure_pursuit.py

Left panel : the whole track (from /ground_truth/track), the car, its driven trace
             and the current plan, with a box marking the zoom region.
Right panel: the car's local view - Delaunay triangulation of the cones the planner
             actually sees, which edges survive the colour/length constraints, the
             midpoint candidates, and the published path.

Frames, kept simple on purpose:
    /ground_truth/track   global map frame   -> left panel only
    /ground_truth/cones   CAR frame          -> right panel, no transform needed
    /path_points          base_footprint     -> right panel directly; transformed
                                                to global for the left panel
    /ground_truth/odom    car pose in map    -> trace + transform

    python3 sim_viz.py --ros-args -p use_sim_time:=true
    python3 sim_viz.py --ros-args -p use_sim_time:=true -p record:=true

Params
    max_edge_m 6.0 | fov_vertex_m -5.0 | fov_radius_m 30.0 | fov_angle_deg 60.0
    zoom_m 15.0 | rate_hz 5.0 | record false | gif_path ~/sim_planning.gif | gif_fps 10
"""

import math
import os
import threading

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy,
                       QoSDurabilityPolicy)

from nav_msgs.msg import Odometry, Path
from eufs_msgs.msg import ConeArrayWithCovariance

try:
    from scipy.spatial import Delaunay
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

BLUE, YELLOW, ORANGE = "#2b6cb0", "#d69e2e", "#dd6b20"
GREEN, PURPLE = "#2f855a", "#6b46c1"


def yaw_from_quat(qx, qy, qz, qw):
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


class SimViz(Node):
    def __init__(self):
        super().__init__("sim_viz")

        self.declare_parameter("track_topic", "/ground_truth/track")
        self.declare_parameter("cones_topic", "/ground_truth/cones")
        self.declare_parameter("path_topic", "/path_points")
        self.declare_parameter("odom_topic", "/ground_truth/odom")
        self.declare_parameter("max_edge_m", 6.0)
        self.declare_parameter("fov_vertex_m", -5.0)
        self.declare_parameter("fov_radius_m", 30.0)
        self.declare_parameter("fov_angle_deg", 60.0)
        self.declare_parameter("zoom_m", 15.0)
        self.declare_parameter("rate_hz", 5.0)
        self.declare_parameter("record", False)
        self.declare_parameter("gif_path", os.path.expanduser("~/sim_planning.gif"))
        self.declare_parameter("gif_fps", 10)
        self.declare_parameter("title", "Delaunay path planning in EUFS Sim")

        gp = self.get_parameter
        self.max_edge = float(gp("max_edge_m").value)
        self.fov_vertex = float(gp("fov_vertex_m").value)
        self.fov_r = float(gp("fov_radius_m").value)
        self.fov_half = math.radians(float(gp("fov_angle_deg").value) / 2)
        self.zoom = float(gp("zoom_m").value)
        self.record = bool(gp("record").value)
        self.gif_path = str(gp("gif_path").value)
        self.gif_fps = int(gp("gif_fps").value)

        rel = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=10,
                         reliability=QoSReliabilityPolicy.RELIABLE,
                         durability=QoSDurabilityPolicy.VOLATILE)
        be = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=10,
                        reliability=QoSReliabilityPolicy.BEST_EFFORT,
                        durability=QoSDurabilityPolicy.VOLATILE)

        self.lock = threading.Lock()
        self.track = {k: np.zeros((0, 2)) for k in ("blue", "yellow", "orange", "big")}
        self.local = {k: np.zeros((0, 2)) for k in ("blue", "yellow", "orange", "big")}
        self.path_local = np.zeros((0, 2))     # base_footprint
        self.pose = None
        self.trace = []
        self.frames = []

        self.create_subscription(ConeArrayWithCovariance, str(gp("track_topic").value),
                                 self.cb_track, rel)
        self.create_subscription(ConeArrayWithCovariance, str(gp("cones_topic").value),
                                 self.cb_cones, rel)
        self.create_subscription(Path, str(gp("path_topic").value), self.cb_path, be)
        self.create_subscription(Odometry, str(gp("odom_topic").value), self.cb_odom, rel)

        self._setup(str(gp("title").value))
        self.create_timer(1.0 / max(float(gp("rate_hz").value), 1e-6), self.redraw)
        self.get_logger().info(
            f"sim_viz up | zoom={self.zoom} m | record={self.record}"
            + (f" -> {self.gif_path}" if self.record else ""))

    # ---------------- subscriptions ----------------

    @staticmethod
    def _arr(lst):
        if not lst:
            return np.zeros((0, 2))
        return np.array([[float(c.point.x), float(c.point.y)] for c in lst])

    def cb_track(self, msg):
        with self.lock:
            self.track = dict(blue=self._arr(msg.blue_cones), yellow=self._arr(msg.yellow_cones),
                              orange=self._arr(msg.orange_cones), big=self._arr(msg.big_orange_cones))

    def cb_cones(self, msg):
        with self.lock:
            self.local = dict(blue=self._arr(msg.blue_cones), yellow=self._arr(msg.yellow_cones),
                              orange=self._arr(msg.orange_cones), big=self._arr(msg.big_orange_cones))

    def cb_path(self, msg):
        pts = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses]) \
            if msg.poses else np.zeros((0, 2))
        with self.lock:
            self.path_local = pts

    def cb_odom(self, msg):
        q = msg.pose.pose.orientation
        p = (float(msg.pose.pose.position.x), float(msg.pose.pose.position.y),
             yaw_from_quat(q.x, q.y, q.z, q.w))
        with self.lock:
            self.pose = p
            if not self.trace or (p[0] - self.trace[-1][0]) ** 2 + \
                    (p[1] - self.trace[-1][1]) ** 2 > 0.04:
                self.trace.append((p[0], p[1]))

    # ---------------- the planner's geometry, recomputed for display ----------------

    def _fov_local(self, cones):
        """Cones inside the sector FOV. Everything is already in the CAR frame,
        so the car is at the origin facing +x - no transform needed."""
        pts, cls = [], []
        for name in ("blue", "yellow", "orange", "big"):
            for (lx, ly) in cones[name]:
                vx, vy = lx - self.fov_vertex, ly
                r = math.hypot(vx, vy)
                if r > self.fov_r or r < 1e-3:
                    continue
                if abs(math.atan2(vy, vx)) <= self.fov_half:
                    pts.append((lx, ly)); cls.append(name)
        return (np.array(pts) if pts else np.zeros((0, 2))), cls

    def _triangulate(self, pts, cls):
        if len(pts) < 3 or not _HAS_SCIPY:
            return [], [], np.zeros((0, 2))
        try:
            simplices = Delaunay(pts).simplices
        except Exception:
            return [], [], np.zeros((0, 2))
        edges = set()
        for tri in simplices:
            i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
            for a, b in ((i, j), (j, k), (k, i)):
                edges.add((min(a, b), max(a, b)))

        orange_like = {"orange", "big"}
        blue_yellow = {"blue", "yellow"}
        kept, rej, mids = [], [], []
        for i, j in edges:
            seg = (pts[i], pts[j])
            ci, cj = cls[i], cls[j]
            too_long = np.sum((pts[i] - pts[j]) ** 2) >= self.max_edge ** 2
            same = (ci == "blue" and cj == "blue") or (ci == "yellow" and cj == "yellow")
            cross = ((ci in orange_like and cj in blue_yellow) or
                     (cj in orange_like and ci in blue_yellow))
            if too_long or same or cross:
                rej.append(seg)
            else:
                kept.append(seg)
                if not (ci == "big" and cj == "big"):
                    mids.append((pts[i] + pts[j]) / 2)
        return kept, rej, (np.array(mids) if mids else np.zeros((0, 2)))

    @staticmethod
    def _to_global(pts, pose):
        if len(pts) == 0:
            return pts
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        return pts @ np.array([[c, -s], [s, c]]).T + np.array([x, y])

    # ---------------- figure ----------------

    def _setup(self, title):
        plt.ion()
        self.fig = plt.figure(figsize=(13.5, 6.2))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.2)
        self.ax1, self.ax2 = self.fig.add_subplot(gs[0]), self.fig.add_subplot(gs[1])
        for ax in (self.ax1, self.ax2):
            ax.set_aspect("equal"); ax.grid(alpha=0.15, lw=0.5)
            ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        self.ax1.set_title("track + car + plan (map frame)")
        self.ax2.set_title("what the planner sees (car frame)")

        self.t_blue = self.ax1.scatter([], [], s=9, c=BLUE, zorder=3)
        self.t_yellow = self.ax1.scatter([], [], s=9, c=YELLOW, zorder=3)
        self.t_orange = self.ax1.scatter([], [], s=12, c=ORANGE, zorder=3)
        self.t_trace, = self.ax1.plot([], [], "--", color="0.45", lw=1.1, zorder=4)
        self.t_path, = self.ax1.plot([], [], "-", color=GREEN, lw=2.2, zorder=5)
        self.t_car, = self.ax1.plot([], [], "s", color="k", ms=7, zorder=6)
        self.t_box, = self.ax1.plot([], [], "-", color="0.35", lw=0.9, zorder=6)

        self.e_keep = LineCollection([], colors="#2d3748", linewidths=1.5, zorder=3)
        self.e_rej = LineCollection([], colors="#a0aec0", linewidths=0.8,
                                   linestyles=":", zorder=2)
        self.ax2.add_collection(self.e_keep); self.ax2.add_collection(self.e_rej)
        self.l_blue = self.ax2.scatter([], [], s=45, c=BLUE, zorder=5,
                                       edgecolors="k", linewidths=0.3)
        self.l_yellow = self.ax2.scatter([], [], s=45, c=YELLOW, zorder=5,
                                         edgecolors="k", linewidths=0.3)
        self.l_orange = self.ax2.scatter([], [], s=55, c=ORANGE, zorder=5,
                                         edgecolors="k", linewidths=0.3)
        self.l_mid = self.ax2.scatter([], [], s=55, c=PURPLE, marker="x",
                                      linewidths=1.8, zorder=6)
        self.l_path, = self.ax2.plot([], [], "-o", color=GREEN, lw=2.4, ms=3, zorder=7)
        self.l_fov, = self.ax2.plot([], [], "-", color="#3182ce", lw=0.8,
                                    alpha=0.55, zorder=1)
        self.l_car, = self.ax2.plot([0], [0], "s", color="k", ms=9, zorder=8)
        self.info = self.ax2.text(0.02, 0.98, "", transform=self.ax2.transAxes,
                                  va="top", fontsize=8.5,
                                  bbox=dict(boxstyle="round", fc="white",
                                            ec="0.75", alpha=0.9))
        self.ax2.legend(handles=[
            Line2D([], [], color="#2d3748", lw=1.5, label="kept edge (blue-yellow, <6 m)"),
            Line2D([], [], color="#a0aec0", lw=1.0, ls=":", label="rejected edge"),
            Line2D([], [], color="#3182ce", lw=0.9, alpha=0.6, label="sector FOV"),
            Line2D([], [], color=PURPLE, marker="x", ls="", label="midpoint candidate"),
            Line2D([], [], color=GREEN, lw=2.2, label="published path"),
        ], fontsize=7.5, loc="lower right", framealpha=0.9)

        self.fig.suptitle(title)
        self.fig.tight_layout()
        self.fig.show()

    @staticmethod
    def _set(sc, pts):
        sc.set_offsets(pts if len(pts) else np.zeros((0, 2)))

    def redraw(self):
        with self.lock:
            pose = self.pose
            track = {k: v.copy() for k, v in self.track.items()}
            local = {k: v.copy() for k, v in self.local.items()}
            path_l = self.path_local.copy()
            trace = np.array(self.trace) if self.trace else np.zeros((0, 2))
        if pose is None:
            return

        fov_pts, fov_cls = self._fov_local(local)
        keep, rej, mids = self._triangulate(fov_pts, fov_cls)
        cx, cy, yaw = pose

        # ---- left: global ----
        self._set(self.t_blue, track["blue"])
        self._set(self.t_yellow, track["yellow"])
        orange = np.vstack([track["orange"], track["big"]]) \
            if len(track["orange"]) or len(track["big"]) else np.zeros((0, 2))
        self._set(self.t_orange, orange)
        if len(trace):
            self.t_trace.set_data(trace[:, 0], trace[:, 1])
        path_g = self._to_global(path_l, pose)
        if len(path_g):
            self.t_path.set_data(path_g[:, 0], path_g[:, 1])
        else:
            self.t_path.set_data([], [])
        self.t_car.set_data([cx], [cy])
        self.t_box.set_data([cx - self.zoom, cx + self.zoom, cx + self.zoom,
                             cx - self.zoom, cx - self.zoom],
                            [cy - self.zoom, cy - self.zoom, cy + self.zoom,
                             cy + self.zoom, cy - self.zoom])
        allc = np.vstack([v for v in track.values() if len(v)]) \
            if any(len(v) for v in track.values()) else np.zeros((0, 2))
        if len(allc):
            m = 5
            self.ax1.set_xlim(allc[:, 0].min() - m, allc[:, 0].max() + m)
            self.ax1.set_ylim(allc[:, 1].min() - m, allc[:, 1].max() + m)

        # ---- right: car frame (car at origin, facing +x) ----
        self.e_keep.set_segments(keep); self.e_rej.set_segments(rej)
        self._set(self.l_blue, local["blue"])
        self._set(self.l_yellow, local["yellow"])
        lo = np.vstack([local["orange"], local["big"]]) \
            if len(local["orange"]) or len(local["big"]) else np.zeros((0, 2))
        self._set(self.l_orange, lo)
        self._set(self.l_mid, mids)
        if len(path_l):
            self.l_path.set_data(path_l[:, 0], path_l[:, 1])
        else:
            self.l_path.set_data([], [])

        apex = np.array([self.fov_vertex, 0.0])
        angs = np.linspace(-self.fov_half, self.fov_half, 30)
        arc = apex + self.fov_r * np.stack([np.cos(angs), np.sin(angs)], axis=1)
        poly = np.vstack([apex, arc, apex])
        self.l_fov.set_data(poly[:, 0], poly[:, 1])
        self.ax2.set_xlim(-self.zoom * 0.4, self.zoom * 1.6)
        self.ax2.set_ylim(-self.zoom, self.zoom)

        self.info.set_text(
            f"car ({cx:.1f}, {cy:.1f})  {math.degrees(yaw):.0f} deg\n"
            f"cones seen: {sum(len(v) for v in local.values())}\n"
            f"cones in FOV: {len(fov_pts)}\n"
            f"edges kept: {len(keep)} / {len(keep) + len(rej)}\n"
            f"midpoints: {len(mids)}\n"
            f"path points: {len(path_l)}")

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        if self.record:
            self.fig.canvas.draw()
            w, h = self.fig.canvas.get_width_height()
            if hasattr(self.fig.canvas, "tostring_rgb"):
                buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
                self.frames.append(buf.reshape(h, w, 3).copy())
            else:
                buf = np.asarray(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
                self.frames.append(buf.reshape(h, w, 4)[:, :, :3].copy())

    def save_gif(self):
        if not self.record or not self.frames:
            if self.record:
                self.get_logger().warn("no frames captured, nothing to save")
            return
        try:
            from PIL import Image
        except ImportError:
            self.get_logger().error("pillow missing: pip3 install pillow")
            return
        imgs = [Image.fromarray(f) for f in self.frames]
        imgs[0].save(self.gif_path, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / max(self.gif_fps, 1)), loop=0)
        self.get_logger().info(f"wrote {self.gif_path} ({len(imgs)} frames)")


def main():
    rclpy.init()
    node = SimViz()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_gif()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
