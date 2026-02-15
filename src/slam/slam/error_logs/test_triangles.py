#!/usr/bin/env python3
"""
step2_triangle_graph_debug_plotter.py

Step 2 visualiser (Matplotlib-only):
- Subscribes to /perception/cones_fused (PointCloud2)
- Treats incoming points as 2D (x,y) in the message frame (e.g., velodyne)
- Builds an observation structural graph as TRIANGLES using k-NN anchoring
- Filters triangles by:
    * max edge length
    * min area
    * (optional) color-pattern rule (default: reject all-same-color triangles)
- Visualises:
    Panel A: cones with indices
    Panel B: cones + triangles overlay + HUD stats

Run:
  ros2 run <your_pkg> step2_triangle_graph_debug_plotter --ros-args -p use_sim_time:=true
or:
  python3 step2_triangle_graph_debug_plotter.py --ros-args -p use_sim_time:=true

Notes:
- Uses TkAgg backend; needs a desktop session.
- If you run in sim with /clock, set use_sim_time:=true.
"""

import math
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from rclpy.time import Time as RclTime

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


@dataclass
class ObsFrame:
    t: float
    frame_id: str
    pts: np.ndarray   # (N,2)
    cls: np.ndarray   # (N,)


def tri_area2(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Twice signed area (2D cross product)."""
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def pairwise_dist2(P: np.ndarray) -> np.ndarray:
    """Return NxN squared distance matrix."""
    # (N,1,2) - (1,N,2) => (N,N,2)
    D = P[:, None, :] - P[None, :, :]
    return np.sum(D * D, axis=2)


class Step2TriangleGraphPlotter(Node):
    def __init__(self):
        super().__init__("step2_triangle_graph_debug_plotter")

        # ---------------- Params ----------------
        self.declare_parameter("topics.cones_in", "/perception/cones_fused")

        self.declare_parameter("qos.best_effort", True)
        self.declare_parameter("qos.depth", 5)

        # k-NN triangle builder
        self.declare_parameter("tri.k", 4)                 # neighbors per anchor point (3 or 4 recommended)
        self.declare_parameter("tri.max_edge_m", 6.0)      # reject triangles with any edge > this
        self.declare_parameter("tri.min_area_m2", 0.30)    # reject triangles with area < this
        self.declare_parameter("tri.reject_all_same_color", True)

        # Plot bounds (auto-scale if 0)
        self.declare_parameter("plot.fixed_bounds", False)
        self.declare_parameter("plot.xlim", 30.0)
        self.declare_parameter("plot.ylim", 15.0)

        gp = self.get_parameter
        self.cones_topic = str(gp("topics.cones_in").value)

        best_effort = bool(gp("qos.best_effort").value)
        depth = int(gp("qos.depth").value)

        self.k = int(gp("tri.k").value)
        self.max_edge = float(gp("tri.max_edge_m").value)
        self.min_area = float(gp("tri.min_area_m2").value)
        self.reject_same = bool(gp("tri.reject_all_same_color").value)

        self.fixed_bounds = bool(gp("plot.fixed_bounds").value)
        self.xlim = float(gp("plot.xlim").value)
        self.ylim = float(gp("plot.ylim").value)

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=depth,
            reliability=QoSReliabilityPolicy.BEST_EFFORT if best_effort else QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # ---------------- State ----------------
        self.lock = threading.Lock()
        self.latest: Optional[ObsFrame] = None

        # For stability monitoring (optional)
        self.prev_triangles: Optional[Set[Tuple[int, int, int]]] = None

        self.create_subscription(PointCloud2, self.cones_topic, self.cb_cones, qos)

        self.get_logger().info(
            f"[step2] cones_in={self.cones_topic} | k={self.k} | max_edge={self.max_edge}m | "
            f"min_area={self.min_area}m^2 | reject_same_color={self.reject_same}"
        )

    def cb_cones(self, msg: PointCloud2):
        t_c = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        frame_id = str(msg.header.frame_id)

        try:
            pts = []
            cls = []
            for x, y, _z, cid in pc2.read_points(msg, field_names=("x", "y", "z", "class_id"), skip_nans=True):
                pts.append((float(x), float(y)))
                cls.append(int(cid))
        except Exception as e:
            self.get_logger().warn(f"[step2] parse error: {e}")
            return

        P = np.asarray(pts, dtype=np.float64) if pts else np.zeros((0, 2), dtype=np.float64)
        C = np.asarray(cls, dtype=np.int32) if cls else np.zeros((0,), dtype=np.int32)

        with self.lock:
            self.latest = ObsFrame(t=t_c, frame_id=frame_id, pts=P, cls=C)

    # ---------------- Triangle builder ----------------
    def build_triangles(self, P: np.ndarray, C: np.ndarray):
        """
        Returns:
          triangles_all: set of index triplets (i,j,k) canonical sorted
          triangles_ok: set of index triplets after filters
          stats: dict
        """
        N = P.shape[0]
        triangles_all: Set[Tuple[int, int, int]] = set()
        triangles_ok: Set[Tuple[int, int, int]] = set()

        stats = {
            "N": N,
            "raw_triangles": 0,
            "kept_triangles": 0,
            "rej_edge": 0,
            "rej_area": 0,
            "rej_color": 0,
        }

        if N < 3:
            return triangles_all, triangles_ok, stats

        k = max(1, min(self.k, N - 1))
        D2 = pairwise_dist2(P)
        max_edge2 = self.max_edge * self.max_edge

        # Precompute kNN lists (exclude self)
        nn_list: List[List[int]] = []
        for i in range(N):
            order = np.argsort(D2[i])  # includes i at index 0
            neigh = [int(j) for j in order[1:1 + k]]
            nn_list.append(neigh)

        # Generate triangles anchored at i from its neighbor pairs
        for i in range(N):
            neigh = nn_list[i]
            if len(neigh) < 2:
                continue
            # all pairs among neigh
            for a_idx in range(len(neigh)):
                for b_idx in range(a_idx + 1, len(neigh)):
                    j = neigh[a_idx]
                    k2 = neigh[b_idx]
                    tri = tuple(sorted((i, j, k2)))
                    triangles_all.add(tri)

        stats["raw_triangles"] = len(triangles_all)

        # Filters
        for (i, j, k2) in triangles_all:
            a, b, c = P[i], P[j], P[k2]

            # Edge length filter
            dij2 = float(np.sum((a - b) ** 2))
            dik2 = float(np.sum((a - c) ** 2))
            djk2 = float(np.sum((b - c) ** 2))
            if max(dij2, dik2, djk2) > max_edge2:
                stats["rej_edge"] += 1
                continue

            # Area filter (use absolute area)
            area2 = abs(tri_area2(a, b, c))
            area = 0.5 * area2
            if area < self.min_area:
                stats["rej_area"] += 1
                continue

            # Color filter
            if self.reject_same:
                ci, cj, ck = int(C[i]), int(C[j]), int(C[k2])
                if ci == cj == ck:
                    stats["rej_color"] += 1
                    continue

            triangles_ok.add((i, j, k2))

        stats["kept_triangles"] = len(triangles_ok)
        return triangles_all, triangles_ok, stats


def run_plot(node: Step2TriangleGraphPlotter):
    plt.ion()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle("Step 2 — Observation Triangles (kNN)")

    # Setup axes
    for ax in (ax_a, ax_b):
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.axhline(0.0, linewidth=1.0)
        ax.axvline(0.0, linewidth=1.0)

    ax_a.set_title("A) Cones with indices")
    ax_a.set_xlabel("x [m]")
    ax_a.set_ylabel("y [m]")

    ax_b.set_title("B) Cones + filtered triangles")
    ax_b.set_xlabel("x [m]")
    ax_b.set_ylabel("y [m]")

    # HUD
    hud = ax_b.text(
        0.02, 0.98, "", transform=ax_b.transAxes,
        va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.15)
    )

    # Scatter handles
    scat_a = ax_a.scatter([], [], s=30)
    scat_b = ax_b.scatter([], [], s=30)

    # For labels
    txt_a: List[plt.Text] = []
    txt_b: List[plt.Text] = []

    # Triangle line artists
    tri_lines: List[plt.Line2D] = []

    def clear_text(text_list):
        for t in text_list:
            try:
                t.remove()
            except Exception:
                pass
        text_list.clear()

    def clear_lines(lines):
        for ln in lines:
            try:
                ln.remove()
            except Exception:
                pass
        lines.clear()

    def class_to_marker_color(cls_id: int):
        # Let matplotlib choose default cycle colors; we just segment by class
        # We'll approximate: 0-blue,1-yellow,2-orange,3-big orange, else grey
        if cls_id == 0:
            return "C0"
        if cls_id == 1:
            return "C1"
        if cls_id == 2:
            return "C2"
        if cls_id == 3:
            return "C3"
        return "C7"

    def update_once():
        with node.lock:
            fr = node.latest

        if fr is None:
            hud.set_text("waiting for /perception/cones_fused ...")
            fig.canvas.draw_idle()
            return

        P = fr.pts
        C = fr.cls
        N = P.shape[0]

        # Auto bounds
        if node.fixed_bounds:
            for ax in (ax_a, ax_b):
                ax.set_xlim(-2.0, node.xlim)
                ax.set_ylim(-node.ylim, node.ylim)
        else:
            if N > 0:
                xmin, xmax = float(np.min(P[:, 0])), float(np.max(P[:, 0]))
                ymin, ymax = float(np.min(P[:, 1])), float(np.max(P[:, 1]))
                pad_x = max(2.0, 0.15 * (xmax - xmin + 1e-6))
                pad_y = max(2.0, 0.15 * (ymax - ymin + 1e-6))
                for ax in (ax_a, ax_b):
                    ax.set_xlim(xmin - pad_x, xmax + pad_x)
                    ax.set_ylim(ymin - pad_y, ymax + pad_y)
            else:
                for ax in (ax_a, ax_b):
                    ax.set_xlim(-2.0, 30.0)
                    ax.set_ylim(-15.0, 15.0)

        # Update scatters by class (we draw as one scatter per panel for simplicity)
        if N == 0:
            scat_a.set_offsets(np.zeros((0, 2)))
            scat_b.set_offsets(np.zeros((0, 2)))
            clear_text(txt_a)
            clear_text(txt_b)
            clear_lines(tri_lines)
            hud.set_text(f"t={fr.t:.3f}s frame={fr.frame_id} | N=0")
            fig.canvas.draw_idle()
            return

        # Color array per point
        colors = [class_to_marker_color(int(cid)) for cid in C]
        scat_a.set_offsets(P)
        scat_a.set_color(colors)
        scat_b.set_offsets(P)
        scat_b.set_color(colors)

        # Labels
        clear_text(txt_a)
        clear_text(txt_b)
        for i in range(N):
            x, y = float(P[i, 0]), float(P[i, 1])
            txt_a.append(ax_a.text(x, y, str(i), fontsize=9))
            txt_b.append(ax_b.text(x, y, str(i), fontsize=9, alpha=0.7))

        # Triangles
        tris_all, tris_ok, stats = node.build_triangles(P, C)

        # Stability metric: Jaccard overlap of filtered triangles vs previous
        jacc = None
        if node.prev_triangles is not None:
            A = node.prev_triangles
            B = tris_ok
            inter = len(A.intersection(B))
            uni = len(A.union(B))
            jacc = inter / uni if uni > 0 else 1.0
        node.prev_triangles = set(tris_ok)

        clear_lines(tri_lines)
        # Draw triangles (filtered only)
        for (i, j, k2) in tris_ok:
            poly = np.array([P[i], P[j], P[k2], P[i]], dtype=np.float64)
            ln, = ax_b.plot(poly[:, 0], poly[:, 1], linewidth=1.0, alpha=0.8)
            tri_lines.append(ln)

        # HUD text
        hud_lines = [
            f"t={fr.t:.3f}s frame={fr.frame_id}",
            f"N={stats['N']} | raw_tris={stats['raw_triangles']} | kept={stats['kept_triangles']}",
            f"rej_edge={stats['rej_edge']} rej_area={stats['rej_area']} rej_color={stats['rej_color']}",
            f"params: k={node.k} max_edge={node.max_edge:.2f}m min_area={node.min_area:.2f}m²",
        ]
        if jacc is not None:
            hud_lines.append(f"stability (Jaccard vs prev): {jacc:.2f}")
        hud.set_text("\n".join(hud_lines))

        fig.canvas.draw_idle()

    try:
        while rclpy.ok():
            update_once()
            plt.pause(0.05)  # ~20 Hz UI refresh
    except KeyboardInterrupt:
        pass


def main():
    rclpy.init()
    node = Step2TriangleGraphPlotter()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        run_plot(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
