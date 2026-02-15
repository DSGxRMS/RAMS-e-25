#!/usr/bin/env python3
"""
step4_structural_backend_viz_4p1_PATCHED.py

Patch goals:
- Stop stray/garbage tracks from generating triangles
- Reduce duplication via merge pass
- Visualize trusted vs untrusted tracks

Key changes:
1) Track merge: same class within merge_dist
2) Triangle eligibility filter (TRUSTED tracks only):
   - hits >= tri.min_hits
   - age <= tri.max_track_age_sec
   - degree >= tri.min_degree (neighbors within tri.degree_radius_m)
3) Untrusted tracks plotted as faint 'x', triangles only from trusted subset
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

from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat(qx, qy, qz, qw) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def tri_area2(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def pairwise_dist2(P: np.ndarray) -> np.ndarray:
    D = P[:, None, :] - P[None, :, :]
    return np.sum(D * D, axis=2)


def class_to_color(cls_id: int) -> str:
    if cls_id == 0:
        return "C0"
    if cls_id == 1:
        return "C1"
    if cls_id == 2:
        return "C2"
    if cls_id == 3:
        return "C3"
    return "C7"


def color_token(cls_id: int) -> str:
    if cls_id == 0:
        return "B"
    if cls_id == 1:
        return "Y"
    if cls_id == 2:
        return "O"
    if cls_id == 3:
        return "A"
    return "U"


@dataclass
class ObsFrame:
    t: float
    frame_id: str
    pts_body: np.ndarray
    cls: np.ndarray


@dataclass
class Track:
    id: int
    x: float
    y: float
    cls: int
    last_seen: float
    hits: int = 1


class Step4p1StructuralViz(Node):
    def __init__(self):
        super().__init__("step4_structural_backend_viz_4p1")

        # Topics
        self.declare_parameter("topics.cones_in", "/perception/cones_fused")
        self.declare_parameter("topics.odom_in", "/slam/odom_raw")

        # QoS
        self.declare_parameter("qos.best_effort", True)
        self.declare_parameter("qos.depth", 10)

        # Odom buffer
        self.declare_parameter("odom_buffer_sec", 6.0)

        # Triangle build
        self.declare_parameter("tri.k", 4)
        self.declare_parameter("tri.max_edge_m", 6.0)
        self.declare_parameter("tri.min_area_m2", 0.30)
        self.declare_parameter("tri.reject_all_same_color", True)

        # Signature binning
        self.declare_parameter("sig.edge_bin_m", 0.20)

        # Track map (odom)
        self.declare_parameter("tracks.max_age_sec", 6.0)
        self.declare_parameter("tracks.assoc_dist_m", 0.9)
        self.declare_parameter("tracks.ema_alpha", 0.35)

        # NEW: map hygiene
        self.declare_parameter("tracks.merge_dist_m", 0.7)   # same-class merge radius
        self.declare_parameter("tracks.merge_min_hits", 2)   # prefer keeping higher-hit tracks

        # NEW: triangle eligibility (TRUSTED set)
        self.declare_parameter("tri.min_hits", 2)                # only tracks with >= this hits can be vertices
        self.declare_parameter("tri.max_track_age_sec", 1.0)     # only tracks seen within this age can be vertices
        self.declare_parameter("tri.min_degree", 2)              # local neighbor count threshold
        self.declare_parameter("tri.degree_radius_m", 4.0)       # radius for degree counting

        # Plot bounds
        self.declare_parameter("plot.fixed_bounds", False)
        self.declare_parameter("plot.xlim", 35.0)
        self.declare_parameter("plot.ylim", 18.0)

        gp = self.get_parameter
        self.cones_topic = str(gp("topics.cones_in").value)
        self.odom_topic = str(gp("topics.odom_in").value)

        best_effort = bool(gp("qos.best_effort").value)
        depth = int(gp("qos.depth").value)

        self.odom_buffer_sec = float(gp("odom_buffer_sec").value)

        self.k = int(gp("tri.k").value)
        self.max_edge = float(gp("tri.max_edge_m").value)
        self.min_area = float(gp("tri.min_area_m2").value)
        self.reject_same = bool(gp("tri.reject_all_same_color").value)

        self.edge_bin = float(gp("sig.edge_bin_m").value)

        self.tr_max_age = float(gp("tracks.max_age_sec").value)
        self.tr_assoc = float(gp("tracks.assoc_dist_m").value)
        self.tr_assoc2 = self.tr_assoc * self.tr_assoc
        self.tr_alpha = float(gp("tracks.ema_alpha").value)

        self.merge_dist = float(gp("tracks.merge_dist_m").value)
        self.merge_dist2 = self.merge_dist * self.merge_dist
        self.merge_min_hits = int(gp("tracks.merge_min_hits").value)

        self.tri_min_hits = int(gp("tri.min_hits").value)
        self.tri_max_age = float(gp("tri.max_track_age_sec").value)
        self.tri_min_degree = int(gp("tri.min_degree").value)
        self.tri_deg_r = float(gp("tri.degree_radius_m").value)
        self.tri_deg_r2 = self.tri_deg_r * self.tri_deg_r

        self.fixed_bounds = bool(gp("plot.fixed_bounds").value)
        self.xlim = float(gp("plot.xlim").value)
        self.ylim = float(gp("plot.ylim").value)

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=depth,
            reliability=QoSReliabilityPolicy.BEST_EFFORT if best_effort else QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.lock = threading.Lock()
        self.latest_obs: Optional[ObsFrame] = None
        self.odom_buf: List[Tuple[float, float, float, float]] = []
        self.tracks: List[Track] = []
        self.next_track_id = 0
        self.last_track_stats = {"matched": 0, "spawned": 0, "pruned": 0, "merged": 0}

        self.create_subscription(Odometry, self.odom_topic, self.cb_odom, qos)
        self.create_subscription(PointCloud2, self.cones_topic, self.cb_cones, qos)

        self.get_logger().info(
            f"[step4.1 patched] cones_in={self.cones_topic} odom_in={self.odom_topic} | "
            f"merge_dist={self.merge_dist} | trusted: hits>={self.tri_min_hits}, age<={self.tri_max_age}s, "
            f"deg>={self.tri_min_degree}@{self.tri_deg_r}m"
        )

    # ------------------- Odom buffer -------------------

    def cb_odom(self, msg: Odometry):
        t = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

        with self.lock:
            self.odom_buf.append((t, x, y, yaw))
            tmin = t - self.odom_buffer_sec
            while self.odom_buf and self.odom_buf[0][0] < tmin:
                self.odom_buf.pop(0)

    def pose_at(self, t_query: float) -> Optional[Tuple[float, float, float]]:
        with self.lock:
            buf = list(self.odom_buf)

        if not buf:
            return None
        if t_query <= buf[0][0]:
            return buf[0][1], buf[0][2], buf[0][3]
        if t_query >= buf[-1][0]:
            return buf[-1][1], buf[-1][2], buf[-1][3]

        times = [it[0] for it in buf]
        idx = int(np.searchsorted(times, t_query, side="left"))
        idx = max(1, min(idx, len(buf) - 1))

        t0, x0, y0, yaw0 = buf[idx - 1]
        t1, x1, y1, yaw1 = buf[idx]
        if t1 == t0:
            return x0, y0, yaw0

        a = (t_query - t0) / (t1 - t0)
        x = x0 + a * (x1 - x0)
        y = y0 + a * (y1 - y0)
        yaw = wrap(yaw0 + a * wrap(yaw1 - yaw0))
        return x, y, yaw

    def body_to_odom(self, P_body: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
        c = math.cos(yaw)
        s = math.sin(yaw)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        return np.array([x, y], dtype=np.float64)[None, :] + (P_body @ R.T)

    # ------------------- Cones input + tracks -------------------

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
            self.get_logger().warn(f"[step4.1] parse error: {e}")
            return

        P_body = np.asarray(pts, dtype=np.float64) if pts else np.zeros((0, 2), dtype=np.float64)
        C = np.asarray(cls, dtype=np.int32) if cls else np.zeros((0,), dtype=np.int32)

        with self.lock:
            self.latest_obs = ObsFrame(t=t_c, frame_id=frame_id, pts_body=P_body, cls=C)

        pose = self.pose_at(t_c)
        if pose is None:
            return

        # prune always
        self._prune_tracks(t_c)

        if P_body.shape[0] == 0:
            return

        x, y, yaw = pose
        P_odom = self.body_to_odom(P_body, x, y, yaw)

        self._update_tracks(t_c, P_odom, C)

        merged = self._merge_tracks(t_c)
        if merged > 0:
            self.last_track_stats["merged"] = merged

    def _prune_tracks(self, t_now: float) -> int:
        before = len(self.tracks)
        self.tracks = [tr for tr in self.tracks if (t_now - tr.last_seen) <= self.tr_max_age]
        pruned = before - len(self.tracks)
        self.last_track_stats["pruned"] = pruned
        return pruned

    def _update_tracks(self, t_now: float, P_odom: np.ndarray, C: np.ndarray):
        matched = 0
        spawned = 0

        tracks_by_cls: Dict[int, List[int]] = {}
        for ti, tr in enumerate(self.tracks):
            tracks_by_cls.setdefault(int(tr.cls), []).append(ti)

        used_tracks: Set[int] = set()

        for oi in range(P_odom.shape[0]):
            ox, oy = float(P_odom[oi, 0]), float(P_odom[oi, 1])
            cls = int(C[oi])

            cand = tracks_by_cls.get(cls, [])
            best_ti = -1
            best_d2 = self.tr_assoc2

            for ti in cand:
                if ti in used_tracks:
                    continue
                tr = self.tracks[ti]
                dx = tr.x - ox
                dy = tr.y - oy
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_d2 = d2
                    best_ti = ti

            if best_ti >= 0:
                tr = self.tracks[best_ti]
                a = self.tr_alpha
                tr.x = (1.0 - a) * tr.x + a * ox
                tr.y = (1.0 - a) * tr.y + a * oy
                tr.last_seen = t_now
                tr.hits += 1
                used_tracks.add(best_ti)
                matched += 1
            else:
                self.tracks.append(Track(
                    id=self.next_track_id, x=ox, y=oy, cls=cls, last_seen=t_now, hits=1
                ))
                self.next_track_id += 1
                spawned += 1

        self.last_track_stats["matched"] = matched
        self.last_track_stats["spawned"] = spawned

    def _merge_tracks(self, t_now: float) -> int:
        """
        Merge same-class tracks within merge_dist.
        Weighted by hits. Keeps one representative, deletes the other.
        """
        if len(self.tracks) < 2:
            return 0

        P = np.array([[tr.x, tr.y] for tr in self.tracks], dtype=np.float64)
        C = np.array([int(tr.cls) for tr in self.tracks], dtype=np.int32)
        H = np.array([int(tr.hits) for tr in self.tracks], dtype=np.int32)

        D2 = pairwise_dist2(P)
        merged = 0
        alive = np.ones(len(self.tracks), dtype=bool)

        # simple greedy merge
        for i in range(len(self.tracks)):
            if not alive[i]:
                continue
            for j in range(i + 1, len(self.tracks)):
                if not alive[j]:
                    continue
                if C[i] != C[j]:
                    continue
                if D2[i, j] > self.merge_dist2:
                    continue

                # choose keeper (prefer higher hits)
                if H[i] >= H[j]:
                    keep, kill = i, j
                else:
                    keep, kill = j, i

                tr_k = self.tracks[keep]
                tr_d = self.tracks[kill]

                wk = float(tr_k.hits)
                wd = float(tr_d.hits)
                wsum = max(wk + wd, 1.0)

                tr_k.x = (wk * tr_k.x + wd * tr_d.x) / wsum
                tr_k.y = (wk * tr_k.y + wd * tr_d.y) / wsum
                tr_k.hits = int(wk + wd)
                tr_k.last_seen = max(tr_k.last_seen, tr_d.last_seen)

                alive[kill] = False
                merged += 1

        if merged > 0:
            self.tracks = [tr for k, tr in enumerate(self.tracks) if alive[k]]

        return merged

    # ------------------- TRUSTED set selection -------------------

    def trusted_track_mask(self, t_now: float, P: np.ndarray, tracks: List[Track]) -> np.ndarray:
        """
        TRUSTED = (hits >= tri_min_hits) AND (age <= tri_max_age) AND (degree >= tri_min_degree)
        degree computed as number of neighbors within tri_deg_r (excluding itself).
        """
        if len(tracks) == 0:
            return np.zeros((0,), dtype=bool)

        hits = np.array([tr.hits for tr in tracks], dtype=np.int32)
        ages = np.array([max(0.0, t_now - tr.last_seen) for tr in tracks], dtype=np.float64)

        base = (hits >= self.tri_min_hits) & (ages <= self.tri_max_age)

        if np.count_nonzero(base) < 3:
            return base

        D2 = pairwise_dist2(P)
        deg = np.sum((D2 <= self.tri_deg_r2), axis=1) - 1  # exclude self
        return base & (deg >= self.tri_min_degree)

    # ------------------- Triangles + signatures -------------------

    def build_triangles(self, P: np.ndarray, C: np.ndarray, eligible: Optional[np.ndarray] = None):
        """
        Build triangles from P, but only using vertices where eligible[i] == True if provided.
        """
        stats = {"N": int(P.shape[0]), "eligible": int(np.count_nonzero(eligible)) if eligible is not None else int(P.shape[0]),
                 "raw": 0, "kept": 0, "rej_edge": 0, "rej_area": 0, "rej_color": 0}
        if P.shape[0] < 3:
            return set(), [], stats

        if eligible is None:
            eligible = np.ones((P.shape[0],), dtype=bool)

        idxs = np.where(eligible)[0].tolist()
        if len(idxs) < 3:
            return set(), [], stats

        # work in compressed index space for stability
        P2 = P[idxs]
        C2 = C[idxs]
        N = P2.shape[0]

        k = max(1, min(self.k, N - 1))
        D2 = pairwise_dist2(P2)

        nn_list: List[List[int]] = []
        for i in range(N):
            order = np.argsort(D2[i])
            nn_list.append([int(j) for j in order[1:1 + k]])

        tris_all: Set[Tuple[int, int, int]] = set()
        for i in range(N):
            neigh = nn_list[i]
            if len(neigh) < 2:
                continue
            for a in range(len(neigh)):
                for b in range(a + 1, len(neigh)):
                    j = neigh[a]
                    k2 = neigh[b]
                    tris_all.add(tuple(sorted((i, j, k2))))

        stats["raw"] = len(tris_all)

        tris_ok: Set[Tuple[int, int, int]] = set()
        sigs: List[Tuple[str, Tuple[int, int, int]]] = []

        for (i, j, k2) in tris_all:
            a, b, c = P2[i], P2[j], P2[k2]
            dij = float(np.linalg.norm(a - b))
            dik = float(np.linalg.norm(a - c))
            djk = float(np.linalg.norm(b - c))

            if max(dij, dik, djk) > self.max_edge:
                stats["rej_edge"] += 1
                continue

            area = 0.5 * abs(tri_area2(a, b, c))
            if area < self.min_area:
                stats["rej_area"] += 1
                continue

            if self.reject_same:
                ci, cj, ck = int(C2[i]), int(C2[j]), int(C2[k2])
                if ci == cj == ck:
                    stats["rej_color"] += 1
                    continue

            # map back to original indices
            oi, oj, ok = idxs[i], idxs[j], idxs[k2]
            tris_ok.add((oi, oj, ok))

            cols = sorted([color_token(int(C[oi])), color_token(int(C[oj])), color_token(int(C[ok]))])
            col_sig = "".join(cols)

            edges = sorted([dij, dik, djk])
            b = max(self.edge_bin, 1e-6)
            edges_q = tuple(int(round(e / b)) for e in edges)
            sig = f"{col_sig}|{edges_q[0]}-{edges_q[1]}-{edges_q[2]}"

            sigs.append((sig, (oi, oj, ok)))

        stats["kept"] = len(tris_ok)
        return tris_ok, sigs, stats


def run_plot(node: Step4p1StructuralViz):
    plt.ion()
    fig, (ax_obs, ax_map, ax_sig) = plt.subplots(1, 3, figsize=(19, 7))
    fig.suptitle("SGMH-SLAM Step 4.1 (Patched) — Obs/Map Triangles + Signatures (Clean triangles)")

    for ax in (ax_obs, ax_map):
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.axhline(0.0, linewidth=1.0)
        ax.axvline(0.0, linewidth=1.0)

    ax_obs.set_title("A) Observation in ODOM frame + obs triangles")
    ax_obs.set_xlabel("X [m]")
    ax_obs.set_ylabel("Y [m]")

    ax_map.set_title("B) Tracks (ODOM): trusted points only make triangles")
    ax_map.set_xlabel("X [m]")
    ax_map.set_ylabel("Y [m]")

    hud = ax_map.text(
        0.02, 0.98, "", transform=ax_map.transAxes,
        va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.15)
    )

    scat_obs = ax_obs.scatter([], [], s=30)
    scat_map = ax_map.scatter([], [], s=30)
    scat_untrusted = ax_map.scatter([], [], marker="x", s=35)  # show junk points

    txt_obs: List[plt.Text] = []
    txt_map: List[plt.Text] = []
    lines_obs: List[plt.Line2D] = []
    lines_map: List[plt.Line2D] = []

    def clear_text(lst):
        for t in lst:
            try:
                t.remove()
            except Exception:
                pass
        lst.clear()

    def clear_lines(lst):
        for ln in lst:
            try:
                ln.remove()
            except Exception:
                pass
        lst.clear()

    def set_bounds(P_all: np.ndarray):
        if node.fixed_bounds:
            for ax in (ax_obs, ax_map):
                ax.set_xlim(-5.0, node.xlim)
                ax.set_ylim(-node.ylim, node.ylim)
            return
        if P_all.size == 0:
            for ax in (ax_obs, ax_map):
                ax.set_xlim(-5.0, 35.0)
                ax.set_ylim(-18.0, 18.0)
            return
        xmin, xmax = float(np.min(P_all[:, 0])), float(np.max(P_all[:, 0]))
        ymin, ymax = float(np.min(P_all[:, 1])), float(np.max(P_all[:, 1]))
        pad_x = max(3.0, 0.15 * (xmax - xmin + 1e-6))
        pad_y = max(3.0, 0.15 * (ymax - ymin + 1e-6))
        for ax in (ax_obs, ax_map):
            ax.set_xlim(xmin - pad_x, xmax + pad_x)
            ax.set_ylim(ymin - pad_y, ymax + pad_y)

    try:
        while rclpy.ok():
            with node.lock:
                obs = node.latest_obs
                tracks = list(node.tracks)
                tr_stats = dict(node.last_track_stats)

            if obs is None:
                hud.set_text("waiting for cones + odom ...")
                fig.canvas.draw_idle()
                plt.pause(0.05)
                continue

            pose = node.pose_at(obs.t)
            if pose is None:
                hud.set_text(f"t={obs.t:.3f}s | waiting for odom buffer ...")
                fig.canvas.draw_idle()
                plt.pause(0.05)
                continue

            x, y, yaw = pose
            P_obs_odom = node.body_to_odom(obs.pts_body, x, y, yaw)
            C_obs = obs.cls

            if tracks:
                P_map = np.array([[tr.x, tr.y] for tr in tracks], dtype=np.float64)
                C_map = np.array([int(tr.cls) for tr in tracks], dtype=np.int32)
            else:
                P_map = np.zeros((0, 2), dtype=np.float64)
                C_map = np.zeros((0,), dtype=np.int32)

            P_all = np.vstack([P_obs_odom, P_map]) if (P_obs_odom.size and P_map.size) else (P_map if P_map.size else P_obs_odom)
            set_bounds(P_all)

            # Obs triangles (no trust filtering)
            tris_obs, sigs_obs, st_obs = node.build_triangles(P_obs_odom, C_obs, eligible=None)

            # TRUSTED filtering for map triangles
            trusted = node.trusted_track_mask(obs.t, P_map, tracks) if len(tracks) else np.zeros((0,), dtype=bool)
            tris_map, sigs_map, st_map = node.build_triangles(P_map, C_map, eligible=trusted) if P_map.shape[0] >= 3 else (set(), [], {"N": int(P_map.shape[0]), "eligible": 0, "raw": 0, "kept": 0, "rej_edge": 0, "rej_area": 0, "rej_color": 0})

            # Panel A
            clear_text(txt_obs)
            clear_lines(lines_obs)
            if P_obs_odom.shape[0] == 0:
                scat_obs.set_offsets(np.zeros((0, 2)))
            else:
                colors = [class_to_color(int(cid)) for cid in C_obs]
                scat_obs.set_offsets(P_obs_odom)
                scat_obs.set_color(colors)
                for i in range(P_obs_odom.shape[0]):
                    txt_obs.append(ax_obs.text(float(P_obs_odom[i, 0]), float(P_obs_odom[i, 1]), str(i), fontsize=8))
                for (i, j, k2) in tris_obs:
                    poly = np.array([P_obs_odom[i], P_obs_odom[j], P_obs_odom[k2], P_obs_odom[i]], dtype=np.float64)
                    ln, = ax_obs.plot(poly[:, 0], poly[:, 1], linewidth=1.0, alpha=0.7)
                    lines_obs.append(ln)

            # Panel B (tracks)
            clear_text(txt_map)
            clear_lines(lines_map)

            if P_map.shape[0] == 0:
                scat_map.set_offsets(np.zeros((0, 2)))
                scat_untrusted.set_offsets(np.zeros((0, 2)))
            else:
                # plot trusted as filled, untrusted as x
                trusted_idx = np.where(trusted)[0].tolist()
                untrusted_idx = np.where(~trusted)[0].tolist()

                if trusted_idx:
                    P_t = P_map[trusted_idx]
                    C_t = C_map[trusted_idx]
                    scat_map.set_offsets(P_t)
                    scat_map.set_color([class_to_color(int(c)) for c in C_t])
                    scat_map.set_sizes(np.full((len(trusted_idx),), 60.0))
                else:
                    scat_map.set_offsets(np.zeros((0, 2)))

                if untrusted_idx:
                    P_u = P_map[untrusted_idx]
                    C_u = C_map[untrusted_idx]
                    scat_untrusted.set_offsets(P_u)
                    scat_untrusted.set_color([class_to_color(int(c)) for c in C_u])
                    scat_untrusted.set_alpha(0.35)
                else:
                    scat_untrusted.set_offsets(np.zeros((0, 2)))

                # annotate track ids (only trusted, keep it readable)
                for ti in trusted_idx:
                    tr = tracks[ti]
                    txt_map.append(ax_map.text(float(tr.x), float(tr.y), f"{tr.id}", fontsize=7, alpha=0.8))

                # draw only map triangles from trusted set (already filtered)
                for (i, j, k2) in tris_map:
                    poly = np.array([P_map[i], P_map[j], P_map[k2], P_map[i]], dtype=np.float64)
                    ln, = ax_map.plot(poly[:, 0], poly[:, 1], linewidth=1.0, alpha=0.7)
                    lines_map.append(ln)

            # Panel C signature histogram
            ax_sig.cla()
            ax_sig.set_title("C) Signature counts (top patterns)")
            ax_sig.set_xlabel("count")
            ax_sig.set_ylabel("signature")

            cnt_map: Dict[str, int] = {}
            for s, _tri in sigs_map:
                cnt_map[s] = cnt_map.get(s, 0) + 1
            cnt_obs: Dict[str, int] = {}
            for s, _tri in sigs_obs:
                cnt_obs[s] = cnt_obs.get(s, 0) + 1

            source = cnt_map if cnt_map else cnt_obs
            top = sorted(source.items(), key=lambda kv: kv[1], reverse=True)[:12]
            if top:
                labels = [kv[0] for kv in top]
                vals = [kv[1] for kv in top]
                y_pos = np.arange(len(labels))
                ax_sig.barh(y_pos, vals)
                ax_sig.set_yticks(y_pos)
                ax_sig.set_yticklabels(labels, fontsize=8)
                ax_sig.invert_yaxis()

            hud.set_text(
                f"t={obs.t:.3f}s | pose@t x={x:.2f} y={y:.2f} yaw={math.degrees(yaw):.1f}deg\n"
                f"OBS: N={st_obs['N']} kept_tris={st_obs['kept']} | "
                f"MAP: tracks={len(tracks)} trusted={int(np.count_nonzero(trusted))} kept_tris={st_map['kept']}\n"
                f"track stats: matched={tr_stats.get('matched',0)} spawned={tr_stats.get('spawned',0)} "
                f"pruned={tr_stats.get('pruned',0)} merged={tr_stats.get('merged',0)}\n"
                f"trusted rule: hits>={node.tri_min_hits}, age<={node.tri_max_age:.1f}s, "
                f"deg>={node.tri_min_degree}@{node.tri_deg_r:.1f}m | merge_dist={node.merge_dist:.2f}m"
            )

            fig.canvas.draw_idle()
            plt.pause(0.05)

    except KeyboardInterrupt:
        pass


def main():
    rclpy.init()
    node = Step4p1StructuralViz()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        run_plot(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
