#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SGMH-SLAM MVP (Option 1):
- Keep odom_raw as front-end prediction.
- Maintain a rolling local map of tracks in a "corrected map" frame.
- Build triangles from:
    * obs cones projected into corrected frame (using current correction)
    * trusted map tracks
- Structural matching:
    * triangle signatures (color pattern + quantized edge lengths)
    * estimate SE(2) from triangle pairs (color-consistent correspondences)
    * score transform by inlier count vs map tracks
- Inject correction only when it strongly improves alignment and passes gates.
"""

import math
import random
from dataclasses import dataclass
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
)
from rclpy.time import Time as RclTime

from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from eufs_msgs.msg import ConeArrayWithCovariance, ConeWithCovariance


# -------------------- helpers --------------------

def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat(qx, qy, qz, qw) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def stamp_from_float_seconds(t: float):
    sec = int(math.floor(t))
    nanosec = int((t - sec) * 1e9)
    return rclpy.time.Time(seconds=sec, nanoseconds=nanosec).to_msg()


def class_token(cls_id: int) -> str:
    # EUFS typical: 0 blue, 1 yellow, 2 orange, 3 big_orange
    if cls_id == 0: return "B"
    if cls_id == 1: return "Y"
    if cls_id == 2: return "O"
    if cls_id == 3: return "A"
    return "U"


def tri_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    # area = 0.5 * |cross(b-a, c-a)|
    return 0.5 * abs((b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0]))


def kabsch_se2(P: np.ndarray, Q: np.ndarray):
    """
    Find R,t that minimizes ||R P + t - Q|| in least squares sense (2D).
    P,Q: (N,2) with N>=2
    Returns (R(2,2), t(2,), yaw)
    """
    if P.shape[0] < 2:
        return None

    muP = np.mean(P, axis=0)
    muQ = np.mean(Q, axis=0)
    X = P - muP
    Y = Q - muQ

    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # enforce proper rotation
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T

    t = muQ - (R @ muP)
    yaw = math.atan2(R[1, 0], R[0, 0])
    return R, t, yaw


# -------------------- data --------------------

@dataclass
class Track:
    x: float
    y: float
    cls_id: int
    last_seen: float
    hits: int = 1


# -------------------- node --------------------

class SGMHSLAMMVP(Node):
    def __init__(self):
        super().__init__("sgmh_slam_mvp")

        # Topics
        self.declare_parameter("topics.odom_in", "/slam/odom_raw")
        self.declare_parameter("topics.cones_in", "/perception/cones_fused")
        self.declare_parameter("topics.slam_odom_out", "/slam/odom")
        self.declare_parameter("topics.map_cones_out", "/slam/map_cones")

        # QoS
        self.declare_parameter("qos.best_effort", True)
        self.declare_parameter("qos.depth", 5)

        # Odom buffer
        self.declare_parameter("odom_buffer_sec", 6.0)

        # Drop stale cone frames
        self.declare_parameter("cones.max_age_sec", 0.25)

        # Rolling tracks (in corrected MAP frame)
        self.declare_parameter("tracks.max_age_sec", 6.0)
        self.declare_parameter("tracks.assoc_dist_m", 0.9)
        self.declare_parameter("tracks.ema_alpha", 0.35)
        self.declare_parameter("tracks.min_hits_publish", 2)

        # Track merge (duplicate control)
        self.declare_parameter("tracks.merge_dist_m", 0.7)

        # Trusted tracks for map-structure
        self.declare_parameter("trust.min_hits", 2)
        self.declare_parameter("trust.max_age_sec", 1.0)
        self.declare_parameter("trust.min_degree", 2)
        self.declare_parameter("trust.degree_radius_m", 4.0)

        # Triangles
        self.declare_parameter("tri.k", 4)
        self.declare_parameter("tri.max_edge_m", 6.0)
        self.declare_parameter("tri.min_area_m2", 0.30)
        self.declare_parameter("tri.edge_bin_m", 0.20)
        self.declare_parameter("tri.max_candidates_total", 120)   # cap work/frame

        # Structural correction gates (hard)
        self.declare_parameter("corr.enable", True)
        self.declare_parameter("corr.rate_hz", 5.0)
        self.declare_parameter("corr.inlier_dist_m", 0.9)
        self.declare_parameter("corr.min_inliers", 6)
        self.declare_parameter("corr.min_ratio", 0.45)
        self.declare_parameter("corr.improve_margin", 2)
        self.declare_parameter("corr.max_yaw_deg", 12.0)
        self.declare_parameter("corr.max_trans_m", 2.0)

        gp = self.get_parameter
        self.odom_topic = str(gp("topics.odom_in").value)
        self.cones_topic = str(gp("topics.cones_in").value)
        self.slam_odom_topic = str(gp("topics.slam_odom_out").value)
        self.map_cones_topic = str(gp("topics.map_cones_out").value)

        best_effort = bool(gp("qos.best_effort").value)
        depth = int(gp("qos.depth").value)

        self.odom_buffer_sec = float(gp("odom_buffer_sec").value)
        self.cones_max_age = float(gp("cones.max_age_sec").value)

        self.tr_max_age = float(gp("tracks.max_age_sec").value)
        self.tr_assoc = float(gp("tracks.assoc_dist_m").value)
        self.tr_assoc2 = self.tr_assoc * self.tr_assoc
        self.tr_alpha = float(gp("tracks.ema_alpha").value)
        self.tr_min_hits_pub = int(gp("tracks.min_hits_publish").value)
        self.tr_merge_dist = float(gp("tracks.merge_dist_m").value)
        self.tr_merge2 = self.tr_merge_dist * self.tr_merge_dist

        self.trust_min_hits = int(gp("trust.min_hits").value)
        self.trust_max_age = float(gp("trust.max_age_sec").value)
        self.trust_min_deg = int(gp("trust.min_degree").value)
        self.trust_deg_r = float(gp("trust.degree_radius_m").value)
        self.trust_deg_r2 = self.trust_deg_r * self.trust_deg_r

        self.tri_k = int(gp("tri.k").value)
        self.tri_max_edge = float(gp("tri.max_edge_m").value)
        self.tri_min_area = float(gp("tri.min_area_m2").value)
        self.tri_bin = float(gp("tri.edge_bin_m").value)
        self.tri_max_candidates_total = int(gp("tri.max_candidates_total").value)

        self.corr_enable = bool(gp("corr.enable").value)
        self.corr_rate_hz = float(gp("corr.rate_hz").value)
        self.corr_inlier_dist = float(gp("corr.inlier_dist_m").value)
        self.corr_inlier2 = self.corr_inlier_dist * self.corr_inlier_dist
        self.corr_min_inliers = int(gp("corr.min_inliers").value)
        self.corr_min_ratio = float(gp("corr.min_ratio").value)
        self.corr_improve_margin = int(gp("corr.improve_margin").value)
        self.corr_max_yaw = math.radians(float(gp("corr.max_yaw_deg").value))
        self.corr_max_trans = float(gp("corr.max_trans_m").value)

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=depth,
            reliability=QoSReliabilityPolicy.BEST_EFFORT if best_effort else QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # subs/pubs
        self.create_subscription(Odometry, self.odom_topic, self.cb_odom, qos)
        self.create_subscription(PointCloud2, self.cones_topic, self.cb_cones, qos)

        self.pub_slam_odom = self.create_publisher(Odometry, self.slam_odom_topic, 5)
        self.pub_map_cones = self.create_publisher(ConeArrayWithCovariance, self.map_cones_topic, 5)

        # buffers/state
        self.odom_buf: deque = deque()
        self.tracks: List[Track] = []

        # correction mapping: corrected = corr_R * raw + corr_t
        self.corr_R = np.eye(2, dtype=np.float64)
        self.corr_t = np.zeros((2,), dtype=np.float64)
        self.last_corr_time = -1e9

        self.get_logger().info(
            f"[sgmh_mvp] odom_in={self.odom_topic}, cones_in={self.cones_topic} -> "
            f"slam_odom_out={self.slam_odom_topic}, map_out={self.map_cones_topic}"
        )

    # ------------- odom buffer -------------
    def cb_odom(self, msg: Odometry):
        t = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

        self.odom_buf.append((t, x, y, yaw))
        tmin = t - self.odom_buffer_sec
        while self.odom_buf and self.odom_buf[0][0] < tmin:
            self.odom_buf.popleft()

    def _pose_raw_at(self, t_query: float):
        if not self.odom_buf:
            return None

        times = [it[0] for it in self.odom_buf]
        idx = int(np.searchsorted(times, t_query, side="left"))

        if idx <= 0:
            _, x, y, yaw = self.odom_buf[0]
            return x, y, yaw
        if idx >= len(self.odom_buf):
            _, x, y, yaw = self.odom_buf[-1]
            return x, y, yaw

        t0, x0, y0, yaw0 = self.odom_buf[idx - 1]
        t1, x1, y1, yaw1 = self.odom_buf[idx]
        if t1 == t0:
            return x0, y0, yaw0

        a = (t_query - t0) / (t1 - t0)
        x = x0 + a * (x1 - x0)
        y = y0 + a * (y1 - y0)
        yaw = wrap(yaw0 + a * wrap(yaw1 - yaw0))
        return x, y, yaw

    # ------------- correction application/composition -------------
    def _apply_corr_point(self, x: float, y: float) -> Tuple[float, float]:
        p = np.array([x, y], dtype=np.float64)
        p2 = self.corr_R @ p + self.corr_t
        return float(p2[0]), float(p2[1])

    def _apply_corr_pose(self, x: float, y: float, yaw: float) -> Tuple[float, float, float]:
        x2, y2 = self._apply_corr_point(x, y)
        dyaw = math.atan2(self.corr_R[1, 0], self.corr_R[0, 0])
        return x2, y2, wrap(yaw + dyaw)

    def _compose_corr_and_transform_map(self, R_new: np.ndarray, t_new: np.ndarray):
        """
        corr <- new ∘ corr
        BUT since tracks are stored in corrected frame, and corr defines that frame,
        we must also transform all existing tracks by the same (R_new, t_new) so the map stays consistent.
        """
        # transform map (tracks) into new corrected frame
        for tr in self.tracks:
            p = np.array([tr.x, tr.y], dtype=np.float64)
            p2 = R_new @ p + t_new
            tr.x = float(p2[0])
            tr.y = float(p2[1])

        # compose correction
        self.corr_t = (R_new @ self.corr_t) + t_new
        self.corr_R = R_new @ self.corr_R

    # ------------- ego->map transform -------------
    @staticmethod
    def _obs_ego_to_world(bx: float, by: float, x: float, y: float, yaw: float) -> Tuple[float, float]:
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        wx = x + cy * bx - sy * by
        wy = y + sy * bx + cy * by
        return wx, wy

    # ------------- tracks (rolling local map) -------------
    def _prune_tracks(self, t_now: float):
        self.tracks = [tr for tr in self.tracks if (t_now - tr.last_seen) <= self.tr_max_age]

    def _merge_tracks(self):
        if len(self.tracks) < 2:
            return
        P = np.array([[tr.x, tr.y] for tr in self.tracks], dtype=np.float64)
        C = np.array([int(tr.cls_id) for tr in self.tracks], dtype=np.int32)
        H = np.array([int(tr.hits) for tr in self.tracks], dtype=np.int32)

        alive = np.ones(len(self.tracks), dtype=bool)

        # O(N^2) small map - fine
        for i in range(len(self.tracks)):
            if not alive[i]:
                continue
            for j in range(i + 1, len(self.tracks)):
                if not alive[j]:
                    continue
                if C[i] != C[j]:
                    continue
                d2 = float(np.sum((P[i] - P[j]) ** 2))
                if d2 > self.tr_merge2:
                    continue

                # keep higher-hits as anchor
                if H[i] >= H[j]:
                    keep, kill = i, j
                else:
                    keep, kill = j, i

                trk = self.tracks[keep]
                trd = self.tracks[kill]
                wk = float(trk.hits)
                wd = float(trd.hits)
                w = max(wk + wd, 1.0)

                trk.x = (wk * trk.x + wd * trd.x) / w
                trk.y = (wk * trk.y + wd * trd.y) / w
                trk.hits = int(wk + wd)
                trk.last_seen = max(trk.last_seen, trd.last_seen)

                alive[kill] = False

        self.tracks = [tr for k, tr in enumerate(self.tracks) if alive[k]]

    def _update_tracks(self, t_now: float, obs_map: List[Tuple[float, float, int]]):
        """
        Greedy 1:1 NN association per observation (same class).
        Assumes obs_map already in corrected map frame.
        """
        self._prune_tracks(t_now)

        used = set()
        a = self.tr_alpha
        for mx, my, cls_id in obs_map:
            best_i = None
            best_d2 = self.tr_assoc2

            for i, tr in enumerate(self.tracks):
                if i in used:
                    continue
                if tr.cls_id != cls_id:
                    continue
                dx = tr.x - mx
                dy = tr.y - my
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_d2 = d2
                    best_i = i

            if best_i is not None:
                tr = self.tracks[best_i]
                tr.x = (1.0 - a) * tr.x + a * mx
                tr.y = (1.0 - a) * tr.y + a * my
                tr.last_seen = t_now
                tr.hits += 1
                used.add(best_i)
            else:
                self.tracks.append(Track(x=mx, y=my, cls_id=int(cls_id), last_seen=t_now, hits=1))

        self._merge_tracks()

    def _trusted_mask(self, t_now: float) -> np.ndarray:
        """
        TRUSTED = hits>=min_hits AND age<=max_age AND degree>=min_degree within radius
        """
        n = len(self.tracks)
        if n == 0:
            return np.zeros((0,), dtype=bool)

        hits = np.array([tr.hits for tr in self.tracks], dtype=np.int32)
        age = np.array([max(0.0, t_now - tr.last_seen) for tr in self.tracks], dtype=np.float64)
        base = (hits >= self.trust_min_hits) & (age <= self.trust_max_age)

        if np.count_nonzero(base) < 3:
            return base

        P = np.array([[tr.x, tr.y] for tr in self.tracks], dtype=np.float64)
        D2 = np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=2)
        deg = np.sum(D2 <= self.trust_deg_r2, axis=1) - 1
        return base & (deg >= self.trust_min_deg)

    # ------------- triangles + signatures -------------
    def _build_triangles(self, P: np.ndarray, C: np.ndarray, k: int):
        """
        Build triangles using kNN graph:
        - for each point i, connect to k nearest neighbors
        - triangles are (i,j,k) from combinations of neighbors
        Filter by max edge and min area.
        """
        N = P.shape[0]
        if N < 3:
            return []

        k = max(1, min(k, N - 1))

        # pairwise distances
        D2 = np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=2)
        nn = []
        for i in range(N):
            order = np.argsort(D2[i])
            nn.append([int(j) for j in order[1:1 + k]])

        tris = set()
        for i in range(N):
            neigh = nn[i]
            if len(neigh) < 2:
                continue
            for a in range(len(neigh)):
                for b in range(a + 1, len(neigh)):
                    j = neigh[a]
                    k2 = neigh[b]
                    tris.add(tuple(sorted((i, j, k2))))

        out = []
        for i, j, k2 in tris:
            a = P[i]; b = P[j]; c = P[k2]
            dij = float(np.linalg.norm(a - b))
            dik = float(np.linalg.norm(a - c))
            djk = float(np.linalg.norm(b - c))
            if max(dij, dik, djk) > self.tri_max_edge:
                continue
            area = tri_area(a, b, c)
            if area < self.tri_min_area:
                continue
            out.append((i, j, k2))
        return out

    def _tri_signature(self, P: np.ndarray, C: np.ndarray, tri: Tuple[int, int, int]) -> str:
        i, j, k = tri
        a = P[i]; b = P[j]; c = P[k]
        edges = sorted([
            float(np.linalg.norm(a - b)),
            float(np.linalg.norm(a - c)),
            float(np.linalg.norm(b - c)),
        ])
        bsz = max(self.tri_bin, 1e-6)
        q = tuple(int(round(e / bsz)) for e in edges)

        cols = sorted([class_token(int(C[i])), class_token(int(C[j])), class_token(int(C[k]))])
        col_sig = "".join(cols)
        return f"{col_sig}|{q[0]}-{q[1]}-{q[2]}"

    # ------------- scoring / association -------------
    def _count_inliers(self, obs_pts: np.ndarray, obs_cls: np.ndarray,
                       map_pts: np.ndarray, map_cls: np.ndarray) -> int:
        """
        Count class-consistent NN inliers: for each obs point, if any map point of same class
        within inlier_dist -> inlier.
        """
        inl = 0
        for (ox, oy), cls in zip(obs_pts, obs_cls):
            # small map; brute force ok
            best2 = None
            for (mx, my), mcls in zip(map_pts, map_cls):
                if int(mcls) != int(cls):
                    continue
                dx = mx - ox
                dy = my - oy
                d2 = dx*dx + dy*dy
                if best2 is None or d2 < best2:
                    best2 = d2
            if best2 is not None and best2 <= self.corr_inlier2:
                inl += 1
        return inl

    def _color_consistent_correspondences(self, obs_tri_idx, map_tri_idx, C_obs, C_map):
        """
        Generate color-consistent vertex index mappings between two triangles.
        Handles repeated colors (e.g., BYY).
        Returns list of pairs (obs_order[3], map_order[3]) where colors match position-wise.
        """
        oi, oj, ok = obs_tri_idx
        mi, mj, mk = map_tri_idx

        obs_vs = [oi, oj, ok]
        map_vs = [mi, mj, mk]

        obs_cols = [int(C_obs[v]) for v in obs_vs]
        map_cols = [int(C_map[v]) for v in map_vs]

        # all permutations of map vertices
        perms = [
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        ]

        out = []
        for p in perms:
            if obs_cols[0] == map_cols[p[0]] and obs_cols[1] == map_cols[p[1]] and obs_cols[2] == map_cols[p[2]]:
                out.append((obs_vs, [map_vs[p[0]], map_vs[p[1]], map_vs[p[2]]]))
        return out

    def _best_transform_from_tri_pair(self, P_obs, C_obs, tri_obs, P_map, C_map, tri_map):
        """
        For a triangle-pair, try all color-consistent correspondences and return best (R,t,yaw, rms).
        """
        corrs = self._color_consistent_correspondences(tri_obs, tri_map, C_obs, C_map)
        if not corrs:
            return None

        best = None
        for obs_order, map_order in corrs:
            P = np.array([P_obs[i] for i in obs_order], dtype=np.float64)
            Q = np.array([P_map[i] for i in map_order], dtype=np.float64)

            est = kabsch_se2(P, Q)
            if est is None:
                continue
            R, t, yaw = est

            # quick residual on the triangle itself
            P2 = (P @ R.T) + t[None, :]
            rms = float(np.sqrt(np.mean(np.sum((P2 - Q) ** 2, axis=1))))

            if best is None or rms < best[-1]:
                best = (R, t, yaw, rms)

        return best

    def _try_structural_correction(self, t_now: float, obs_pts: np.ndarray, obs_cls: np.ndarray):
        if not self.corr_enable:
            return False

        if (t_now - self.last_corr_time) < (1.0 / max(self.corr_rate_hz, 1e-6)):
            return False

        if len(self.tracks) < 8 or obs_pts.shape[0] < 5:
            return False

        # map points (trusted only)
        trusted = self._trusted_mask(t_now)
        idx_map = np.where(trusted)[0].tolist()
        if len(idx_map) < 6:
            return False

        P_map = np.array([[self.tracks[i].x, self.tracks[i].y] for i in idx_map], dtype=np.float64)
        C_map = np.array([int(self.tracks[i].cls_id) for i in idx_map], dtype=np.int32)

        # baseline alignment score (current corr already applied in obs_pts)
        base_inl = self._count_inliers(obs_pts, obs_cls, P_map, C_map)

        # build triangles
        obs_tris = self._build_triangles(obs_pts, obs_cls, self.tri_k)
        map_tris = self._build_triangles(P_map, C_map, self.tri_k)

        if not obs_tris or not map_tris:
            return False

        # signature -> triangles
        map_sig: Dict[str, List[Tuple[int, int, int]]] = defaultdict(list)
        for tri in map_tris:
            s = self._tri_signature(P_map, C_map, tri)
            map_sig[s].append(tri)

        # candidate pairs (signature match + small bin neighborhood)
        def neighbor_keys(sig: str):
            # expand ±1 bins to be tolerant to small scale noise
            try:
                col, rest = sig.split("|")
                a, b, c = rest.split("-")
                qa, qb, qc = int(a), int(b), int(c)
            except Exception:
                return [sig]
            keys = []
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        keys.append(f"{col}|{qa+da}-{qb+db}-{qc+dc}")
            return keys

        candidates = []
        for tri_o in obs_tris:
            sig_o = self._tri_signature(obs_pts, obs_cls, tri_o)
            for key in neighbor_keys(sig_o):
                for tri_m in map_sig.get(key, []):
                    candidates.append((tri_o, tri_m))
                    if len(candidates) >= self.tri_max_candidates_total:
                        break
                if len(candidates) >= self.tri_max_candidates_total:
                    break
            if len(candidates) >= self.tri_max_candidates_total:
                break

        if not candidates:
            return False

        # Evaluate candidates
        best = None  # (inliers, ratio, rms, R, t, yaw)
        Nobs = int(obs_pts.shape[0])

        for tri_o, tri_m in candidates:
            est = self._best_transform_from_tri_pair(obs_pts, obs_cls, tri_o, P_map, C_map, tri_m)
            if est is None:
                continue
            R, t, yaw, rms = est

            # hard gates early
            if abs(yaw) > self.corr_max_yaw:
                continue
            if float(np.linalg.norm(t)) > self.corr_max_trans:
                continue

            # score transform by inliers on *all obs points*
            obs_tf = (obs_pts @ R.T) + t[None, :]
            inl = self._count_inliers(obs_tf, obs_cls, P_map, C_map)
            ratio = inl / max(Nobs, 1)

            if best is None or inl > best[0] or (inl == best[0] and rms < best[2]):
                best = (inl, ratio, rms, R, t, yaw)

        if best is None:
            return False

        inl, ratio, rms, R_best, t_best, yaw_best = best

        # acceptance gates
        if inl < self.corr_min_inliers:
            return False
        if ratio < self.corr_min_ratio:
            return False
        if inl < (base_inl + self.corr_improve_margin):
            return False

        # apply correction: corr <- (R_best, t_best) ∘ corr, and map tracks transformed too
        self._compose_corr_and_transform_map(R_best, t_best)
        self.last_corr_time = t_now

        self.get_logger().info(
            f"[corr] accept: base={base_inl} -> inl={inl}/{Nobs} ({ratio:.2f}) "
            f"rms={rms:.2f} dYaw={math.degrees(yaw_best):.2f}deg |t|={float(np.linalg.norm(t_best)):.2f}m",
            throttle_duration_sec=0.3
        )
        return True

    # ------------- cones callback -------------
    def cb_cones(self, msg: PointCloud2):
        t_c = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        t_now = self.get_clock().now().nanoseconds * 1e-9

        # stale protection
        if (t_now - t_c) > self.cones_max_age:
            return

        pose_raw = self._pose_raw_at(t_c)
        if pose_raw is None:
            return

        # parse ego cones (bx,by,cls_id)
        try:
            meas = [(float(bx), float(by), int(cid))
                    for (bx, by, _bz, cid) in pc2.read_points(
                        msg, field_names=("x", "y", "z", "class_id"), skip_nans=True)]
        except Exception as e:
            self.get_logger().warn(f"[sgmh_mvp] cone parse error: {e}")
            return

        # if no cones: still prune and publish map
        if not meas:
            self._prune_tracks(t_c)
            x_raw, y_raw, yaw_raw = pose_raw
            x_pub, y_pub, yaw_pub = self._apply_corr_pose(x_raw, y_raw, yaw_raw)
            self._publish_slam_odom(t_c, x_pub, y_pub, yaw_pub)
            self._publish_map_cones(t_c)
            return

        # corrected pose (current corr)
        x_raw, y_raw, yaw_raw = pose_raw
        x_map, y_map, yaw_map = self._apply_corr_pose(x_raw, y_raw, yaw_raw)

        # obs points in corrected map frame
        obs_map = []
        for bx, by, cls_id in meas:
            wx, wy = self._obs_ego_to_world(bx, by, x_map, y_map, yaw_map)
            obs_map.append((wx, wy, int(cls_id)))

        obs_pts = np.array([[x, y] for (x, y, _c) in obs_map], dtype=np.float64)
        obs_cls = np.array([int(c) for (_x, _y, c) in obs_map], dtype=np.int32)

        # try structural correction (uses obs_pts, and current map tracks)
        corrected = self._try_structural_correction(t_c, obs_pts, obs_cls)

        if corrected:
            # recompute corrected pose under updated corr (corr changed)
            x_map, y_map, yaw_map = self._apply_corr_pose(x_raw, y_raw, yaw_raw)

            # also move obs_pts into new corrected frame:
            # NOTE: _compose_corr_and_transform_map already redefined corrected frame,
            # but our obs_pts were in the *old* corrected frame.
            # easiest: rebuild obs_map from ego using new pose.
            obs_map = []
            for bx, by, cls_id in meas:
                wx, wy = self._obs_ego_to_world(bx, by, x_map, y_map, yaw_map)
                obs_map.append((wx, wy, int(cls_id)))

        # update tracks with final obs_map in final corrected frame
        self._update_tracks(t_c, obs_map)

        # publish corrected odom + map
        self._publish_slam_odom(t_c, x_map, y_map, yaw_map)
        self._publish_map_cones(t_c)

    # ------------- publishers -------------
    def _publish_slam_odom(self, t: float, x: float, y: float, yaw: float):
        od = Odometry()
        od.header.stamp = stamp_from_float_seconds(t)
        od.header.frame_id = "map"
        od.child_frame_id = "base_link"

        od.pose.pose.position.x = float(x)
        od.pose.pose.position.y = float(y)
        od.pose.pose.position.z = 0.0

        half = 0.5 * float(yaw)
        od.pose.pose.orientation.x = 0.0
        od.pose.pose.orientation.y = 0.0
        od.pose.pose.orientation.z = math.sin(half)
        od.pose.pose.orientation.w = math.cos(half)

        self.pub_slam_odom.publish(od)

    def _publish_map_cones(self, t: float):
        msg = ConeArrayWithCovariance()
        msg.header.stamp = stamp_from_float_seconds(t)
        msg.header.frame_id = "map"

        for tr in self.tracks:
            if tr.hits < self.tr_min_hits_pub:
                continue
            cone = ConeWithCovariance()
            cone.point.x = float(tr.x)
            cone.point.y = float(tr.y)
            cone.point.z = 0.0

            if tr.cls_id == 0:
                msg.blue_cones.append(cone)
            elif tr.cls_id == 1:
                msg.yellow_cones.append(cone)
            elif tr.cls_id == 2:
                msg.orange_cones.append(cone)
            elif tr.cls_id == 3:
                msg.big_orange_cones.append(cone)
            else:
                try:
                    msg.unknown_color_cones.append(cone)
                except AttributeError:
                    pass

        self.pub_map_cones.publish(msg)


def main():
    rclpy.init()
    node = SGMHSLAMMVP()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
