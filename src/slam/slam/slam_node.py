#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SGMH-SLAM Full Pipeline (ROS2, Python)

Front-end:
  - Subscribe /slam/odom_raw
  - Publish /slam/odom at odom rate (corrected using best hypothesis)

Back-end (async timer):
  - Subscribe /perception/cones_fused (PointCloud2 x,y,z,class_id)
  - Build structural triangles for obs + map
  - Multi-hypothesis tracking:
      - For each hypothesis: propose candidate SE(2) corrections via triangle matching
      - Branch top B children + keep parent (no-update branch)
      - Score/prune to top K
  - Update shared "best hypothesis correction" used by front-end.

Map:
  - Rolling local tracks per hypothesis in corrected frame
  - NN association + EMA update + prune + merge
  - Trusted track filtering for structural matching

Frame handling:
  cones.frame_mode:
    - "ego": assumes PointCloud2 x,y are ego coordinates (base_link) relative to car
    - "velodyne": assumes PointCloud2 points in velodyne frame; applies fixed extrinsics to convert to ego

Outputs:
  - /slam/odom : Odometry in "map"
  - /slam/map_cones : map cones from best hypothesis tracks (in "map")
"""

import math
import threading
from dataclasses import dataclass
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
)
from rclpy.time import Time as RclTime

from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from eufs_msgs.msg import ConeArrayWithCovariance, ConeWithCovariance


# ---------------------- math helpers ----------------------

def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat(qx, qy, qz, qw) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def stamp_from_float_seconds(t: float):
    sec = int(math.floor(t))
    nanosec = int((t - sec) * 1e9)
    return rclpy.time.Time(seconds=sec, nanoseconds=nanosec).to_msg()


def class_token(cls_id: int) -> str:
    # 0 blue, 1 yellow, 2 orange, 3 big_orange
    if cls_id == 0: return "B"
    if cls_id == 1: return "Y"
    if cls_id == 2: return "O"
    if cls_id == 3: return "A"
    return "U"


def tri_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))


def kabsch_se2(P: np.ndarray, Q: np.ndarray):
    """
    Find R,t that minimize || R P + t - Q ||, P,Q: (N,2) N>=2
    Returns (R(2,2), t(2,), yaw)
    """
    if P.shape[0] < 2:
        return None

    muP = np.mean(P, axis=0)
    muQ = np.mean(Q, axis=0)
    X = P - muP
    Y = Q - muQ

    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T

    t = muQ - (R @ muP)
    yaw = math.atan2(R[1, 0], R[0, 0])
    return R, t, yaw


# ---------------------- data ----------------------

@dataclass
class Track:
    x: float
    y: float
    cls_id: int
    last_seen: float
    hits: int = 1


@dataclass
class Hypothesis:
    # correction: corrected = R * raw + t
    R: np.ndarray
    t: np.ndarray

    tracks: List[Track]
    score: float = 0.0
    last_update_t: float = -1e9

    # bookkeeping for pruning
    inliers_last: int = 0
    ratio_last: float = 0.0
    rms_last: float = 0.0
    dyaw_last: float = 0.0
    trans_last: float = 0.0


# ---------------------- main node ----------------------

class SGMHSLAMFull(Node):
    def __init__(self):
        super().__init__("sgmh_slam_full")

        # Topics
        self.declare_parameter("topics.odom_in", "/slam/odom_raw")
        self.declare_parameter("topics.cones_in", "/perception/cones_fused")
        self.declare_parameter("topics.slam_odom_out", "/slam/odom")
        self.declare_parameter("topics.map_cones_out", "/slam/map_cones")

        # QoS
        self.declare_parameter("qos.best_effort", True)
        self.declare_parameter("qos.depth", 5)

        # Time/Buffer
        self.declare_parameter("odom_buffer_sec", 8.0)
        self.declare_parameter("cones.max_age_sec", 0.30)

        # Cones frame handling
        self.declare_parameter("cones.frame_mode", "velodyne")  # "ego" or "velodyne"
        # velodyne->base_link fixed extrinsic (2D)
        self.declare_parameter("cones.extrinsic.tx", 0.0)
        self.declare_parameter("cones.extrinsic.ty", 0.0)
        self.declare_parameter("cones.extrinsic.yaw_deg", 0.0)

        # Tracks per hypothesis
        self.declare_parameter("tracks.max_age_sec", 6.0)
        self.declare_parameter("tracks.assoc_dist_m", 0.9)
        self.declare_parameter("tracks.ema_alpha", 0.35)
        self.declare_parameter("tracks.min_hits_publish", 2)
        self.declare_parameter("tracks.merge_dist_m", 0.7)

        # Trusted filtering (for structural matching)
        self.declare_parameter("trust.min_hits", 2)
        self.declare_parameter("trust.max_age_sec", 1.0)
        self.declare_parameter("trust.min_degree", 2)
        self.declare_parameter("trust.degree_radius_m", 4.0)

        # Triangles
        self.declare_parameter("tri.k", 4)
        self.declare_parameter("tri.max_edge_m", 6.0)
        self.declare_parameter("tri.min_area_m2", 0.30)
        self.declare_parameter("tri.edge_bin_m", 0.20)
        self.declare_parameter("tri.max_pairs", 140)

        # Multi-hypothesis
        self.declare_parameter("mh.enable", True)
        self.declare_parameter("mh.K", 5)        # keep top K
        self.declare_parameter("mh.B", 3)        # branch top B per hypothesis per backend tick
        self.declare_parameter("mh.backend_rate_hz", 7.0)

        # Scoring / acceptance
        self.declare_parameter("corr.inlier_dist_m", 0.9)
        self.declare_parameter("corr.min_inliers", 6)
        self.declare_parameter("corr.min_ratio", 0.40)
        self.declare_parameter("corr.max_yaw_deg", 12.0)
        self.declare_parameter("corr.max_trans_m", 2.0)

        # Require improvement vs "no correction"
        self.declare_parameter("corr.improve_margin", 2)

        # Soft weights for scoring
        self.declare_parameter("score.w_inliers", 1.0)
        self.declare_parameter("score.w_ratio", 3.0)
        self.declare_parameter("score.w_rms", 2.0)
        self.declare_parameter("score.w_penalty", 2.5)

        gp = self.get_parameter

        self.odom_topic = str(gp("topics.odom_in").value)
        self.cones_topic = str(gp("topics.cones_in").value)
        self.slam_odom_topic = str(gp("topics.slam_odom_out").value)
        self.map_cones_topic = str(gp("topics.map_cones_out").value)

        best_effort = bool(gp("qos.best_effort").value)
        depth = int(gp("qos.depth").value)

        self.odom_buffer_sec = float(gp("odom_buffer_sec").value)
        self.cones_max_age = float(gp("cones.max_age_sec").value)

        self.cones_frame_mode = str(gp("cones.frame_mode").value).lower().strip()
        tx = float(gp("cones.extrinsic.tx").value)
        ty = float(gp("cones.extrinsic.ty").value)
        yaw_ex = math.radians(float(gp("cones.extrinsic.yaw_deg").value))
        c, s = math.cos(yaw_ex), math.sin(yaw_ex)
        self.R_v2b = np.array([[c, -s], [s, c]], dtype=np.float64)
        self.t_v2b = np.array([tx, ty], dtype=np.float64)

        self.tr_max_age = float(gp("tracks.max_age_sec").value)
        self.tr_assoc = float(gp("tracks.assoc_dist_m").value)
        self.tr_assoc2 = self.tr_assoc * self.tr_assoc
        self.tr_alpha = float(gp("tracks.ema_alpha").value)
        self.tr_min_hits_pub = int(gp("tracks.min_hits_publish").value)
        self.tr_merge = float(gp("tracks.merge_dist_m").value)
        self.tr_merge2 = self.tr_merge * self.tr_merge

        self.trust_min_hits = int(gp("trust.min_hits").value)
        self.trust_max_age = float(gp("trust.max_age_sec").value)
        self.trust_min_deg = int(gp("trust.min_degree").value)
        self.trust_deg_r = float(gp("trust.degree_radius_m").value)
        self.trust_deg_r2 = self.trust_deg_r * self.trust_deg_r

        self.tri_k = int(gp("tri.k").value)
        self.tri_max_edge = float(gp("tri.max_edge_m").value)
        self.tri_min_area = float(gp("tri.min_area_m2").value)
        self.tri_bin = float(gp("tri.edge_bin_m").value)
        self.tri_max_pairs = int(gp("tri.max_pairs").value)

        self.mh_enable = bool(gp("mh.enable").value)
        self.mh_K = int(gp("mh.K").value)
        self.mh_B = int(gp("mh.B").value)
        self.backend_rate = float(gp("mh.backend_rate_hz").value)

        self.inlier_dist = float(gp("corr.inlier_dist_m").value)
        self.inlier2 = self.inlier_dist * self.inlier_dist
        self.min_inliers = int(gp("corr.min_inliers").value)
        self.min_ratio = float(gp("corr.min_ratio").value)
        self.max_yaw = math.radians(float(gp("corr.max_yaw_deg").value))
        self.max_trans = float(gp("corr.max_trans_m").value)
        self.improve_margin = int(gp("corr.improve_margin").value)

        self.w_inl = float(gp("score.w_inliers").value)
        self.w_ratio = float(gp("score.w_ratio").value)
        self.w_rms = float(gp("score.w_rms").value)
        self.w_pen = float(gp("score.w_penalty").value)

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=depth,
            reliability=QoSReliabilityPolicy.BEST_EFFORT if best_effort else QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # subscribers
        self.create_subscription(Odometry, self.odom_topic, self.cb_odom, qos)
        self.create_subscription(PointCloud2, self.cones_topic, self.cb_cones, qos)

        # publishers
        self.pub_slam_odom = self.create_publisher(Odometry, self.slam_odom_topic, 10)
        self.pub_map = self.create_publisher(ConeArrayWithCovariance, self.map_cones_topic, 5)

        # buffers
        self.odom_buf = deque()  # (t, x, y, yaw, full_msg)
        self.latest_cones = None  # (t_c, meas_list[(x,y,cls)], header)
        self.latest_cones_wall = -1e9

        self.lock = threading.Lock()

        # hypotheses init: identity correction, empty tracks
        h0 = Hypothesis(R=np.eye(2, dtype=np.float64), t=np.zeros((2,), dtype=np.float64), tracks=[], score=0.0)
        self.hyps: List[Hypothesis] = [h0]
        self.best_idx: int = 0

        # backend timer
        period = 1.0 / max(self.backend_rate, 1e-6)
        self.create_timer(period, self.backend_tick)

        self.get_logger().info(
            f"[sgmh_full] odom_in={self.odom_topic}, cones_in={self.cones_topic}, "
            f"odom_out={self.slam_odom_topic}, map_out={self.map_cones_topic} | "
            f"frame_mode={self.cones_frame_mode} K={self.mh_K} B={self.mh_B} backend={self.backend_rate}Hz"
        )

    # ------------------ odom buffering + publishing ------------------

    def cb_odom(self, msg: Odometry):
        t = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

        with self.lock:
            self.odom_buf.append((t, x, y, yaw, msg))
            tmin = t - self.odom_buffer_sec
            while self.odom_buf and self.odom_buf[0][0] < tmin:
                self.odom_buf.popleft()

            # publish corrected at odom rate (front-end)
            hyp = self.hyps[self.best_idx]
            x2, y2 = self._apply_corr_point(hyp.R, hyp.t, x, y)
            dyaw = math.atan2(hyp.R[1, 0], hyp.R[0, 0])
            yaw2 = wrap(yaw + dyaw)

        self._publish_odom(t, x2, y2, yaw2)

    def _publish_odom(self, t: float, x: float, y: float, yaw: float):
        od = Odometry()
        od.header.stamp = stamp_from_float_seconds(t)
        od.header.frame_id = "map"
        od.child_frame_id = "base_link"
        od.pose.pose.position.x = float(x)
        od.pose.pose.position.y = float(y)
        od.pose.pose.position.z = 0.0
        half = 0.5 * float(yaw)
        od.pose.pose.orientation.z = math.sin(half)
        od.pose.pose.orientation.w = math.cos(half)
        self.pub_slam_odom.publish(od)

    # ------------------ cones capture ------------------

    def cb_cones(self, msg: PointCloud2):
        t_c = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        t_now = self.get_clock().now().nanoseconds * 1e-9

        # stale drop (sim time safe if use_sim_time=True)
        if (t_now - t_c) > self.cones_max_age:
            return

        # parse points: (x,y,class_id)
        meas = []
        try:
            for x, y, _z, cid in pc2.read_points(msg, field_names=("x", "y", "z", "class_id"), skip_nans=True):
                meas.append((float(x), float(y), int(cid)))
        except Exception as e:
            self.get_logger().warn(f"[sgmh_full] cones parse error: {e}")
            return

        with self.lock:
            self.latest_cones = (t_c, meas, msg.header)
            self.latest_cones_wall = t_now

    # ------------------ backend tick (structural matching + MH) ------------------

    def backend_tick(self):
        with self.lock:
            if self.latest_cones is None:
                return
            t_c, meas, header = self.latest_cones

            # snapshot odom buffer for pose lookup
            odom_snapshot = list(self.odom_buf)

        if not odom_snapshot:
            return

        # get raw pose at cone time
        pose = self._interp_pose(odom_snapshot, t_c)
        if pose is None:
            return
        x_raw, y_raw, yaw_raw = pose

        # if no cones, still publish map from best hyp
        if not meas:
            with self.lock:
                best = self.hyps[self.best_idx]
                tracks_best = best.tracks[:]  # shallow copy ok for publish
            self._publish_map(t_c, tracks_best)
            return

        # run MH update
        new_hyps: List[Hypothesis] = []

        # precompute ego obs (bx,by) in base_link; handle velodyne if needed
        obs_ego = self._cones_to_ego(meas)

        # We also compute a baseline "no extra correction" inlier score per hyp for improve checks.
        for h in self.hyps:
            # project obs into corrected map frame for this hypothesis
            obs_map_pts, obs_cls = self._obs_ego_to_map_points(obs_ego, x_raw, y_raw, yaw_raw, h.R, h.t)

            # update tracks even without correction (front-end map growth)
            self._update_tracks(h, t_c, obs_map_pts, obs_cls)

            # always keep parent hypothesis (no-update branch)
            base_inl, base_ratio, _ = self._score_inliers(h, obs_map_pts, obs_cls, t_c)
            h.inliers_last = base_inl
            h.ratio_last = base_ratio
            new_hyps.append(h)

            # propose correction branches
            if self.mh_enable:
                children = self._propose_children(h, t_c, obs_map_pts, obs_cls)
                new_hyps.extend(children)

        # prune to top K by score
        new_hyps = self._prune_hypotheses(new_hyps)

        with self.lock:
            self.hyps = new_hyps
            self.best_idx = int(np.argmax([h.score for h in self.hyps]))

            best = self.hyps[self.best_idx]
            tracks_best = best.tracks[:]

        # publish best map cones
        self._publish_map(t_c, tracks_best)

    # ------------------ frame conversion ------------------

    def _cones_to_ego(self, meas_xyc: List[Tuple[float, float, int]]):
        """
        Returns list of (bx, by, cls) in base_link ego frame.
        - if cones.frame_mode == "ego": assume x,y already ego
        - if "velodyne": apply fixed velodyne->base_link 2D transform
        """
        out = []
        if self.cones_frame_mode == "ego":
            for x, y, c in meas_xyc:
                out.append((x, y, c))
            return out

        # velodyne -> base_link
        for vx, vy, c in meas_xyc:
            p = np.array([vx, vy], dtype=np.float64)
            pb = self.R_v2b @ p + self.t_v2b
            out.append((float(pb[0]), float(pb[1]), int(c)))
        return out

    # ------------------ odom interpolation ------------------

    def _interp_pose(self, odom_list, t_query: float):
        # odom_list: [(t,x,y,yaw,msg), ...] in time order
        if not odom_list:
            return None
        times = [it[0] for it in odom_list]
        idx = int(np.searchsorted(times, t_query, side="left"))

        if idx <= 0:
            _, x, y, yaw, _ = odom_list[0]
            return x, y, yaw
        if idx >= len(odom_list):
            _, x, y, yaw, _ = odom_list[-1]
            return x, y, yaw

        t0, x0, y0, yaw0, _ = odom_list[idx - 1]
        t1, x1, y1, yaw1, _ = odom_list[idx]
        if t1 == t0:
            return x0, y0, yaw0

        a = (t_query - t0) / (t1 - t0)
        x = x0 + a * (x1 - x0)
        y = y0 + a * (y1 - y0)
        yaw = wrap(yaw0 + a * wrap(yaw1 - yaw0))
        return x, y, yaw

    # ------------------ correction application ------------------

    @staticmethod
    def _apply_corr_point(R: np.ndarray, t: np.ndarray, x: float, y: float):
        p = np.array([x, y], dtype=np.float64)
        p2 = R @ p + t
        return float(p2[0]), float(p2[1])

    # ------------------ projection: ego obs -> corrected map points ------------------

    def _obs_ego_to_map_points(self, obs_ego, x_raw, y_raw, yaw_raw, Rcorr, tcorr):
        # raw pose -> corrected pose
        x_map, y_map = self._apply_corr_point(Rcorr, tcorr, x_raw, y_raw)
        dyaw = math.atan2(Rcorr[1, 0], Rcorr[0, 0])
        yaw_map = wrap(yaw_raw + dyaw)

        cy = math.cos(yaw_map)
        sy = math.sin(yaw_map)

        pts = []
        cls = []
        for bx, by, cid in obs_ego:
            mx = x_map + cy * bx - sy * by
            my = y_map + sy * bx + cy * by
            pts.append((mx, my))
            cls.append(int(cid))
        return np.asarray(pts, dtype=np.float64), np.asarray(cls, dtype=np.int32)

    # ------------------ tracks per hypothesis ------------------

    def _prune_tracks(self, h: Hypothesis, t_now: float):
        h.tracks = [tr for tr in h.tracks if (t_now - tr.last_seen) <= self.tr_max_age]

    def _merge_tracks(self, h: Hypothesis):
        if len(h.tracks) < 2:
            return
        alive = [True] * len(h.tracks)
        P = np.array([[tr.x, tr.y] for tr in h.tracks], dtype=np.float64)
        C = np.array([tr.cls_id for tr in h.tracks], dtype=np.int32)
        H = np.array([tr.hits for tr in h.tracks], dtype=np.int32)

        for i in range(len(h.tracks)):
            if not alive[i]:
                continue
            for j in range(i + 1, len(h.tracks)):
                if not alive[j]:
                    continue
                if C[i] != C[j]:
                    continue
                d2 = float(np.sum((P[i] - P[j]) ** 2))
                if d2 > self.tr_merge2:
                    continue

                keep, kill = (i, j) if H[i] >= H[j] else (j, i)
                trk = h.tracks[keep]
                trd = h.tracks[kill]

                wk, wd = float(trk.hits), float(trd.hits)
                w = max(wk + wd, 1.0)
                trk.x = (wk * trk.x + wd * trd.x) / w
                trk.y = (wk * trk.y + wd * trd.y) / w
                trk.hits = int(wk + wd)
                trk.last_seen = max(trk.last_seen, trd.last_seen)

                alive[kill] = False

        h.tracks = [tr for i, tr in enumerate(h.tracks) if alive[i]]

    def _update_tracks(self, h: Hypothesis, t_now: float, obs_pts: np.ndarray, obs_cls: np.ndarray):
        self._prune_tracks(h, t_now)

        used = set()
        a = self.tr_alpha

        for (mx, my), cid in zip(obs_pts, obs_cls):
            best_i = None
            best_d2 = self.tr_assoc2
            for i, tr in enumerate(h.tracks):
                if i in used:
                    continue
                if tr.cls_id != int(cid):
                    continue
                dx = tr.x - mx
                dy = tr.y - my
                d2 = dx*dx + dy*dy
                if d2 < best_d2:
                    best_d2 = d2
                    best_i = i

            if best_i is not None:
                tr = h.tracks[best_i]
                tr.x = (1.0 - a) * tr.x + a * mx
                tr.y = (1.0 - a) * tr.y + a * my
                tr.last_seen = t_now
                tr.hits += 1
                used.add(best_i)
            else:
                h.tracks.append(Track(x=float(mx), y=float(my), cls_id=int(cid), last_seen=t_now, hits=1))

        self._merge_tracks(h)

    def _trusted_mask(self, h: Hypothesis, t_now: float) -> np.ndarray:
        n = len(h.tracks)
        if n == 0:
            return np.zeros((0,), dtype=bool)

        hits = np.array([tr.hits for tr in h.tracks], dtype=np.int32)
        age = np.array([max(0.0, t_now - tr.last_seen) for tr in h.tracks], dtype=np.float64)
        base = (hits >= self.trust_min_hits) & (age <= self.trust_max_age)

        if np.count_nonzero(base) < 3:
            return base

        P = np.array([[tr.x, tr.y] for tr in h.tracks], dtype=np.float64)
        D2 = np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=2)
        deg = np.sum(D2 <= self.trust_deg_r2, axis=1) - 1
        return base & (deg >= self.trust_min_deg)

    # ------------------ triangles ------------------

    def _build_triangles(self, P: np.ndarray, C: np.ndarray, k: int):
        N = P.shape[0]
        if N < 3:
            return []

        k = max(1, min(k, N - 1))
        D2 = np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=2)

        nn = []
        for i in range(N):
            order = np.argsort(D2[i])
            nn.append([int(j) for j in order[1:1+k]])

        tris = set()
        for i in range(N):
            neigh = nn[i]
            if len(neigh) < 2:
                continue
            for a in range(len(neigh)):
                for b in range(a+1, len(neigh)):
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
            if tri_area(a, b, c) < self.tri_min_area:
                continue
            out.append((i, j, k2))
        return out

    def _tri_signature(self, P: np.ndarray, C: np.ndarray, tri):
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
        return f"{''.join(cols)}|{q[0]}-{q[1]}-{q[2]}"

    def _neighbor_keys(self, sig: str):
        # ±1 bin expansion
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

    def _color_consistent_maps(self, tri_obs, tri_map, C_obs, C_map):
        oi, oj, ok = tri_obs
        mi, mj, mk = tri_map
        obs_vs = [oi, oj, ok]
        map_vs = [mi, mj, mk]
        obs_cols = [int(C_obs[v]) for v in obs_vs]
        map_cols = [int(C_map[v]) for v in map_vs]

        perms = [
            (0,1,2),(0,2,1),
            (1,0,2),(1,2,0),
            (2,0,1),(2,1,0)
        ]
        out = []
        for p in perms:
            if obs_cols[0] == map_cols[p[0]] and obs_cols[1] == map_cols[p[1]] and obs_cols[2] == map_cols[p[2]]:
                out.append((obs_vs, [map_vs[p[0]], map_vs[p[1]], map_vs[p[2]]]))
        return out

    def _estimate_from_pair(self, P_obs, C_obs, tri_obs, P_map, C_map, tri_map):
        maps = self._color_consistent_maps(tri_obs, tri_map, C_obs, C_map)
        if not maps:
            return None

        best = None
        for oorder, morder in maps:
            P = np.array([P_obs[i] for i in oorder], dtype=np.float64)
            Q = np.array([P_map[i] for i in morder], dtype=np.float64)
            est = kabsch_se2(P, Q)
            if est is None:
                continue
            R, t, yaw = est
            P2 = (P @ R.T) + t[None, :]
            rms = float(np.sqrt(np.mean(np.sum((P2 - Q)**2, axis=1))))
            if best is None or rms < best[-1]:
                best = (R, t, yaw, rms)
        return best

    # ------------------ inlier scoring ------------------

    def _count_inliers(self, obs_pts: np.ndarray, obs_cls: np.ndarray,
                       map_pts: np.ndarray, map_cls: np.ndarray) -> int:
        inl = 0
        for (ox, oy), c in zip(obs_pts, obs_cls):
            best2 = None
            for (mx, my), mc in zip(map_pts, map_cls):
                if int(mc) != int(c):
                    continue
                dx = mx - ox
                dy = my - oy
                d2 = dx*dx + dy*dy
                if best2 is None or d2 < best2:
                    best2 = d2
            if best2 is not None and best2 <= self.inlier2:
                inl += 1
        return inl

    def _score_inliers(self, h: Hypothesis, obs_pts: np.ndarray, obs_cls: np.ndarray, t_now: float):
        trusted = self._trusted_mask(h, t_now)
        idx = np.where(trusted)[0].tolist()
        if len(idx) < 4:
            return 0, 0.0, (np.empty((0,2)), np.empty((0,), dtype=np.int32))

        map_pts = np.array([[h.tracks[i].x, h.tracks[i].y] for i in idx], dtype=np.float64)
        map_cls = np.array([h.tracks[i].cls_id for i in idx], dtype=np.int32)

        inl = self._count_inliers(obs_pts, obs_cls, map_pts, map_cls)
        ratio = inl / max(int(obs_pts.shape[0]), 1)
        return inl, ratio, (map_pts, map_cls)

    # ------------------ children proposal (structural MH) ------------------

    def _propose_children(self, h: Hypothesis, t_now: float, obs_pts: np.ndarray, obs_cls: np.ndarray):
        # need enough map to match
        inl_base, ratio_base, (map_pts, map_cls) = self._score_inliers(h, obs_pts, obs_cls, t_now)
        if map_pts.shape[0] < 6 or obs_pts.shape[0] < 5:
            return []

        # triangles
        obs_tris = self._build_triangles(obs_pts, obs_cls, self.tri_k)
        map_tris = self._build_triangles(map_pts, map_cls, self.tri_k)
        if not obs_tris or not map_tris:
            return []

        map_sig = defaultdict(list)
        for tri in map_tris:
            map_sig[self._tri_signature(map_pts, map_cls, tri)].append(tri)

        # candidate triangle pairs
        pairs = []
        for tobs in obs_tris:
            sig_o = self._tri_signature(obs_pts, obs_cls, tobs)
            for ksig in self._neighbor_keys(sig_o):
                for tmap in map_sig.get(ksig, []):
                    pairs.append((tobs, tmap))
                    if len(pairs) >= self.tri_max_pairs:
                        break
                if len(pairs) >= self.tri_max_pairs:
                    break
            if len(pairs) >= self.tri_max_pairs:
                break

        if not pairs:
            return []

        # score transforms
        scored = []
        for tobs, tmap in pairs:
            est = self._estimate_from_pair(obs_pts, obs_cls, tobs, map_pts, map_cls, tmap)
            if est is None:
                continue
            R, t, dyaw, rms = est
            if abs(dyaw) > self.max_yaw:
                continue
            if float(np.linalg.norm(t)) > self.max_trans:
                continue

            obs_tf = (obs_pts @ R.T) + t[None, :]
            inl = self._count_inliers(obs_tf, obs_cls, map_pts, map_cls)
            ratio = inl / max(int(obs_pts.shape[0]), 1)

            # must improve baseline significantly
            if inl < (inl_base + self.improve_margin):
                continue
            if inl < self.min_inliers or ratio < self.min_ratio:
                continue

            # score (bigger is better)
            penalty = (abs(dyaw) / max(self.max_yaw, 1e-6)) + (float(np.linalg.norm(t)) / max(self.max_trans, 1e-6))
            score = (self.w_inl * inl) + (self.w_ratio * ratio) - (self.w_rms * rms) - (self.w_pen * penalty)

            scored.append((score, inl, ratio, rms, R, t, dyaw))

        if not scored:
            return []

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:max(1, self.mh_B)]

        children = []
        for (score, inl, ratio, rms, R_new, t_new, dyaw) in top:
            # child hypothesis = new ∘ old
            R_child = R_new @ h.R
            t_child = (R_new @ h.t) + t_new

            # IMPORTANT: tracks are in corrected frame -> must transform them into child's frame
            new_tracks = []
            for tr in h.tracks:
                p = np.array([tr.x, tr.y], dtype=np.float64)
                p2 = R_new @ p + t_new
                new_tracks.append(Track(
                    x=float(p2[0]),
                    y=float(p2[1]),
                    cls_id=int(tr.cls_id),
                    last_seen=tr.last_seen,
                    hits=int(tr.hits)
                ))

            ch = Hypothesis(
                R=R_child,
                t=t_child,
                tracks=new_tracks,
                score=score,
                last_update_t=t_now,
                inliers_last=inl,
                ratio_last=ratio,
                rms_last=rms,
                dyaw_last=float(dyaw),
                trans_last=float(np.linalg.norm(t_new))
            )

            children.append(ch)

        return children

    # ------------------ hypothesis pruning ------------------

    def _prune_hypotheses(self, hyps: List[Hypothesis]) -> List[Hypothesis]:
        # small dedupe by (rounded corr params) to avoid clones exploding
        uniq = {}
        for h in hyps:
            yaw = math.atan2(h.R[1,0], h.R[0,0])
            key = (round(yaw, 3), round(float(h.t[0]), 2), round(float(h.t[1]), 2), len(h.tracks))
            if key not in uniq or h.score > uniq[key].score:
                uniq[key] = h

        arr = list(uniq.values())
        arr.sort(key=lambda hh: hh.score, reverse=True)
        arr = arr[:max(1, self.mh_K)]

        # ensure at least one hypothesis exists
        if not arr:
            arr = [Hypothesis(R=np.eye(2), t=np.zeros((2,)), tracks=[], score=0.0)]

        # logging (throttled)
        best = arr[0]
        self.get_logger().info(
            f"[mh] keep={len(arr)} bestScore={best.score:.2f} inl={best.inliers_last} "
            f"ratio={best.ratio_last:.2f} rms={best.rms_last:.2f} "
            f"dYaw={math.degrees(best.dyaw_last):.2f}deg |t|={best.trans_last:.2f}",
            throttle_duration_sec=0.5
        )
        return arr

    # ------------------ publish map cones ------------------

    def _publish_map(self, t: float, tracks: List[Track]):
        msg = ConeArrayWithCovariance()
        msg.header.stamp = stamp_from_float_seconds(t)
        msg.header.frame_id = "map"

        for tr in tracks:
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

        self.pub_map.publish(msg)


def main():
    rclpy.init()
    node = SGMHSLAMFull()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
