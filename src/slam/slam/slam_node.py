#!/usr/bin/env python3
#
# fslam_core.py
#
# FastSLAM (PF + per-particle landmark EKFs) with:
#  - JCBB (color-strict) + greedy fallback
#  - Adaptive gating (context + residual-driven)
#  - Confidence-based retention + pruning
#  - Drift nudging (small Gauss-Newton step on Δ per particle; guarded)
#
# Inputs:
#   /slam/odom_raw         (nav_msgs/Odometry, IMU-based, WORLD frame)
#   /perception/cones_fused (sensor_msgs/PointCloud2, BASE frame, x,y,z,class_id)
#
# Output (for visualizer):
#   /slam/odom       (nav_msgs/Odometry, map frame)
#   /slam/map_cones  (eufs_msgs/ConeArrayWithCovariance)
#
import math
import time
import bisect
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from collections import deque

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time as RclTime
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
)

from nav_msgs.msg import Odometry
from geometry_msgs.msg import (
    Pose,
    PoseWithCovariance,
    Twist,
    TwistWithCovariance,
    Point,
    Quaternion,
    Vector3,
)
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from eufs_msgs.msg import ConeArrayWithCovariance, ConeWithCovariance


# --------------------- helpers ---------------------
def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def rot2d(th: float) -> np.ndarray:
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]], float)


def yaw_from_quat(x, y, z, w) -> float:
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def quat_from_yaw(yaw: float) -> Quaternion:
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def _norm_ppf(p: float) -> float:
    if p <= 0.0:
        return -float("inf")
    if p >= 1.0:
        return float("inf")

    a1 = -3.969683028665376e01
    a2 = 2.209460984245205e02
    a3 = -2.759285104469687e02
    a4 = 1.383577518672690e02
    a5 = -3.066479806614716e01
    a6 = 2.506628277459239e00

    b1 = -5.447609879822406e01
    b2 = 1.615858368580409e02
    b3 = -1.556989798598866e02
    b4 = 6.680131188771972e01
    b5 = -1.328068155288572e01

    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e00
    c4 = -2.549732539343734e00
    c5 = 4.374664141464968e00
    c6 = 2.938163982698783e00

    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e00
    d4 = 3.754408661907416e00

    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (
            (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
            / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        )

    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
            / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        )

    q = p - 0.5
    r = q * q
    return (
        (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        * q
        / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    )


def chi2_quantile(dof: int, p: float) -> float:
    k = max(1, int(dof))
    z = _norm_ppf(p)
    t = 1.0 - 2.0 / (9.0 * k) + z * math.sqrt(2.0 / (9.0 * k))
    return k * (t ** 3)


# --------------------- landmark + particle data ---------------------
@dataclass
class Landmark:
    mean: np.ndarray  # (2,) world coords in this particle's map frame
    cov: np.ndarray  # (2x2)
    color: str  # 'blue' | 'yellow' | 'orange' | 'big'
    promoted: bool = False
    last_seen_t: float = -1.0
    last_hit_t: float = -1.0
    distinct_hits: int = 0
    conf: float = 0.0  # EWMA [0..1]
    retained: bool = False  # never deleted if True


@dataclass
class Particle:
    # Δ = [tx, ty, dθ] = SE(2) transform from odom → map:
    # map_p = R(dθ) * odom_p + [tx, ty]
    delta: np.ndarray
    weight: float
    lms: List[Landmark] = field(default_factory=list)


# --------------------- FastSLAM core node ---------------------
class FastSLAMCore(Node):
    def __init__(self):
        super().__init__(
            "fastslam_core", automatically_declare_parameters_from_overrides=True
        )

        # ---- topics ----
        self.declare_parameter("topics.pose_in", "/slam/odom_raw")
        self.declare_parameter("topics.cones_in", "/perception/cones_fused")
        self.declare_parameter("topics.odom_out", "/slam/odom")
        self.declare_parameter("topics.map_out", "/slam/map_cones")

        # ---- QoS ----
        self.declare_parameter("qos.best_effort", True)
        self.declare_parameter("qos.depth", 200)

        # ---- DA / measurement ----
        # detections_frame is effectively "base": cones_fused is in BASE frame (car frame)
        self.declare_parameter("detections_frame", "base")
        self.declare_parameter("chi2_gate_2d", 9.21)  # ~99% for 2 dof
        self.declare_parameter("joint_sig", 0.95)
        self.declare_parameter("joint_relax", 1.4)
        self.declare_parameter("min_k_for_joint", 3)
        # This is now the ONLY measurement floor: we do not read per-cone covariances anymore.
        self.declare_parameter("meas_sigma_floor_xy", 0.30)
        self.declare_parameter("alpha_trans", 0.9)
        self.declare_parameter("beta_rot", 0.9)
        self.declare_parameter("target_cone_hz", 15.0)
        self.declare_parameter("odom_buffer_sec", 2.0)
        self.declare_parameter("extrapolation_cap_ms", 60.0)
        self.declare_parameter("max_pairs_print", 6)

        # ---- PF ----
        self.declare_parameter("pf.num_particles", 400)
        self.declare_parameter("pf.resample_neff_ratio", 0.5)
        # Looser exploration (defaults only; still overridable)
        self.declare_parameter("pf.process_std_xy_m", 0.06)
        self.declare_parameter("pf.process_std_yaw_rad", 0.012)
        self.declare_parameter("pf.likelihood_floor", 1e-12)
        self.declare_parameter("pf.init_std_xy_m", 0.0)
        self.declare_parameter("pf.init_std_yaw_rad", 0.0)
        self.declare_parameter("seed", 7)

        # ---- Birth / Promotion / Deletion ----
        self.declare_parameter("birth.single_gate_sig", 0.95)
        self.declare_parameter("promotion.k_in_n", [2, 3])  # use only k (distinct hits >= k)
        self.declare_parameter("promotion_min_dt_s", 0.17)
        self.declare_parameter("delete.cov_trace_max", 1.0)

        # ---- Retention / Confidence ----
        self.declare_parameter("retain.conf_beta", 0.15)
        self.declare_parameter("retain.decay_hz", 0.05)
        self.declare_parameter("retain.retain_conf_thr", 0.6)
        self.declare_parameter("retain.retain_min_hits", 2)
        self.declare_parameter("retain.retain_cov_tr_max", 0.25)
        self.declare_parameter("retain.prune_conf_thr", 0.10)
        self.declare_parameter("retain.max_unseen_s", 30.0)
        # Merge threshold tuned: default ~ chi²(2, 0.99) so we MERGE instead of birthing duplicates
        self.declare_parameter("retain.merge_m2_thr", 9.21)

        # ---- JCBB bounds & logging ----
        self.declare_parameter("jcbb.max_obs_total", 8)
        self.declare_parameter("jcbb.max_obs_per_color", 2)
        self.declare_parameter("jcbb.cand_per_obs", 3)
        self.declare_parameter("jcbb.time_budget_ms", 600.0)  # total per-frame budget (shared)
        self.declare_parameter("jcbb.crop_radius_m", 17.0)
        self.declare_parameter("log.jcbb", True)

        # ---- Optional: drop stale cone frames (OFF by default) ----
        self.declare_parameter("drop_stale_ms", 0.0)  # set >0 (e.g., 200) to skip old frames

        # ---- Map publish thresholds ----
        self.declare_parameter("map.publish_min_conf", 0.15)
        self.declare_parameter("map.publish_min_hits", 1)
        # Align published map to Δ̂ so it matches /slam/odom
        self.declare_parameter("map.align_to_delta_hat", True)

        # ---- Adaptive gating params ----
        self.declare_parameter("gate.adapt_enable", True)
        self.declare_parameter("gate.scale_min", 0.6)
        self.declare_parameter("gate.scale_max", 2.5)
        self.declare_parameter("gate.target_cands", 1.5)
        self.declare_parameter("gate.k_pos", 0.10)
        self.declare_parameter("gate.k_yaw", 0.50)

        # ---- Perception-trust knobs ----
        self.declare_parameter("meas.trust_gain", 2.3)  # R_eff = R / trust_gain^2
        self.declare_parameter("like.temperature", 0.5)  # <1 => sharper toward perception

        # ---- Drift nudge ----
        self.declare_parameter("nudge.enable", True)
        self.declare_parameter("nudge.gain_xy", 0.4)
        self.declare_parameter("nudge.gain_yaw", 0.2)
        self.declare_parameter("nudge.max_xy_step", 0.5)
        self.declare_parameter("nudge.max_yaw_step", 6.5)  # deg clamp per update

        # ---- Class-id mapping (for cones_fused PointCloud2) ----
        # Defaults assume: 0=blue, 1=yellow, 2=orange, 3=big, others=ignored/unknown
        self.declare_parameter("class_id.blue", 0)
        self.declare_parameter("class_id.yellow", 1)
        self.declare_parameter("class_id.orange", 2)
        self.declare_parameter("class_id.big", 3)

        # ---- read params ----
        gp = self.get_parameter
        self.pose_topic_in = str(gp("topics.pose_in").value)
        self.cones_topic_in = str(gp("topics.cones_in").value)
        self.odom_topic_out = str(gp("topics.odom_out").value)
        self.map_topic_out = str(gp("topics.map_out").value)

        best_effort = bool(gp("qos.best_effort").value)
        depth = int(gp("qos.depth").value)
        self.qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=depth,
            reliability=(
                QoSReliabilityPolicy.BEST_EFFORT
                if best_effort
                else QoSReliabilityPolicy.RELIABLE
            ),
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.detections_frame = str(gp("detections_frame").value).lower()
        self.chi2_gate_base = float(gp("chi2_gate_2d").value)
        self.joint_sig = float(gp("joint_sig").value)
        self.joint_relax = float(gp("joint_relax").value)
        self.min_k_for_joint = int(gp("min_k_for_joint").value)
        self.sigma0 = float(gp("meas_sigma_floor_xy").value)
        self.alpha = float(gp("alpha_trans").value)
        self.beta = float(gp("beta_rot").value)
        self.target_cone_hz = float(gp("target_cone_hz").value)
        self.odom_buffer_sec = float(gp("odom_buffer_sec").value)
        self.extrap_cap = float(gp("extrapolation_cap_ms").value) / 1000.0
        self.max_pairs_print = int(gp("max_pairs_print").value)

        self.Np = int(gp("pf.num_particles").value)
        self.neff_ratio = float(gp("pf.resample_neff_ratio").value)
        self.p_std_xy = float(gp("pf.process_std_xy_m").value)
        self.p_std_yaw = float(gp("pf.process_std_yaw_rad").value)
        self.like_floor = float(gp("pf.likelihood_floor").value)
        self.init_std_xy = float(gp("pf.init_std_xy_m").value)
        self.init_std_yaw = float(gp("pf.init_std_yaw_rad").value)
        self.seed = int(gp("seed").value)

        self.birth_sig = float(gp("birth.single_gate_sig").value)
        self.prom_k, _ = [int(x) for x in gp("promotion.k_in_n").value]
        self.prom_dt_min = float(gp("promotion_min_dt_s").value)
        self.del_covtr = float(gp("delete.cov_trace_max").value)

        self.conf_beta = float(gp("retain.conf_beta").value)
        self.decay_hz = float(gp("retain.decay_hz").value)
        self.retain_thr = float(gp("retain.retain_conf_thr").value)
        self.retain_hits = int(gp("retain.retain_min_hits").value)
        self.retain_cov_tr = float(gp("retain.retain_cov_tr_max").value)
        self.prune_thr = float(gp("retain.prune_conf_thr").value)
        self.max_unseen_s = float(gp("retain.max_unseen_s").value)
        self.merge_m2_thr = float(gp("retain.merge_m2_thr").value)

        self.max_obs_total = int(gp("jcbb.max_obs_total").value)
        self.max_obs_per_col = int(gp("jcbb.max_obs_per_color").value)
        self.max_obs_per_color = self.max_obs_per_col
        self.cand_per_obs = int(gp("jcbb.cand_per_obs").value)
        self.jcbb_budget_s = float(gp("jcbb.time_budget_ms").value) / 1000.0
        self.crop_radius_m = float(gp("jcbb.crop_radius_m").value)
        self.log_jcbb = bool(gp("log.jcbb").value)
        self.drop_stale_s = float(gp("drop_stale_ms").value) / 1000.0

        self.map_pub_conf = float(gp("map.publish_min_conf").value)
        self.map_pub_hits = int(gp("map.publish_min_hits").value)
        self.map_align_hat = bool(gp("map.align_to_delta_hat").value)

        self.adapt_gate = bool(gp("gate.adapt_enable").value)
        self.gate_min = float(gp("gate.scale_min").value)
        self.gate_max = float(gp("gate.scale_max").value)
        self.gate_target = float(gp("gate.target_cands").value)
        self.gate_k_pos = float(gp("gate.k_pos").value)
        self.gate_k_yaw = float(gp("gate.k_yaw").value)

        self.trust_gain = max(1.0, float(gp("meas.trust_gain").value))
        self.like_temp = max(0.2, float(gp("like.temperature").value))  # clamp

        self.nudge_enable = bool(gp("nudge.enable").value)
        self.nudge_xy = float(gp("nudge.gain_xy").value)
        self.nudge_yaw = float(gp("nudge.gain_yaw").value)
        self.nudge_max_xy = float(gp("nudge.max_xy_step").value)
        self.nudge_max_yaw = math.radians(float(gp("nudge.max_yaw_step").value))

        # class-id mapping
        self.cls_blue = int(gp("class_id.blue").value)
        self.cls_yellow = int(gp("class_id.yellow").value)
        self.cls_orange = int(gp("class_id.orange").value)
        self.cls_big = int(gp("class_id.big").value)

        # ---- state ----
        np.random.seed(self.seed)
        self.particles: List[Particle] = [
            Particle(delta=np.zeros(3, float), weight=1.0 / self.Np)
            for _ in range(self.Np)
        ]
        if self.init_std_xy > 0.0 or self.init_std_yaw > 0.0:
            for p in self.particles:
                p.delta[0] = np.random.normal(0.0, self.init_std_xy)
                p.delta[1] = np.random.normal(0.0, self.init_std_xy)
                p.delta[2] = np.random.normal(0.0, self.init_std_yaw)

        self.delta_hat = np.zeros(3, float)
        self.have_pose_feed = False

        # odom buffer: (t, x, y, yaw, v, yawrate)
        self.odom_buf: deque = deque()
        self.pose_latest = np.array([0.0, 0.0, 0.0], float)

        self.last_cone_ts: Optional[float] = None
        self.last_cone_stamp = None  # ROS time of last accepted cone frame

        # ---- I/O ----
        self.create_subscription(Odometry, self.pose_topic_in, self.cb_odom, self.qos)
        self.create_subscription(PointCloud2, self.cones_topic_in, self.cb_cones, self.qos)
        self.pub_odom = self.create_publisher(Odometry, self.odom_topic_out, 10)
        self.pub_map = self.create_publisher(ConeArrayWithCovariance, self.map_topic_out, 10)

        self.get_logger().info(
            f"[fastslam] up. in:pose={self.pose_topic_in} cones={self.cones_topic_in} | "
            f"out:odom={self.odom_topic_out} map={self.map_topic_out}"
        )
        self.get_logger().info(
            f"[fastslam] Np={self.Np} QoS={'BEST_EFFORT' if best_effort else 'RELIABLE'} "
            f"detections_frame={self.detections_frame}"
        )

    # -------------------- class-id → color --------------------
    def _class_to_color(self, cls_id: int) -> Optional[str]:
        if cls_id == self.cls_blue:
            return "blue"
        if cls_id == self.cls_yellow:
            return "yellow"
        if cls_id == self.cls_orange:
            return "orange"
        if cls_id == self.cls_big:
            return "big"
        return None  # ignore unknown / gray

    # -------------------- odom buffer helpers --------------------
    def _pose_at(self, t_query: float):
        """Return pose at t_query (x,y,yaw), plus timing mismatch dt_pose used for inflation."""
        if not self.odom_buf:
            return self.pose_latest.copy(), float("inf"), 0.0, 0.0

        times = [it[0] for it in self.odom_buf]
        idx = bisect.bisect_left(times, t_query)

        if idx == 0:
            t0, x0, y0, yaw0, v0, yr0 = self.odom_buf[0]
            return np.array([x0, y0, yaw0], float), 0.0, v0, yr0

        if idx >= len(self.odom_buf):
            t1, x1, y1, yaw1, v1, yr1 = self.odom_buf[-1]
            dt = t_query - t1
            dt_c = max(0.0, min(dt, self.extrap_cap))
            if abs(yr1) < 1e-3:
                x = x1 + v1 * dt_c * math.cos(yaw1)
                y = y1 + v1 * dt_c * math.sin(yaw1)
                yaw = wrap(yaw1)
            else:
                x = x1 + (v1 / yr1) * (
                    math.sin(yaw1 + yr1 * dt_c) - math.sin(yaw1)
                )
                y = y1 - (v1 / yr1) * (
                    math.cos(yaw1 + yr1 * dt_c) - math.cos(yaw1)
                )
                yaw = wrap(yaw1 + yr1 * dt_c)
            return np.array([x, y, yaw], float), abs(dt_c), v1, yr1

        t0, x0, y0, yaw0, v0, yr0 = self.odom_buf[idx - 1]
        t1, x1, y1, yaw1, v1, yr1 = self.odom_buf[idx]
        if t1 == t0:
            return np.array([x0, y0, yaw0], float), 0.0, v0, yr0
        a = (t_query - t0) / (t1 - t0)
        x = x0 + a * (x1 - x0)
        y = y0 + a * (y1 - y0)
        dyaw = wrap(yaw1 - yaw0)
        yaw = wrap(yaw0 + a * dyaw)
        v = (1 - a) * v0 + a * v1
        yr = (1 - a) * yr0 + a * yr1
        return np.array([x, y, yaw], float), 0.0, v, yr

    # -------------------- cap observations to bound JCBB time --------------------
    def _cap_observations(
        self, obs: List[Tuple[np.ndarray, np.ndarray, str]]
    ) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """Per-color cap, then global cap by nearest range in BODY frame."""
        if not obs:
            return obs

        def _r(o):
            return float(np.linalg.norm(o[0]))

        by_color = {"blue": [], "yellow": [], "orange": [], "big": []}
        for o in obs:
            col = o[2]
            if col not in by_color:
                by_color[col] = []
            by_color[col].append(o)

        capped = []
        per_col = max(1, self.max_obs_per_color)
        for arr in by_color.values():
            if arr:
                arr.sort(key=_r)
                capped.extend(arr[:per_col])

        total_cap = max(1, self.max_obs_total)
        if len(capped) <= total_cap:
            return capped
        capped.sort(key=_r)
        return capped[:total_cap]

    # -------------------- Landmark pruning --------------------
    def _prune_landmarks(self, p: Particle, t_now: float):
        """Remove weak / stale landmarks for a single particle."""
        kept = []
        for lm in p.lms:
            if lm.retained:
                kept.append(lm)
                continue

            if lm.last_seen_t < 0.0:
                kept.append(lm)
                continue

            age = t_now - lm.last_seen_t
            cov_tr = float(np.trace(lm.cov))

            if age > self.max_unseen_s:
                continue
            if cov_tr > self.del_covtr:
                continue
            if lm.conf < self.prune_thr:
                continue

            kept.append(lm)

        p.lms = kept

    # -------------------- Odometry covariance from particle spread --------------------
    def _fill_odom_covariances(self, od: Odometry):
        if not self.particles:
            return

        w = np.array([p.weight for p in self.particles], float)
        mu = self.delta_hat

        dxs = np.array([p.delta[0] - mu[0] for p in self.particles], float)
        dys = np.array([p.delta[1] - mu[1] for p in self.particles], float)
        dths = np.array([wrap(p.delta[2] - mu[2]) for p in self.particles], float)

        var_x = float(np.sum(w * dxs * dxs)) + 1e-4
        var_y = float(np.sum(w * dys * dys)) + 1e-4
        var_th = float(np.sum(w * dths * dths)) + math.radians(1.0) ** 2

        pose_cov = [0.0] * 36
        pose_cov[0] = var_x
        pose_cov[7] = var_y
        pose_cov[35] = var_th

        od.pose.covariance = pose_cov

        twist_cov = [0.0] * 36
        twist_cov[0] = 0.25
        twist_cov[7] = 0.25
        twist_cov[35] = math.radians(5.0) ** 2
        od.twist.covariance = twist_cov

    # -------------------- Odometry IN → publish corrected odom at same rate --------------------
    def cb_odom(self, msg: Odometry):
        t = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        v = math.hypot(vx, vy)
        yr = float(msg.twist.twist.angular.z)

        self.pose_latest = np.array([x, y, yaw], float)
        self.odom_buf.append((t, x, y, yaw, v, yr))
        tmin = t - self.odom_buffer_sec
        while self.odom_buf and self.odom_buf[0][0] < tmin:
            self.odom_buf.popleft()

        if not self.have_pose_feed:
            self.have_pose_feed = True
            self.get_logger().info("[fastslam] proposal feed ready.")

        # Use mean Δ̂ as SE(2) map←odom transform
        tx, ty, dth = self.delta_hat
        R_d = rot2d(dth)

        p_o = np.array([x, y], float)
        p_m = R_d @ p_o + np.array([tx, ty], float)
        yaw_m = wrap(yaw + dth)

        od = Odometry()
        od.header.stamp = msg.header.stamp
        od.header.frame_id = "map"
        od.child_frame_id = "base_link"
        od.pose = PoseWithCovariance()
        od.twist = TwistWithCovariance()
        od.pose.pose = Pose(
            position=Point(x=float(p_m[0]), y=float(p_m[1]), z=0.0),
            orientation=quat_from_yaw(yaw_m),
        )
        od.twist.twist = Twist(
            linear=Vector3(x=v * math.cos(yaw_m), y=v * math.sin(yaw_m), z=0.0),
            angular=Vector3(x=0.0, y=0.0, z=yr),
        )

        self._fill_odom_covariances(od)
        self.pub_odom.publish(od)

        self._publish_map(t)

    # -------------------- Cones IN (PointCloud2) → per-particle JCBB + EKF map update --------------------
    def cb_cones(self, msg: PointCloud2):
        if not self.have_pose_feed:
            return

        t_frame_start = time.perf_counter()
        t_z = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9

        # simple downsampler to ~target_cone_hz and frame Δt
        if self.last_cone_ts is not None:
            dt_frame = t_z - self.last_cone_ts
            if dt_frame < (1.0 / max(1e-3, self.target_cone_hz)):
                return
        else:
            dt_frame = 0.0
        self.last_cone_ts = t_z
        self.last_cone_stamp = msg.header.stamp

        # optional: skip stale frames to avoid backlog (OFF by default)
        if self.drop_stale_s > 0.0:
            now_s = self.get_clock().now().nanoseconds * 1e-9
            if (now_s - t_z) > self.drop_stale_s:
                return

        # Build obs in BODY frame (pb, Sb, color) from PointCloud2 (x,y,z,class_id)
        obs: List[Tuple[np.ndarray, np.ndarray, str]] = []

        odom_pose, dt_pose, v, yawrate = self._pose_at(t_z)
        x_o, y_o, yaw_o = odom_pose
        yaw_o = float(yaw_o)

        # fixed BODY-frame covariance: we trust fusion, so just a constant floor
        Sb_body = (self.sigma0 ** 2) * np.eye(2, dtype=float)

        try:
            for x, y, _, cls_id in pc2.read_points(
                msg, field_names=("x", "y", "z", "class_id"), skip_nans=True
            ):
                color = self._class_to_color(int(cls_id))
                if color is None:
                    continue  # ignore unknown/gray
                pb = np.array([float(x), float(y)], float)
                obs.append((pb, Sb_body, color))
        except Exception as e:
            self.get_logger().warn(f"[fastslam] error parsing cones_fused cloud: {e}")
            return

        if not obs:
            return

        # ENFORCE CAPS to keep JCBB bounded
        obs = self._cap_observations(obs)

        # PF predict on Δ
        self._pf_predict()

        # Adaptive chi² gate for the frame
        chi2_gate = self.chi2_gate_base
        if self.adapt_gate:
            drift_pos = math.hypot(self.delta_hat[0], self.delta_hat[1])
            drift_yaw = abs(self.delta_hat[2])
            scale = 1.0 + self.gate_k_pos * drift_pos + self.gate_k_yaw * drift_yaw
            scale = min(max(scale, self.gate_min), self.gate_max)
            chi2_gate *= scale

        new_w = np.zeros(self.Np, float)

        # Per-FRAME budget shared across particles
        pp_budget = self.jcbb_budget_s / float(min(self.Np, 64))
        pp_budget = max(0.001, min(pp_budget, 0.015))

        trust2 = self.trust_gain * self.trust_gain
        J = np.array([[0.0, -1.0], [1.0, 0.0]])

        for i, p in enumerate(self.particles):
            tx, ty, dth = p.delta
            R_d = rot2d(dth)
            yaw_c = wrap(yaw_o + dth)
            Rc = rot2d(yaw_c)
            pc_map = R_d @ np.array([x_o, y_o], float) + np.array([tx, ty], float)

            cands: List[List[Tuple[int, float]]] = []
            total_cands = 0
            gate_eff = chi2_gate

            # Precompute R_meas per obs (depends on v, yawrate, dt_pose) but not on particle
            R_meas_list: List[np.ndarray] = []

            for (pb, Sb, _) in obs:
                # world-frame predicted meas cov from BODY-floor
                S0_w = Rc @ Sb @ Rc.T

                trans_var = self.alpha * (v ** 2) * (dt_pose ** 2)
                sigma_theta2 = self.beta * ((yawrate * dt_pose) ** 2)
                u = Rc @ (J @ pb)
                S_yaw = sigma_theta2 * np.outer(u, u)
                R_meas = (S0_w + np.eye(2) * trans_var + S_yaw) / trust2
                R_meas[0, 0] += 1e-9
                R_meas[1, 1] += 1e-9
                R_meas_list.append(R_meas)

            # association candidates per obs
            for oi, (pb, _, color) in enumerate(obs):
                pw_pred = pc_map + Rc @ pb
                R_meas = R_meas_list[oi]

                near_idx = []
                if p.lms:
                    for j, lm in enumerate(p.lms):
                        if lm.color != color:
                            continue
                        if (
                            abs(lm.mean[0] - pw_pred[0]) > self.crop_radius_m
                            or abs(lm.mean[1] - pw_pred[1]) > self.crop_radius_m
                        ):
                            continue
                        if (
                            (lm.mean - pw_pred) @ (lm.mean - pw_pred)
                            > (self.crop_radius_m ** 2)
                        ):
                            continue
                        near_idx.append(j)

                lst = []
                for j in near_idx:
                    lm = p.lms[j]
                    innov = pw_pred - lm.mean
                    S_gate = lm.cov + R_meas
                    try:
                        Sinv = np.linalg.inv(S_gate)
                    except np.linalg.LinAlgError:
                        Sinv = np.linalg.pinv(S_gate)
                    m2_val = float(innov.T @ Sinv @ innov)
                    if m2_val <= gate_eff:
                        lst.append((j, m2_val))

                if lst:
                    lst.sort(key=lambda t: t[1])
                    if self.cand_per_obs > 0:
                        lst = lst[: self.cand_per_obs]
                total_cands += len(lst)
                cands.append(lst)

            avg_c = (total_cands / max(1, len(obs)))
            if self.adapt_gate:
                if avg_c > (self.gate_target * 1.5):
                    gate_eff = max(
                        self.gate_min * self.chi2_gate_base, gate_eff * 0.8
                    )
                    keep = max(1, int(math.ceil(self.gate_target)))
                    for k in range(len(cands)):
                        if cands[k]:
                            cands[k] = cands[k][:keep]
                elif avg_c < 0.5:
                    gate_eff = min(
                        self.gate_max * self.chi2_gate_base, gate_eff * 1.25
                    )

            # ---------- JCBB with per-particle deadline ----------
            order = sorted(
                range(len(obs)),
                key=lambda k: len(cands[k]) if cands[k] else 9999,
            )
            cands_ord = [cands[k] for k in order]
            best_pairs: List[Tuple[int, int, float]] = []
            best_K = 0
            used = set()
            t_deadline = time.perf_counter() + pp_budget
            timed_out = False

            def dfs(idx, cur_pairs, cur_sum):
                nonlocal best_pairs, best_K, timed_out
                if time.perf_counter() > t_deadline:
                    timed_out = True
                    return
                cur_K = len(cur_pairs)
                if cur_K + (len(cands_ord) - idx) < best_K:
                    return
                if cur_K >= self.min_k_for_joint:
                    df_loc = 2 * cur_K
                    thresh = self.joint_relax * chi2_quantile(df_loc, self.joint_sig)
                    if cur_sum > thresh:
                        return
                if idx == len(cands_ord):
                    if cur_K > best_K:
                        best_K = cur_K
                        best_pairs = cur_pairs.copy()
                    return
                dfs(idx + 1, cur_pairs, cur_sum)
                for (lmj, m2_val) in cands_ord[idx]:
                    if lmj in used:
                        continue
                    used.add(lmj)
                    cur_pairs.append((order[idx], lmj, m2_val))
                    dfs(idx + 1, cur_pairs, cur_sum + m2_val)
                    cur_pairs.pop()
                    used.remove(lmj)

            dfs(0, [], 0.0)

            # Fallback: greedy NN only if JCBB did NOT time out
            if len(best_pairs) == 0 and not timed_out:
                used_lm = set()
                for oi in order:
                    if not cands[oi]:
                        continue
                    lmj, m2_val = min(cands[oi], key=lambda t: t[1])
                    if lmj in used_lm:
                        continue
                    if m2_val <= gate_eff:
                        best_pairs.append((oi, lmj, m2_val))
                        used_lm.add(lmj)

            if timed_out and self.log_jcbb:
                self.get_logger().warn(
                    "[fastslam/jcbb] per-particle JCBB timed out; using no associations for this particle"
                )

            # Likelihood
            m2_sum = float(sum(m2_val for (_, _, m2_val) in best_pairs))
            K = len(best_pairs)
            df_joint = 2 * max(1, K)
            gate_joint = self.joint_relax * chi2_quantile(df_joint, self.joint_sig)
            have_valid = (K >= 2 and m2_sum <= gate_joint)
            if have_valid:
                like = max(
                    math.exp(-0.5 * m2_sum / self.like_temp), self.like_floor
                )
            elif K == 1 and m2_sum <= gate_eff:
                like = max(
                    math.exp(-0.5 * m2_sum / self.like_temp), self.like_floor
                )
            else:
                like = self.like_floor
            new_w[i] = p.weight * like

            # Conf decay for unmatched observations (use real frame Δt)
            decay = math.exp(-self.decay_hz * max(0.0, float(dt_frame)))
            matched_lm_ids = {lmj for (_, lmj, _) in best_pairs}
            for lj, lm in enumerate(p.lms):
                if lj not in matched_lm_ids:
                    lm.conf *= decay

            # EKF updates + promotion/retention
            yaw_terms = []
            for (oi, lmj, _) in best_pairs:
                pb, Sb, color = obs[oi]
                pw_pred = pc_map + Rc @ pb
                R_meas = R_meas_list[oi]

                lm = p.lms[lmj]
                P = lm.cov
                S = P + R_meas
                try:
                    Sinv = np.linalg.inv(S)
                except np.linalg.LinAlgError:
                    Sinv = np.linalg.pinv(S)
                Kmat = P @ Sinv
                innov = pw_pred - lm.mean
                lm.mean = lm.mean + Kmat @ innov
                lm.cov = (np.eye(2) - Kmat) @ P

                yaw_terms.append((Rc @ pb, innov, R_meas))

                lm.conf = (1.0 - self.conf_beta) * lm.conf + self.conf_beta * 1.0
                if (lm.last_hit_t < 0.0) or (
                    (t_z - lm.last_hit_t) >= self.prom_dt_min
                ):
                    lm.distinct_hits += 1
                    lm.last_hit_t = t_z
                lm.last_seen_t = t_z
                if (not lm.promoted) and (lm.distinct_hits >= self.prom_k):
                    lm.promoted = True
                if (
                    not lm.retained
                    and lm.promoted
                    and lm.conf >= self.retain_thr
                    and lm.distinct_hits >= self.retain_hits
                    and float(np.trace(lm.cov)) <= self.retain_cov_tr
                ):
                    lm.retained = True

            # Δ-nudge (Gauss–Newton step)
            if self.nudge_enable and len(yaw_terms) >= 1:
                A = np.zeros((3, 3), float)
                b_vec = np.zeros((3,), float)
                for (Rc_pb, innov, R_meas) in yaw_terms:
                    r_vec = -innov
                    u_vec = J @ Rc_pb
                    H = np.array(
                        [
                            [1.0, 0.0, float(u_vec[0])],
                            [0.0, 1.0, float(u_vec[1])],
                        ],
                        float,
                    )
                    try:
                        Rinv = np.linalg.inv(R_meas)
                    except np.linalg.LinAlgError:
                        Rinv = np.linalg.pinv(R_meas)
                    At = H.T @ Rinv
                    A += At @ H
                    b_vec += At @ r_vec

                try:
                    d_step_raw = np.linalg.solve(A, b_vec)
                except np.linalg.LinAlgError:
                    d_step_raw = np.linalg.pinv(A) @ b_vec

                if np.all(np.isfinite(d_step_raw)):
                    dxy = math.hypot(d_step_raw[0], d_step_raw[1])
                    scale_xy = (
                        min(1.0, self.nudge_max_xy / dxy) if dxy > 0.0 else 1.0
                    )
                    dyaw = float(d_step_raw[2])
                    scale_yaw = (
                        min(1.0, self.nudge_max_yaw / abs(dyaw))
                        if abs(dyaw) > 0.0
                        else 1.0
                    )

                    d_step = np.array(
                        [
                            self.nudge_xy * d_step_raw[0] * scale_xy,
                            self.nudge_xy * d_step_raw[1] * scale_xy,
                            self.nudge_yaw * dyaw * scale_yaw,
                        ],
                        float,
                    )

                    p.delta[0] += d_step[0]
                    p.delta[1] += d_step[1]
                    p.delta[2] = wrap(p.delta[2] + d_step[2])

            # births / merges for unmatched obs – now driven by merge_m2_thr tuned to chi²
            matched_obs_idx = set([oi for (oi, _, _) in best_pairs])
            for oi, (pb, Sb, color) in enumerate(obs):
                if oi in matched_obs_idx:
                    continue
                pw_pred = pc_map + Rc @ pb
                R_meas = R_meas_list[oi]

                best_j, best_m2 = -1, float("inf")
                for j, lm2 in enumerate(p.lms):
                    if lm2.color != color:
                        continue
                    Ssum = lm2.cov + R_meas
                    try:
                        Sinv = np.linalg.inv(Ssum)
                    except np.linalg.LinAlgError:
                        Sinv = np.linalg.pinv(Ssum)
                    diff = pw_pred - lm2.mean
                    m2_val = float(diff.T @ Sinv @ diff)
                    if m2_val < best_m2:
                        best_m2, best_j = m2_val, j

                if best_m2 <= self.merge_m2_thr and best_j >= 0:
                    lm2 = p.lms[best_j]
                    Kb = lm2.cov @ np.linalg.pinv(lm2.cov + R_meas)
                    lm2.mean = lm2.mean + Kb @ (pw_pred - lm2.mean)
                    lm2.cov = (np.eye(2) - Kb) @ lm2.cov
                    lm2.last_seen_t = t_z
                    if (lm2.last_hit_t < 0.0) or (
                        (t_z - lm2.last_hit_t) >= self.prom_dt_min
                    ):
                        lm2.distinct_hits += 1
                        lm2.last_hit_t = t_z
                    lm2.conf = (1.0 - self.conf_beta) * lm2.conf + self.conf_beta * 1.0
                    if (not lm2.promoted) and lm2.distinct_hits >= self.prom_k:
                        lm2.promoted = True
                    if (
                        not lm2.retained
                        and lm2.promoted
                        and lm2.conf >= self.retain_thr
                        and lm2.distinct_hits >= self.retain_hits
                        and float(np.trace(lm2.cov)) <= self.retain_cov_tr
                    ):
                        lm2.retained = True
                else:
                    p.lms.append(
                        Landmark(
                            mean=pw_pred.copy(),
                            cov=R_meas.copy(),
                            color=color,
                            promoted=False,
                            last_seen_t=t_z,
                            last_hit_t=t_z,
                            distinct_hits=1,
                            conf=self.conf_beta,
                            retained=False,
                        )
                    )

        # normalize + resample + Δ̂
        sw = float(np.sum(new_w))
        if sw <= 0.0 or not math.isfinite(sw):
            for p in self.particles:
                p.weight = 1.0 / self.Np
        else:
            for i, p in enumerate(self.particles):
                p.weight = new_w[i] / sw

        neff = 1.0 / float(
            np.sum([p.weight * p.weight for p in self.particles])
        )
        if neff < self.neff_ratio * self.Np:
            self._systematic_resample()

        self.delta_hat = self._estimate_delta()

        for p in self.particles:
            self._prune_landmarks(p, t_z)

        proc_ms = int((time.perf_counter() - t_frame_start) * 1000)
        if self.log_jcbb:
            self.get_logger().info(
                f"[fastslam/jcbb] t={t_z:7.2f}s obs={len(obs)} proc={proc_ms}ms neff={neff:.1f} "
                f"Δ̂=[{self.delta_hat[0]:+.3f},{self.delta_hat[1]:+.3f},{math.degrees(self.delta_hat[2]):+.1f}°]"
            )

    # -------------------- PF core over Δ --------------------
    def _pf_predict(self):
        if self.p_std_xy > 0.0:
            noise_xy = np.random.normal(0.0, self.p_std_xy, size=(self.Np, 2))
        else:
            noise_xy = np.zeros((self.Np, 2))
        if self.p_std_yaw > 0.0:
            noise_y = np.random.normal(0.0, self.p_std_yaw, size=(self.Np,))
        else:
            noise_y = np.zeros((self.Np,))
        for i, p in enumerate(self.particles):
            p.delta[0] += noise_xy[i, 0]
            p.delta[1] += noise_xy[i, 1]
            p.delta[2] = wrap(p.delta[2] + noise_y[i])

    def _systematic_resample(self):
        N = self.Np
        w = np.array([p.weight for p in self.particles], float)
        positions = (np.arange(N) + np.random.uniform()) / N
        indexes = np.zeros(N, dtype=int)
        cs = np.cumsum(w)
        i, j = 0, 0
        while i < N:
            if positions[i] < cs[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        new_particles: List[Particle] = []
        for idx in indexes:
            src = self.particles[idx]
            new_lms = [
                Landmark(
                    mean=lm.mean.copy(),
                    cov=lm.cov.copy(),
                    color=lm.color,
                    promoted=lm.promoted,
                    last_seen_t=lm.last_seen_t,
                    last_hit_t=lm.last_hit_t,
                    distinct_hits=lm.distinct_hits,
                    conf=lm.conf,
                    retained=lm.retained,
                )
                for lm in src.lms
            ]
            new_particles.append(
                Particle(delta=src.delta.copy(), weight=1.0 / N, lms=new_lms)
            )
        self.particles = new_particles

    def _estimate_delta(self) -> np.ndarray:
        w = np.array([p.weight for p in self.particles], float)
        tx = float(np.sum(w * np.array([p.delta[0] for p in self.particles])))
        ty = float(np.sum(w * np.array([p.delta[1] for p in self.particles])))
        ang = np.array([p.delta[2] for p in self.particles], float)
        cs = float(np.sum(w * np.cos(ang)))
        sn = float(np.sum(w * np.sin(ang)))
        dth = math.atan2(sn, cs)
        return np.array([tx, ty, dth], float)

    # -------------------- Publish map from best particle (aligned to Δ̂) --------------------
    def _publish_map(self, t_now: float):
        if not self.particles:
            return
        idx_best = int(np.argmax([p.weight for p in self.particles]))
        p = self.particles[idx_best]

        out = ConeArrayWithCovariance()
        if self.last_cone_stamp is not None:
            out.header.stamp = self.last_cone_stamp
        else:
            out.header.stamp = rclpy.time.Time(seconds=t_now).to_msg()
        out.header.frame_id = "map"

        # Optionally realign best-particle landmarks to the Δ̂ frame so map matches odom output
        if self.map_align_hat:
            tx_b, ty_b, dth_b = p.delta
            tx_h, ty_h, dth_h = self.delta_hat
            Rb = rot2d(dth_b)
            Rh = rot2d(dth_h)
            R_rel = Rh @ Rb.T
            t_rel = np.array([tx_h, ty_h]) - R_rel @ np.array([tx_b, ty_b])

            def xform_mean_cov(m, S):
                m2 = R_rel @ m + t_rel
                S2 = R_rel @ S @ R_rel.T
                return m2, S2
        else:

            def xform_mean_cov(m, S):
                return m, S

        for lm in p.lms:
            if not (
                lm.retained
                or lm.promoted
                or lm.conf >= self.map_pub_conf
                or lm.distinct_hits >= self.map_pub_hits
            ):
                continue
            m_pub, S_pub = xform_mean_cov(lm.mean, lm.cov)

            c = ConeWithCovariance()
            c.point.x = float(m_pub[0])
            c.point.y = float(m_pub[1])
            c.point.z = 0.0
            c.covariance = [
                float(S_pub[0, 0]),
                float(S_pub[0, 1]),
                float(S_pub[1, 0]),
                float(S_pub[1, 1]),
            ]
            if lm.color == "blue":
                out.blue_cones.append(c)
            elif lm.color == "yellow":
                out.yellow_cones.append(c)
            elif lm.color == "orange":
                out.orange_cones.append(c)
            elif lm.color == "big":
                out.big_orange_cones.append(c)

        self.pub_map.publish(out)


# --------------------- main ---------------------
def main():
    rclpy.init()
    node = FastSLAMCore()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
