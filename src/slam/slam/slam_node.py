#!/usr/bin/env python3
import math
import bisect
import random
from dataclasses import dataclass
from collections import deque
from typing import List, Tuple, Optional

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

from .slam_utils.pf_slam import ParticleFilterSLAM, wrap


def yaw_from_quat(qx, qy, qz, qw) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def stamp_from_float_seconds(t: float):
    """Safe float seconds -> builtin_interfaces/Time"""
    sec = int(math.floor(t))
    nanosec = int((t - sec) * 1e9)
    return rclpy.time.Time(seconds=sec, nanoseconds=nanosec).to_msg()


@dataclass
class ConeTrack:
    x: float
    y: float
    cls_id: int
    last_seen: float
    hits: int = 1


class PFSlamNode(Node):
    """
    PF-SLAM node + snap-back correction.
    Also maintains a rolling LOCAL map window (tracks) with aging,
    so the system doesn't accumulate permanent corruption.

    Key behavior:
      - Maintain track list in 'map' frame (after output correction).
      - Update tracks by NN association from current observations projected into map.
      - Age out tracks not seen for local.max_age_sec seconds.
      - Publish map cones from tracks (optional gating by hits).
      - Use tracks (not PF landmarks) as the reference map for snap-back.
    """

    def __init__(self):
        super().__init__("pf_slam_node")

        # ---- Parameters ----
        self.declare_parameter("topics.odom_in", "/slam/odom_raw")
        self.declare_parameter("topics.cones_in", "/perception/cones_fused")
        self.declare_parameter("topics.slam_odom_out", "/slam/odom")
        self.declare_parameter("topics.map_cones_out", "/slam/map_cones")

        # QoS
        self.declare_parameter("qos.best_effort", True)
        self.declare_parameter("qos.depth", 3)
        self.declare_parameter("odom_buffer_sec", 5.0)

        # Drop stale cone frames (sim lag / low RTF protection)
        self.declare_parameter("cones.max_age_sec", 0.20)

        # Rolling local "map" window (track store)
        self.declare_parameter("local.enable", True)
        self.declare_parameter("local.max_age_sec", 5.0)
        self.declare_parameter("local.assoc_dist_m", 0.9)
        self.declare_parameter("local.ema_alpha", 0.35)
        self.declare_parameter("local.min_hits_publish", 2)

        # Snap-back loop closure
        self.declare_parameter("lc.enable", True)
        self.declare_parameter("lc.rate_hz", 3.0)
        self.declare_parameter("lc.inlier_dist_m", 0.8)
        self.declare_parameter("lc.min_inliers", 10)
        self.declare_parameter("lc.min_inlier_ratio", 0.35)
        self.declare_parameter("lc.ransac_iters", 80)
        self.declare_parameter("lc.max_yaw_deg", 12.0)
        self.declare_parameter("lc.max_trans_m", 2.0)
        self.declare_parameter("lc.min_map_cones", 20)   # lowered because we now use local tracks
        self.declare_parameter("lc.min_obs_cones", 8)

        gp = self.get_parameter
        self.odom_topic = str(gp("topics.odom_in").value)
        self.cones_topic = str(gp("topics.cones_in").value)
        self.slam_odom_topic = str(gp("topics.slam_odom_out").value)
        self.map_cones_topic = str(gp("topics.map_cones_out").value)

        best_effort = bool(gp("qos.best_effort").value)
        depth = int(gp("qos.depth").value)
        self.odom_buffer_sec = float(gp("odom_buffer_sec").value)
        self.cones_max_age = float(gp("cones.max_age_sec").value)

        # local tracks
        self.local_enable = bool(gp("local.enable").value)
        self.local_max_age = float(gp("local.max_age_sec").value)
        self.local_assoc_dist = float(gp("local.assoc_dist_m").value)
        self.local_ema_alpha = float(gp("local.ema_alpha").value)
        self.local_min_hits_publish = int(gp("local.min_hits_publish").value)

        # loop closure
        self.lc_enable = bool(gp("lc.enable").value)
        self.lc_rate_hz = float(gp("lc.rate_hz").value)
        self.lc_inlier_dist = float(gp("lc.inlier_dist_m").value)
        self.lc_min_inliers = int(gp("lc.min_inliers").value)
        self.lc_min_ratio = float(gp("lc.min_inlier_ratio").value)
        self.lc_ransac_iters = int(gp("lc.ransac_iters").value)
        self.lc_max_yaw = math.radians(float(gp("lc.max_yaw_deg").value))
        self.lc_max_trans = float(gp("lc.max_trans_m").value)
        self.lc_min_map_cones = int(gp("lc.min_map_cones").value)
        self.lc_min_obs_cones = int(gp("lc.min_obs_cones").value)

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
        self.create_subscription(Odometry, self.odom_topic, self.cb_odom, qos)
        self.create_subscription(PointCloud2, self.cones_topic, self.cb_cones, qos)

        # ---- Publishers ----
        self.pub_slam_odom = self.create_publisher(Odometry, self.slam_odom_topic, 5)
        self.pub_map_cones = self.create_publisher(ConeArrayWithCovariance, self.map_cones_topic, 5)

        # ---- State ----
        self.odom_buf: deque = deque()
        self.slam_initialised = False
        self.last_cone_odom_pose: Optional[Tuple[float, float, float]] = None

        # PF core
        self.pf = ParticleFilterSLAM(
            num_particles=80,
            process_std_xy=0.03,
            process_std_yaw=0.01,
            meas_sigma_xy=0.18,
            birth_sigma_xy=0.35,
            gate_prob=0.997,
            resample_neff_ratio=0.5,
        )

        # Cumulative correction (applied to outputs): p' = R p + t
        self.corr_R = np.eye(2, dtype=np.float64)
        self.corr_t = np.zeros((2,), dtype=np.float64)

        # loop closure scheduling
        self.last_lc_time = -1e9

        # rolling local tracks (map frame, corrected)
        self.tracks: List[ConeTrack] = []

        self.get_logger().info(
            f"[pf_slam] odom_in={self.odom_topic}, cones_in={self.cones_topic}, "
            f"slam_odom_out={self.slam_odom_topic}, map_cones_out={self.map_cones_topic}"
        )

    # ---------------------- Odom buffer helper ----------------------

    def _pose_at(self, t_query: float):
        if not self.odom_buf:
            return None

        times = [it[0] for it in self.odom_buf]
        idx = bisect.bisect_left(times, t_query)

        if idx == 0:
            _, x0, y0, yaw0 = self.odom_buf[0]
            return x0, y0, yaw0
        if idx >= len(self.odom_buf):
            _, x1, y1, yaw1 = self.odom_buf[-1]
            return x1, y1, yaw1

        t0, x0, y0, yaw0 = self.odom_buf[idx - 1]
        t1, x1, y1, yaw1 = self.odom_buf[idx]

        if t1 == t0:
            return x0, y0, yaw0

        a = (t_query - t0) / (t1 - t0)
        x = x0 + a * (x1 - x0)
        y = y0 + a * (y1 - y0)
        dyaw = ((yaw1 - yaw0 + math.pi) % (2.0 * math.pi)) - math.pi
        yaw = yaw0 + a * dyaw
        return x, y, wrap(yaw)

    # ---------------------- Correction helpers ----------------------

    def _apply_corr_to_point(self, x: float, y: float) -> Tuple[float, float]:
        p = np.array([x, y], dtype=np.float64)
        p2 = self.corr_R @ p + self.corr_t
        return float(p2[0]), float(p2[1])

    def _apply_corr_to_pose(self, x: float, y: float, yaw: float) -> Tuple[float, float, float]:
        x2, y2 = self._apply_corr_to_point(x, y)
        dyaw = math.atan2(self.corr_R[1, 0], self.corr_R[0, 0])
        return x2, y2, wrap(yaw + dyaw)

    def _compose_corr(self, R_new: np.ndarray, t_new: np.ndarray):
        # corr <- new ∘ corr
        self.corr_t = (R_new @ self.corr_t) + t_new
        self.corr_R = R_new @ self.corr_R

    # ---------------------- Rolling local tracks ----------------------

    def _prune_tracks(self, t_now: float):
        if not self.local_enable:
            return
        age = self.local_max_age
        self.tracks = [tr for tr in self.tracks if (t_now - tr.last_seen) <= age]

    def _update_tracks_from_obs_map(self, t_now: float, obs_map: List[Tuple[float, float, int]]):
        """
        obs_map: list of (mx, my, cls_id) in corrected map frame
        """
        if not self.local_enable:
            return

        self._prune_tracks(t_now)
        assoc2 = self.local_assoc_dist ** 2
        a = self.local_ema_alpha

        # Greedy NN association per observation
        for mx, my, cls_id in obs_map:
            best_i = None
            best_d2 = None
            for i, tr in enumerate(self.tracks):
                if tr.cls_id != cls_id:
                    continue
                dx = tr.x - mx
                dy = tr.y - my
                d2 = dx * dx + dy * dy
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best_i = i

            if best_i is not None and best_d2 is not None and best_d2 <= assoc2:
                tr = self.tracks[best_i]
                tr.x = (1.0 - a) * tr.x + a * mx
                tr.y = (1.0 - a) * tr.y + a * my
                tr.last_seen = t_now
                tr.hits += 1
            else:
                self.tracks.append(ConeTrack(x=mx, y=my, cls_id=cls_id, last_seen=t_now, hits=1))

    # ---------------------- Loop closure / snap-back ----------------------

    def _build_map_points(self):
        """
        Reference map for snap-back:
          Prefer local tracks (rolling window).
          Fallback to PF best-particle landmarks if tracks are disabled/empty.
        """
        pts_by_cls = {}

        if self.local_enable and self.tracks:
            for tr in self.tracks:
                pts_by_cls.setdefault(int(tr.cls_id), []).append((float(tr.x), float(tr.y)))
            return pts_by_cls

        best = self.pf.get_best_particle()
        if best is None:
            return {}

        for lm in best.landmarks:
            cls_id = int(lm.cls)
            x = float(lm.mean[0])
            y = float(lm.mean[1])
            x, y = self._apply_corr_to_point(x, y)
            pts_by_cls.setdefault(cls_id, []).append((x, y))
        return pts_by_cls

    def _obs_body_to_map(self, x: float, y: float, yaw: float, meas_body):
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        out = []
        for bx, by, cls_id in meas_body:
            mx = x + cy * bx - sy * by
            my = y + sy * bx + cy * by
            mx, my = self._apply_corr_to_point(mx, my)
            out.append((mx, my, int(cls_id)))
        return out

    def _nn_inliers(self, obs_map, map_by_cls, dist_thr: float) -> int:
        thr2 = dist_thr * dist_thr
        inl = 0
        for ox, oy, cls_id in obs_map:
            cand = map_by_cls.get(cls_id, [])
            if not cand:
                continue
            best2 = None
            for mx, my in cand:
                dx = mx - ox
                dy = my - oy
                d2 = dx * dx + dy * dy
                if best2 is None or d2 < best2:
                    best2 = d2
            if best2 is not None and best2 <= thr2:
                inl += 1
        return inl

    def _estimate_se2_from_pairs(self, p1, p2, q1, q2):
        p1 = np.array(p1, dtype=np.float64)
        p2 = np.array(p2, dtype=np.float64)
        q1 = np.array(q1, dtype=np.float64)
        q2 = np.array(q2, dtype=np.float64)
        vp = p2 - p1
        vq = q2 - q1
        nvp = np.linalg.norm(vp)
        nvq = np.linalg.norm(vq)
        if nvp < 1e-6 or nvq < 1e-6:
            return None
        ap = math.atan2(vp[1], vp[0])
        aq = math.atan2(vq[1], vq[0])
        dth = wrap(aq - ap)
        c = math.cos(dth)
        s = math.sin(dth)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        t = q1 - (R @ p1)
        return R, t, dth

    def _try_snap_back(self, t_now: float, pose_x: float, pose_y: float, pose_yaw: float, meas_body):
        if not self.lc_enable:
            return
        if (t_now - self.last_lc_time) < (1.0 / max(self.lc_rate_hz, 1e-6)):
            return

        map_by_cls = self._build_map_points()
        map_count = sum(len(v) for v in map_by_cls.values())
        if map_count < self.lc_min_map_cones:
            return
        if len(meas_body) < self.lc_min_obs_cones:
            return

        obs_map = self._obs_body_to_map(pose_x, pose_y, pose_yaw, meas_body)

        base_inliers = self._nn_inliers(obs_map, map_by_cls, self.lc_inlier_dist)

        obs_by_cls = {}
        for ox, oy, cls_id in obs_map:
            obs_by_cls.setdefault(cls_id, []).append((ox, oy))
        valid_classes = [c for c in obs_by_cls if len(obs_by_cls[c]) >= 2 and len(map_by_cls.get(c, [])) >= 2]
        if not valid_classes:
            return

        best = None  # (inliers, R, t, dth)
        thr2 = self.lc_inlier_dist * self.lc_inlier_dist

        def nn(pt, candidates):
            px, py = pt
            best_i = None
            best_d2 = None
            for i, (cx, cy) in enumerate(candidates):
                dx = cx - px
                dy = cy - py
                d2 = dx * dx + dy * dy
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best_i = i
            return candidates[best_i]

        for _ in range(self.lc_ransac_iters):
            cls = random.choice(valid_classes)

            p1, p2 = random.sample(obs_by_cls[cls], 2)
            q1 = nn(p1, map_by_cls[cls])
            q2 = nn(p2, map_by_cls[cls])
            if q1 == q2:
                continue

            est = self._estimate_se2_from_pairs(p1, p2, q1, q2)
            if est is None:
                continue
            R, t, dth = est

            if abs(dth) > self.lc_max_yaw:
                continue
            if float(np.linalg.norm(t)) > self.lc_max_trans:
                continue

            inl = 0
            for ox, oy, cls_id in obs_map:
                p = np.array([ox, oy], dtype=np.float64)
                p2t = R @ p + t
                cand = map_by_cls.get(cls_id, [])
                if not cand:
                    continue
                best2 = None
                for mx, my in cand:
                    dx = mx - p2t[0]
                    dy = my - p2t[1]
                    d2 = dx * dx + dy * dy
                    if best2 is None or d2 < best2:
                        best2 = d2
                if best2 is not None and best2 <= thr2:
                    inl += 1

            if best is None or inl > best[0]:
                best = (inl, R, t, dth)

        if best is None:
            return

        inliers, R_best, t_best, dth_best = best
        ratio = inliers / max(len(obs_map), 1)

        if inliers < self.lc_min_inliers:
            return
        if ratio < self.lc_min_ratio:
            return
        if inliers < (base_inliers + 2):
            return

        self._compose_corr(R_best, t_best)
        self.last_lc_time = t_now
        self.get_logger().info(
            f"[lc] snap-back: inliers={inliers}/{len(obs_map)} ({ratio:.2f}) "
            f"dYaw={math.degrees(dth_best):.2f}deg |t|={float(np.linalg.norm(t_best)):.2f}m",
            throttle_duration_sec=0.5
        )

    # ---------------------- Callbacks ----------------------

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

    def cb_cones(self, msg: PointCloud2):
        # drop stale frames if sim is lagging
        t_c = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        t_now = self.get_clock().now().nanoseconds * 1e-9

        if (t_now - t_c) > 0.2:
            return
        if (t_now - t_c) > self.cones_max_age:
            return

        # parse cones
        try:
            meas_body = [(float(bx), float(by), int(cls_id))
                         for (bx, by, _bz, cls_id) in pc2.read_points(
                             msg, field_names=("x", "y", "z", "class_id"), skip_nans=True)]
        except Exception as e:
            self.get_logger().warn(f"[pf_slam] Error parsing cones PointCloud2: {e}")
            return

        if not meas_body:
            # still age out local tracks over time
            self._prune_tracks(t_c)
            self._publish_map_cones(t_c)
            return

        pose_now = self._pose_at(t_c)
        if pose_now is None:
            return
        x_o, y_o, yaw_o = pose_now

        # --------- PF motion (BODY-FRAME increments) ---------
        if not self.slam_initialised or self.last_cone_odom_pose is None:
            self.pf.init_pose(x_o, y_o, yaw_o)
            self.slam_initialised = True
            self.last_cone_odom_pose = (x_o, y_o, yaw_o)
        else:
            x_prev, y_prev, yaw_prev = self.last_cone_odom_pose

            dx_map = x_o - x_prev
            dy_map = y_o - y_prev
            dyaw = wrap(yaw_o - yaw_prev)

            # map -> body using previous yaw
            c = math.cos(-yaw_prev)
            s = math.sin(-yaw_prev)
            dx_body = c * dx_map - s * dy_map
            dy_body = s * dx_map + c * dy_map

            self.pf.predict((dx_body, dy_body, dyaw))
            self.last_cone_odom_pose = (x_o, y_o, yaw_o)

        # --------- PF measurement update ---------
        self.pf.update(meas_body)

        best = self.pf.get_best_particle()
        if best is None:
            return

        # corrected pose BEFORE snap-back attempt
        x_pub, y_pub, yaw_pub = self._apply_corr_to_pose(float(best.x), float(best.y), float(best.yaw))

        # Build obs in corrected map frame using corrected pose
        obs_map = self._obs_body_to_map(x_pub, y_pub, yaw_pub, meas_body)

        # Update rolling local tracks (association + 5s aging)
        self._update_tracks_from_obs_map(t_c, obs_map)

        # snap-back: align obs to local tracks (or PF map fallback)
        self._try_snap_back(t_c, x_pub, y_pub, yaw_pub, meas_body)

        # corrected pose AFTER snap-back (corr may have changed)
        x_pub, y_pub, yaw_pub = self._apply_corr_to_pose(float(best.x), float(best.y), float(best.yaw))

        # publish corrected odom + rolling map
        self._publish_slam_odom(t_c, x_pub, y_pub, yaw_pub)
        self._publish_map_cones(t_c)

    # ---------------------- Publishers ----------------------

    def _publish_slam_odom(self, t: float, x: float, y: float, yaw: float):
        od = Odometry()
        od.header.stamp = stamp_from_float_seconds(t)
        od.header.frame_id = "map"
        od.child_frame_id = "base_link"

        od.pose.pose.position.x = float(x)
        od.pose.pose.position.y = float(y)
        od.pose.pose.position.z = 0.0

        half_yaw = 0.5 * float(yaw)
        od.pose.pose.orientation.x = 0.0
        od.pose.pose.orientation.y = 0.0
        od.pose.pose.orientation.z = math.sin(half_yaw)
        od.pose.pose.orientation.w = math.cos(half_yaw)

        self.pub_slam_odom.publish(od)

    def _publish_map_cones(self, t: float):
        msg = ConeArrayWithCovariance()
        msg.header.stamp = stamp_from_float_seconds(t)
        msg.header.frame_id = "map"

        if self.local_enable and self.tracks:
            # publish local tracks only
            for tr in self.tracks:
                if tr.hits < self.local_min_hits_publish:
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
            return

        # fallback: publish PF best particle landmarks (corrected)
        best = self.pf.get_best_particle()
        if best is None:
            return

        for lm in best.landmarks:
            x = float(lm.mean[0])
            y = float(lm.mean[1])
            cls_id = int(lm.cls)

            x, y = self._apply_corr_to_point(x, y)

            cone = ConeWithCovariance()
            cone.point.x = x
            cone.point.y = y
            cone.point.z = 0.0

            if cls_id == 0:
                msg.blue_cones.append(cone)
            elif cls_id == 1:
                msg.yellow_cones.append(cone)
            elif cls_id == 2:
                msg.orange_cones.append(cone)
            elif cls_id == 3:
                msg.big_orange_cones.append(cone)
            else:
                try:
                    msg.unknown_color_cones.append(cone)
                except AttributeError:
                    pass

        self.pub_map_cones.publish(msg)


def main():
    rclpy.init()
    node = PFSlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
