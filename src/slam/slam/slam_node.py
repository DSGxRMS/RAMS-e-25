#!/usr/bin/env python3
import math
import threading
from collections import deque
import bisect
from typing import Dict, List

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

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Import gating helpers
from .slam_utils.gating import (
    Landmark as GateLandmark,
    GateConfig,
    gate_measurements,
)


def yaw_from_quat(qx, qy, qz, qw) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def rot2d(th: float) -> np.ndarray:
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]], dtype=float)


class VirtualWorldPlotter(Node):
    """
    Visualiser + "dumb SLAM" with Mahalanobis gating and yaw-rate–dependent ellipse.

    - Odom: /slam/odom_raw or /ground_truth/odom (pose in MAP frame).
    - Cones: /perception/cones_fused in BASE frame (x,y,z,class_id).

    Logic:
      1) Use odom buffer to get pose (x,y,yaw,yawrate) at the cones' timestamp.
      2) Transform cones from BASE -> MAP.
      3) For each measurement frame:
           - Use Mahalanobis gating against existing landmarks.
           - Measurements outside gate => new landmarks.
      4) Plot trajectory + landmarks + gate ellipses.
    """

    def __init__(self):
        super().__init__("virtual_world_plotter")

        # ---- Parameters ----
        self.declare_parameter("topics.odom_in", "/slam/odom_raw")
        self.declare_parameter("topics.cones_in", "/perception/cones_fused")

        self.declare_parameter("qos.best_effort", True)
        self.declare_parameter("qos.depth", 200)
        self.declare_parameter("odom_buffer_sec", 5.0)

        # Plot params
        self.declare_parameter("plot.trail_max_points", 20000)
        self.declare_parameter("plot.fixed_window", False)
        self.declare_parameter("plot.window_half_size", 60.0)
        self.declare_parameter("plot.show_gates", True)

        gp = self.get_parameter
        odom_topic = str(gp("topics.odom_in").value)
        cones_topic = str(gp("topics.cones_in").value)

        best_effort = bool(gp("qos.best_effort").value)
        depth = int(gp("qos.depth").value)
        self.odom_buffer_sec = float(gp("odom_buffer_sec").value)

        self.trail_max_points = int(gp("plot.trail_max_points").value)
        self.fixed_window = bool(gp("plot.fixed_window").value)
        self.window_half_size = float(gp("plot.window_half_size").value)
        self.show_gates = bool(gp("plot.show_gates").value)

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
        self.create_subscription(PointCloud2, cones_topic, self.cb_cones, qos)

        # ---- State ----
        self.data_lock = threading.Lock()

        # odom buffer: (t, x, y, yaw, yawrate)
        self.odom_buf: deque = deque()

        # car pose & trajectory (for plotting)
        self.car_x = None
        self.car_y = None
        self.car_yaw = 0.0
        self.car_yawrate = 0.0
        self.traj_x = []
        self.traj_y = []

        # landmarks per class_id: cid -> List[GateLandmark]
        self.landmarks_by_class: Dict[int, List[GateLandmark]] = {}

        # Gating config (tuned for your use case)
        self.gate_cfg = GateConfig(
            chi2_thr=9.21,
            sigma_along_base=0.25,
            sigma_across_base=0.15,
            k_yawrate=2.5,
            scale_min=1.0,
            scale_max=2.5,
        )

        # ---- Plot ----
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.grid(True)
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_title("Virtual World (odom + Mahalanobis-gated map)")

        self.get_logger().info(
            f"[virtual_world_plotter] odom={odom_topic}, cones={cones_topic}"
        )

    # ---------------------- Odom buffer helper ----------------------

    def _pose_at(self, t_query: float):
        """
        Interpolate pose (x,y,yaw,yawrate) at time t_query using odom_buf.
        Returns tuple or None if no odom yet.
        """
        if not self.odom_buf:
            return None

        times = [it[0] for it in self.odom_buf]
        idx = bisect.bisect_left(times, t_query)

        if idx == 0:
            _, x0, y0, yaw0, yr0 = self.odom_buf[0]
            return x0, y0, yaw0, yr0
        if idx >= len(self.odom_buf):
            _, x1, y1, yaw1, yr1 = self.odom_buf[-1]
            return x1, y1, yaw1, yr1

        t0, x0, y0, yaw0, yr0 = self.odom_buf[idx - 1]
        t1, x1, y1, yaw1, yr1 = self.odom_buf[idx]

        if t1 == t0:
            return x0, y0, yaw0, yr0

        a = (t_query - t0) / (t1 - t0)
        x = x0 + a * (x1 - x0)
        y = y0 + a * (y1 - y0)

        # interpolate yaw shortest-arc
        dyaw = ((yaw1 - yaw0 + math.pi) % (2 * math.pi)) - math.pi
        yaw = yaw0 + a * dyaw

        # linear interp yawrate
        yr = yr0 + a * (yr1 - yr0)
        return x, y, yaw, yr

    # ---------------------- Callbacks ----------------------

    def cb_odom(self, msg: Odometry):
        t = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

        # Try to get yawrate from twist; if missing, treat as 0
        try:
            yr = float(msg.twist.twist.angular.z)
        except Exception:
            yr = 0.0

        with self.data_lock:
            # push into buffer
            self.odom_buf.append((t, x, y, yaw, yr))
            tmin = t - self.odom_buffer_sec
            while self.odom_buf and self.odom_buf[0][0] < tmin:
                self.odom_buf.popleft()

            # for path we just trail the raw odom
            self.car_x = x
            self.car_y = y
            self.car_yaw = yaw
            self.car_yawrate = yr
            self.traj_x.append(x)
            self.traj_y.append(y)
            if len(self.traj_x) > self.trail_max_points:
                self.traj_x = self.traj_x[-self.trail_max_points:]
                self.traj_y = self.traj_y[-self.trail_max_points:]

    def cb_cones(self, msg: PointCloud2):
        # read cones in BASE frame
        try:
            cones = list(
                pc2.read_points(
                    msg,
                    field_names=("x", "y", "z", "class_id"),
                    skip_nans=True,
                )
            )
        except Exception as e:
            self.get_logger().warn(
                f"[virtual_world_plotter] Error parsing cones PointCloud2: {e}"
            )
            return

        if not cones:
            return

        # get pose at cones timestamp
        t_c = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9

        with self.data_lock:
            pose = self._pose_at(t_c)
            if pose is None:
                return
            x_w, y_w, yaw, yawrate = pose
            R = rot2d(yaw)

            # group measurements by class_id, in MAP frame
            meas_by_class: Dict[int, List[np.ndarray]] = {}
            for bx, by, bz, cls_id in cones:
                pb = np.array([float(bx), float(by)], dtype=float)
                pw = np.array([x_w, y_w], dtype=float) + R @ pb
                cid = int(cls_id)
                meas_by_class.setdefault(cid, []).append(pw)

            # For each class, run Mahalanobis gating against existing landmarks
            for cid, meas_list in meas_by_class.items():
                if not meas_list:
                    continue

                lms = self.landmarks_by_class.get(cid, [])

                # Gate: which meas are "existing" vs "new"
                associated_idxs, new_idxs = gate_measurements(
                    meas_list, lms, yaw, yawrate, self.gate_cfg
                )

                # Very simple nudging for existing landmarks
                if associated_idxs:
                    S_meas = self._approx_meas_cov_cached(yaw, yawrate)
                    for i_meas in associated_idxs:
                        z = meas_list[i_meas]
                        # Pick nearest lm (in Mahalanobis) and pull it slightly towards z
                        best_j = None
                        best_m2 = None
                        for j, lm in enumerate(lms):
                            innov = z - lm.mean
                            try:
                                Sinv = np.linalg.inv(lm.cov + S_meas)
                            except np.linalg.LinAlgError:
                                Sinv = np.linalg.pinv(lm.cov + S_meas)
                            m2 = float(innov.T @ Sinv @ innov)
                            if best_m2 is None or m2 < best_m2:
                                best_m2 = m2
                                best_j = j
                        if best_j is not None:
                            lm = lms[best_j]
                            lm.mean = 0.8 * lm.mean + 0.2 * z

                # Birth new landmarks
                if new_idxs:
                    S_meas = self._approx_meas_cov_cached(yaw, yawrate)
                    for i_meas in new_idxs:
                        z = meas_list[i_meas]
                        lm = GateLandmark(
                            mean=z.copy(),
                            cov=S_meas.copy(),
                            class_id=cid,
                        )
                        lms.append(lm)

                self.landmarks_by_class[cid] = lms

    def _approx_meas_cov_cached(self, yaw: float, yawrate: float) -> np.ndarray:
        """
        Wrapper to reuse the same logic as gating for local EKF-style updates.
        We just rebuild it here; if you want you can cache per-frame.
        """
        from .slam_utils.gating import _build_meas_cov  # type: ignore
        return _build_meas_cov(yaw, yawrate, self.gate_cfg)

    # ---------------------- Plot update ----------------------

    def update_plot(self):
        with self.data_lock:
            traj_x = list(self.traj_x)
            traj_y = list(self.traj_y)
            car_x = self.car_x
            car_y = self.car_y
            car_yaw = self.car_yaw
            car_yawrate = self.car_yawrate

            # flatten landmarks into per-class XY lists
            cones_by_class: Dict[int, List[tuple]] = {}
            for cid, lms in self.landmarks_by_class.items():
                cones_by_class[cid] = [
                    (float(lm.mean[0]), float(lm.mean[1])) for lm in lms
                ]

            # keep a copy of landmarks for gate drawing
            lms_copy = {
                cid: list(lms) for cid, lms in self.landmarks_by_class.items()
            }

        self.ax.clear()
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.grid(True)
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_title("Virtual World (odom + Mahalanobis-gated map)")

        all_x, all_y = [], []

        # Trajectory
        if traj_x and traj_y:
            self.ax.plot(traj_x, traj_y, "-", linewidth=1.0,
                         color="black", label="odom path")
            all_x.extend(traj_x)
            all_y.extend(traj_y)

        # Car pose
        if car_x is not None and car_y is not None:
            self.ax.plot(car_x, car_y, "ko", markersize=5, label="car")
            L = 2.0
            hx = car_x + L * math.cos(car_yaw)
            hy = car_y + L * math.sin(car_yaw)
            self.ax.arrow(
                car_x, car_y,
                hx - car_x, hy - car_y,
                head_width=0.7, head_length=1.0,
                length_includes_head=True,
            )
            all_x.append(car_x)
            all_y.append(car_y)

        # Cones per class_id
        color_map = {
            0: ("blue", "blue cones"),
            1: ("gold", "yellow cones"),
            2: ("orange", "orange cones"),
            3: ("red", "big orange cones"),
        }

        for cid, pts in cones_by_class.items():
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            all_x.extend(xs)
            all_y.extend(ys)

            if cid in color_map:
                col, label = color_map[cid]
            else:
                col, label = ("gray", f"class {cid}")

            self.ax.scatter(xs, ys, s=10.0, c=col, marker="o", label=label)

        # ---- Gate visualisation (ellipses) ----
        if self.show_gates and car_x is not None and car_y is not None:
            try:
                S_meas = self._approx_meas_cov_cached(car_yaw, car_yawrate)
            except Exception:
                S_meas = None

            if S_meas is not None:
                chi2 = self.gate_cfg.chi2_thr
                for cid, lms in lms_copy.items():
                    if cid in color_map:
                        col, _ = color_map[cid]
                    else:
                        col = "gray"

                    for lm in lms:
                        # Effective gate covariance: landmark cov + measurement cov
                        S_gate = lm.cov + S_meas
                        try:
                            w, V = np.linalg.eigh(S_gate)
                        except np.linalg.LinAlgError:
                            continue

                        # eigenvalues -> axis lengths (1-σ), then scale by sqrt(chi2_thr)
                        w = np.maximum(w, 1e-9)
                        axis_len = np.sqrt(w * chi2)  # [a, b] in meters

                        # convert to width/height for Ellipse (diameters)
                        width = 2.0 * axis_len[0]
                        height = 2.0 * axis_len[1]

                        # angle from first eigenvector
                        vx, vy = V[0, 0], V[1, 0]
                        angle = math.degrees(math.atan2(vy, vx))

                        e = Ellipse(
                            xy=(lm.mean[0], lm.mean[1]),
                            width=width,
                            height=height,
                            angle=angle,
                            fill=False,
                            linestyle="--",
                            linewidth=0.7,
                            edgecolor=col,
                            alpha=0.7,
                        )
                        self.ax.add_patch(e)

        # Windowing
        if self.fixed_window and (car_x is not None and car_y is not None):
            L = self.window_half_size
            self.ax.set_xlim(car_x - L, car_x + L)
            self.ax.set_ylim(car_y - L, car_y + L)
        else:
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
            self.ax.legend(uniq.values(), uniq.keys(),
                           loc="upper right", fontsize=8)


def main():
    rclpy.init()
    node = VirtualWorldPlotter()
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
