#!/usr/bin/env python3
import math
import threading
from collections import deque
import bisect

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
from matplotlib.patches import Circle

from .slam_utils import association   # <- NEW


def yaw_from_quat(qx, qy, qz, qw) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def rot2d(th: float) -> np.ndarray:
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]], dtype=float)


class VirtualWorldPlotter(Node):
    """
    'Dumb SLAM' visualiser with:
      - Time-synchronised odom + cones
      - Cones in base_link frame -> transformed to map
      - Mahalanobis gating with yaw-rate-dependent sigma
      - Hungarian association per colour (1-to-1)
      - Landmark update (simple EMA) + new landmark creation
      - Gate visualisation
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
        self.declare_parameter("plot.cone_max_points_per_class", 20000)
        self.declare_parameter("plot.fixed_window", False)
        self.declare_parameter("plot.window_half_size", 60.0)
        self.declare_parameter("plot.show_gates", True)

        # Gating params (Mahalanobis in 2D, dynamic sigma on yaw-rate)
        self.declare_parameter("gate.base_sigma", 0.4)   # m
        self.declare_parameter("gate.max_sigma", 1.5)    # m
        self.declare_parameter("gate.yaw_ref", 0.6)      # rad/s where we reach max_sigma
        self.declare_parameter("gate.chi2", 9.21)        # ~99% 2D (chi-square with 2 dof)

        # Association params
        self.declare_parameter("assoc.large_cost", 1e6)
        self.declare_parameter("assoc.update_alpha", 0.2)  # EMA blend weight for measurement

        P = lambda k: self.get_parameter(k).value

        odom_topic = str(P("topics.odom_in"))
        cones_topic = str(P("topics.cones_in"))

        best_effort = bool(P("qos.best_effort"))
        depth = int(P("qos.depth"))
        self.odom_buffer_sec = float(P("odom_buffer_sec"))

        self.trail_max_points = int(P("plot.trail_max_points"))
        self.cone_max_points = int(P("plot.cone_max_points_per_class"))
        self.fixed_window = bool(P("plot.fixed_window"))
        self.window_half_size = float(P("plot.window_half_size"))
        self.show_gates = bool(P("plot.show_gates"))

        self.gate_base_sigma = float(P("gate.base_sigma"))
        self.gate_max_sigma = float(P("gate.max_sigma"))
        self.gate_yaw_ref = float(P("gate.yaw_ref"))
        self.gate_chi2 = float(P("gate.chi2"))

        self.large_cost = float(P("assoc.large_cost"))
        self.landmark_update_alpha = float(P("assoc.update_alpha"))

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

        # odom buffer: (t, x, y, yaw)
        self.odom_buf: deque = deque()

        # car pose & trajectory (for drawing)
        self.car_x = None
        self.car_y = None
        self.car_yaw = 0.0
        self.car_yaw_rate = 0.0  # for dynamic gate
        self.traj_x = []
        self.traj_y = []

        # cones in map frame per class_id (landmarks)
        # class_id -> list[(x, y)]
        self.map_cones = {}

        # gating visualisation
        self.last_gate_radius = None  # m

        # ---- Plot ----
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.grid(True)
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_title("Virtual World (odom + cones + Hungarian association)")

        self.get_logger().info(
            f"[virtual_world_plotter] odom={odom_topic}, cones={cones_topic}"
        )

    # ---------------------- Odom buffer helper ----------------------

    def _pose_at(self, t_query: float):
        """
        Interpolate pose (x,y,yaw) at time t_query using odom_buf.
        Returns (x,y,yaw) or None if no odom yet.
        """
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
        dyaw = ((yaw1 - yaw0 + math.pi) % (2 * math.pi)) - math.pi
        yaw = yaw0 + a * dyaw
        return x, y, yaw

    def _effective_gate_sigma(self) -> float:
        """
        Sigma in metres for x,y, scaled up with yaw-rate.
        """
        base = self.gate_base_sigma
        max_s = self.gate_max_sigma
        yaw_ref = max(self.gate_yaw_ref, 1e-3)

        yr = abs(self.car_yaw_rate)
        f = min(1.0, yr / yaw_ref)
        return base + f * (max_s - base)

    # ---------------------- Callbacks ----------------------

    def cb_odom(self, msg: Odometry):
        t = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

        # yaw-rate from twist if present
        yaw_rate = float(msg.twist.twist.angular.z)

        with self.data_lock:
            # push into buffer
            self.odom_buf.append((t, x, y, yaw))
            tmin = t - self.odom_buffer_sec
            while self.odom_buf and self.odom_buf[0][0] < tmin:
                self.odom_buf.popleft()

            # for path we just trail the raw odom
            self.car_x = x
            self.car_y = y
            self.car_yaw = yaw
            self.car_yaw_rate = yaw_rate

            self.traj_x.append(x)
            self.traj_y.append(y)
            if len(self.traj_x) > self.trail_max_points:
                self.traj_x = self.traj_x[-self.trail_max_points:]
                self.traj_y = self.traj_y[-self.trail_max_points:]

    def cb_cones(self, msg: PointCloud2):
        # read cones in base_link frame
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

        t_c = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9

        with self.data_lock:
            pose = self._pose_at(t_c)
            if pose is None:
                # No odom yet; can't transform
                return

            x_w, y_w, yaw = pose
            R = rot2d(yaw)

            # Group current measurements by class_id
            meas_by_class = {}
            for bx, by, bz, cls_id in cones:
                pb = np.array([float(bx), float(by)], dtype=float)
                pw = np.array([x_w, y_w], dtype=float) + R @ pb
                cid = int(cls_id)
                meas_by_class.setdefault(cid, []).append(pw)

            if not meas_by_class:
                return

            # Setup gating for this frame
            sigma_eff = self._effective_gate_sigma()
            self.last_gate_radius = math.sqrt(self.gate_chi2) * sigma_eff

            # Per-class Hungarian association in world coordinates
            for cid, pts in meas_by_class.items():
                meas_xy = np.vstack(pts)  # shape (M, 2)

                old_pts = self.map_cones.get(cid, [])
                if not old_pts:
                    # No landmarks of this class yet -> all become new landmarks
                    self.map_cones[cid] = [
                        (float(p[0]), float(p[1])) for p in meas_xy
                    ]
                    continue

                landmarks_xy = np.asarray(old_pts, dtype=float).reshape(-1, 2)

                cost, valid = association.build_mahalanobis_cost_matrix(
                    landmarks_xy=landmarks_xy,
                    meas_xy=meas_xy,
                    sigma_xy=sigma_eff,
                    chi2_gate=self.gate_chi2,
                    large_cost=self.large_cost,
                )

                matches, unmatched_meas, unmatched_landmarks = association.hungarian_assign(
                    cost=cost,
                    valid_mask=valid,
                    large_cost=self.large_cost,
                )

                new_landmarks = []

                # Matched pairs -> update landmark with EMA toward measurement
                alpha = self.landmark_update_alpha
                for mi, lj, d in matches:
                    old = landmarks_xy[lj]
                    meas = meas_xy[mi]
                    upd = (1.0 - alpha) * old + alpha * meas
                    new_landmarks.append((float(upd[0]), float(upd[1])))

                # Unmatched old landmarks -> keep as-is
                for lj in unmatched_landmarks:
                    old = landmarks_xy[lj]
                    new_landmarks.append((float(old[0]), float(old[1])))

                # Unmatched measurements -> spawn new landmarks
                for mi in unmatched_meas:
                    meas = meas_xy[mi]
                    new_landmarks.append((float(meas[0]), float(meas[1])))

                # Limit count per class if desired
                if len(new_landmarks) > self.cone_max_points:
                    new_landmarks = new_landmarks[-self.cone_max_points:]

                self.map_cones[cid] = new_landmarks

    # ---------------------- Plot update ----------------------

    def update_plot(self):
        with self.data_lock:
            traj_x = list(self.traj_x)
            traj_y = list(self.traj_y)
            car_x = self.car_x
            car_y = self.car_y
            car_yaw = self.car_yaw
            cones_copy = {cid: list(pts) for cid, pts in self.map_cones.items()}
            gate_radius = self.last_gate_radius

        self.ax.clear()
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.grid(True)
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_title("Virtual World (odom + cones + Hungarian association)")

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
            4: ("gray", "unknown cones"),
        }

        for cid, pts in cones_copy.items():
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

            # Draw gates as circles if enabled
            if self.show_gates and gate_radius is not None and gate_radius > 0.0:
                r = gate_radius
                for (x, y) in pts:
                    circ = Circle(
                        (x, y),
                        radius=r,
                        fill=False,
                        linestyle="--",
                        linewidth=0.5,
                        alpha=0.25,
                    )
                    self.ax.add_patch(circ)

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
