#!/usr/bin/env python3
import math
import threading
from collections import deque
import bisect
from typing import Dict, List, Tuple

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

from .slam_utils.pf_slam import ParticleFilterSLAM, wrap


def yaw_from_quat(qx, qy, qz, qw) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class VirtualWorldPlotter(Node):
    """
    PF-SLAM visualiser:

      - Odom: /slam/odom_raw or /ground_truth/odom
      - Cones: /perception/cones_fused (x,y,z,class_id) in BASE/BODY frame

    SLAM logic:
      - Interpolate odom pose at each cone stamp.
      - Use odom increment between successive cone frames as motion input.
      - Run ParticleFilterSLAM (motion + update).
      - Plot best-particle pose + map, alongside raw odom path.
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

        gp = self.get_parameter
        odom_topic = str(gp("topics.odom_in").value)
        cones_topic = str(gp("topics.cones_in").value)
        best_effort = bool(gp("qos.best_effort").value)
        depth = int(gp("qos.depth").value)
        self.odom_buffer_sec = float(gp("odom_buffer_sec").value)

        self.trail_max_points = int(gp("plot.trail_max_points").value)
        self.fixed_window = bool(gp("plot.fixed_window").value)
        self.window_half_size = float(gp("plot.window_half_size").value)

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

        # raw odom path
        self.odom_traj_x: List[float] = []
        self.odom_traj_y: List[float] = []
        self.odom_car_x: float = None
        self.odom_car_y: float = None
        self.odom_car_yaw: float = 0.0

        # SLAM: PF core
        self.pf = ParticleFilterSLAM(
            num_particles=80,
            process_std_xy=0.03,
            process_std_yaw=0.01,
            meas_sigma_xy=0.20,
            birth_sigma_xy=0.40,
            gate_prob=0.997,
            resample_neff_ratio=0.5,
            # persistence defaults are fine for now; tune later if needed
        )

        self.slam_initialised = False
        self.last_cone_odom_pose: Tuple[float, float, float] = None

        # SLAM best-particle trajectory
        self.slam_traj_x: List[float] = []
        self.slam_traj_y: List[float] = []

        # SLAM landmarks from best particle (for plotting only)
        self.slam_landmarks: List[Tuple[float, float, int]] = []

        # ---- Plot ----
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.grid(True)
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_title("PF-SLAM: odom vs SLAM map")

        self.get_logger().info(
            f"[virtual_world_plotter] odom={odom_topic}, cones={cones_topic}"
        )

    # ---------------------- Odom buffer helper ----------------------

    def _pose_at(self, t_query: float):
        """
        Interpolate odom pose (x,y,yaw) at time t_query using odom_buf.
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
        dyaw = ((yaw1 - yaw0 + math.pi) % (2.0 * math.pi)) - math.pi
        yaw = yaw0 + a * dyaw
        return x, y, wrap(yaw)

    # ---------------------- Callbacks ----------------------

    def cb_odom(self, msg: Odometry):
        t = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

        with self.data_lock:
            # push into buffer
            self.odom_buf.append((t, x, y, yaw))
            tmin = t - self.odom_buffer_sec
            while self.odom_buf and self.odom_buf[0][0] < tmin:
                self.odom_buf.popleft()

            # track raw odom path
            self.odom_car_x = x
            self.odom_car_y = y
            self.odom_car_yaw = yaw
            self.odom_traj_x.append(x)
            self.odom_traj_y.append(y)
            if len(self.odom_traj_x) > self.trail_max_points:
                self.odom_traj_x = self.odom_traj_x[-self.trail_max_points:]
                self.odom_traj_y = self.odom_traj_y[-self.trail_max_points:]

    def cb_cones(self, msg: PointCloud2):
        # read cones in BASE/BODY frame (aligned with odom base)
        try:
            cones_body = list(
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

        if not cones_body:
            return

        t_c = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9

        with self.data_lock:
            pose_now = self._pose_at(t_c)
            if pose_now is None:
                # no odom yet
                return
            x_o, y_o, yaw_o = pose_now

            # --------- PF motion: odom increment between cone frames ---------
            if not self.slam_initialised or self.last_cone_odom_pose is None:
                # first time: initialise PF at odom pose
                self.pf.init_pose(x_o, y_o, yaw_o)
                self.slam_initialised = True
                self.last_cone_odom_pose = (x_o, y_o, yaw_o)
            else:
                x_prev, y_prev, yaw_prev = self.last_cone_odom_pose
                dx = x_o - x_prev
                dy = y_o - y_prev
                dyaw = wrap(yaw_o - yaw_prev)
                self.pf.predict((dx, dy, dyaw))
                self.last_cone_odom_pose = (x_o, y_o, yaw_o)

            # --------- PF measurement update ---------
            meas_body = [
                (float(bx), float(by), int(cls_id))
                for (bx, by, _bz, cls_id) in cones_body
            ]
            self.pf.update(meas_body)

            best = self.pf.get_best_particle()
            if best is None:
                return

            # SLAM trajectory
            self.slam_traj_x.append(best.x)
            self.slam_traj_y.append(best.y)
            if len(self.slam_traj_x) > self.trail_max_points:
                self.slam_traj_x = self.slam_traj_x[-self.trail_max_points:]
                self.slam_traj_y = self.slam_traj_y[-self.trail_max_points:]

            # Landmarks for plotting (read-only copy)
            self.slam_landmarks = [
                (float(lm.mean[0]), float(lm.mean[1]), int(lm.cls))
                for lm in best.landmarks
            ]

    # ---------------------- Plot update ----------------------

    def update_plot(self):
        with self.data_lock:
            odom_traj_x = list(self.odom_traj_x)
            odom_traj_y = list(self.odom_traj_y)
            odom_x = self.odom_car_x
            odom_y = self.odom_car_y
            odom_yaw = self.odom_car_yaw

            slam_traj_x = list(self.slam_traj_x)
            slam_traj_y = list(self.slam_traj_y)

            landmarks = list(self.slam_landmarks)

        self.ax.clear()
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.grid(True)
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_title("PF-SLAM: odom vs SLAM map")

        all_x: List[float] = []
        all_y: List[float] = []

        # Raw odom path (reference)
        if odom_traj_x and odom_traj_y:
            self.ax.plot(
                odom_traj_x,
                odom_traj_y,
                "--",
                linewidth=1.0,
                color="0.6",
                label="odom path",
            )
            all_x.extend(odom_traj_x)
            all_y.extend(odom_traj_y)

        # SLAM path (best particle)
        if slam_traj_x and slam_traj_y:
            self.ax.plot(
                slam_traj_x,
                slam_traj_y,
                "-",
                linewidth=1.5,
                color="tab:blue",
                label="SLAM path",
            )
            all_x.extend(slam_traj_x)
            all_y.extend(slam_traj_y)

        # Current odom pose (for reference)
        if odom_x is not None and odom_y is not None:
            self.ax.plot(odom_x, odom_y, "o", color="0.3", markersize=4, label="odom car")
            L = 2.0
            hx = odom_x + L * math.cos(odom_yaw)
            hy = odom_y + L * math.sin(odom_yaw)
            self.ax.arrow(
                odom_x,
                odom_y,
                hx - odom_x,
                hy - odom_y,
                head_width=0.6,
                head_length=0.9,
                length_includes_head=True,
                color="0.3",
            )
            all_x.append(odom_x)
            all_y.append(odom_y)

        # Landmarks from best particle
        if landmarks:
            color_map = {
                0: ("blue", "blue cones"),
                1: ("gold", "yellow cones"),
                2: ("orange", "orange cones"),
                3: ("red", "big orange cones"),
            }
            per_cls: Dict[int, List[Tuple[float, float]]] = {}
            for lx, ly, cls_id in landmarks:
                per_cls.setdefault(cls_id, []).append((lx, ly))

            for cls_id, pts in per_cls.items():
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                all_x.extend(xs)
                all_y.extend(ys)
                if cls_id in color_map:
                    col, label = color_map[cls_id]
                else:
                    col, label = ("gray", f"class {cls_id}")
                self.ax.scatter(xs, ys, s=12.0, c=col, marker="o", label=label)

        # Windowing
        if self.fixed_window and slam_traj_x and slam_traj_y:
            cx = slam_traj_x[-1]
            cy = slam_traj_y[-1]
            L = self.window_half_size
            self.ax.set_xlim(cx - L, cx + L)
            self.ax.set_ylim(cy - L, cy + L)
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
            self.ax.legend(
                uniq.values(),
                uniq.keys(),
                loc="upper right",
                fontsize=8,
            )


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
