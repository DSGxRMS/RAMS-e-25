#!/usr/bin/env python3
"""
step1_frontend_debug_plotter.py

Matplotlib-only visualiser for Step 1:
- Panel A: Ego-frame cones scatter (bx, by) coloured by class_id.
- Panel B: Motion increments at cone timestamps (Δx_body, Δy_body, Δyaw) + timing (dt_cones, latency).

Subscribes:
  - /perception/cones_fused (sensor_msgs/PointCloud2) with fields: x, y, z, class_id
  - /slam/odom_raw          (nav_msgs/Odometry)

Key idea:
- Cones are already in ego/body frame, so Panel A is raw (no transforms).
- For Panel B, we interpolate odom pose at cone stamp, then compute body-frame increments
  between consecutive cone stamps using previous yaw.

Run:
  ros2 run <your_pkg> step1_frontend_debug_plotter
or:
  python3 step1_frontend_debug_plotter.py

Notes:
- Uses TkAgg backend; run with a desktop session (not headless).
- Set use_sim_time=true if you're in sim and your system uses sim time.
"""

import math
import bisect
import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

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


@dataclass
class ConeFrame:
    t: float
    cones: np.ndarray  # (N,3) -> [bx, by, cls_id]


class Step1FrontendDebugPlotter(Node):
    def __init__(self):
        super().__init__("step1_frontend_debug_plotter")

        # ---------------- Params ----------------
        self.declare_parameter("topics.odom_in", "/slam/odom_raw")
        self.declare_parameter("topics.cones_in", "/perception/cones_fused")

        self.declare_parameter("qos.best_effort", True)
        self.declare_parameter("qos.depth", 5)

        # How much odom history to keep for interpolation
        self.declare_parameter("odom_buffer_sec", 8.0)

        # Plot window history
        self.declare_parameter("history.max_frames", 250)

        # Ego plot limits
        self.declare_parameter("ego.xlim", 25.0)
        self.declare_parameter("ego.ylim", 10.0)

        # Corridor overlay (optional sanity)
        self.declare_parameter("corridor.length", 12.0)
        self.declare_parameter("corridor.width", 4.0)

        # Stale protection visualization (not dropping here; only plotting)
        self.declare_parameter("timing.warn_latency_sec", 0.20)

        gp = self.get_parameter

        self.odom_topic = str(gp("topics.odom_in").value)
        self.cones_topic = str(gp("topics.cones_in").value)

        best_effort = bool(gp("qos.best_effort").value)
        depth = int(gp("qos.depth").value)

        self.odom_buffer_sec = float(gp("odom_buffer_sec").value)
        self.max_frames = int(gp("history.max_frames").value)

        self.ego_xlim = float(gp("ego.xlim").value)
        self.ego_ylim = float(gp("ego.ylim").value)

        self.corr_len = float(gp("corridor.length").value)
        self.corr_w = float(gp("corridor.width").value)

        self.warn_latency = float(gp("timing.warn_latency_sec").value)

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=depth,
            reliability=QoSReliabilityPolicy.BEST_EFFORT if best_effort else QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # ---------------- State ----------------
        # Odom buffer items: (t, x, y, yaw)
        self.odom_buf: Deque[Tuple[float, float, float, float]] = deque()

        # Latest cone frame (ego)
        self.latest_cones: Optional[ConeFrame] = None

        # Time series history (per cone frame)
        self.ts_t: Deque[float] = deque(maxlen=self.max_frames)
        self.ts_dt: Deque[float] = deque(maxlen=self.max_frames)
        self.ts_latency: Deque[float] = deque(maxlen=self.max_frames)

        self.ts_dxb: Deque[float] = deque(maxlen=self.max_frames)
        self.ts_dyb: Deque[float] = deque(maxlen=self.max_frames)
        self.ts_dyaw: Deque[float] = deque(maxlen=self.max_frames)

        # For increments
        self.prev_pose_at_cone: Optional[Tuple[float, float, float]] = None
        self.prev_t_cone: Optional[float] = None

        self.lock = threading.Lock()

        # ---------------- ROS I/O ----------------
        self.create_subscription(Odometry, self.odom_topic, self.cb_odom, qos)
        self.create_subscription(PointCloud2, self.cones_topic, self.cb_cones, qos)

        self.get_logger().info(
            f"[step1] Subscribed: odom={self.odom_topic}, cones={self.cones_topic} | "
            f"odom_buffer_sec={self.odom_buffer_sec}, history_frames={self.max_frames}"
        )

    # ---------------- Odom interpolation ----------------
    def _pose_at(self, t_query: float) -> Optional[Tuple[float, float, float]]:
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
        return float(x), float(y), float(wrap(yaw))

    # ---------------- Callbacks ----------------
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
                self.odom_buf.popleft()

    def cb_cones(self, msg: PointCloud2):
        t_c = RclTime.from_msg(msg.header.stamp).nanoseconds * 1e-9
        t_now = self.get_clock().now().nanoseconds * 1e-9
        latency = float(t_now - t_c)

        # Parse cones (ego/body frame)
        try:
            pts = [(float(bx), float(by), int(cls_id))
                   for (bx, by, _bz, cls_id) in pc2.read_points(
                       msg, field_names=("x", "y", "z", "class_id"), skip_nans=True)]
        except Exception as e:
            self.get_logger().warn(f"[step1] PointCloud2 parse error: {e}")
            return

        cones = np.array(pts, dtype=np.float64) if pts else np.zeros((0, 3), dtype=np.float64)

        # Pose at cone time (for increments)
        pose = None
        with self.lock:
            pose = self._pose_at(t_c)

        if pose is None:
            # still store timing + cones for Panel A
            with self.lock:
                self.latest_cones = ConeFrame(t=t_c, cones=cones)
                self._push_timing_only(t_c, latency)
            return

        x_o, y_o, yaw_o = pose

        # Compute body-frame increments between consecutive cone timestamps
        dxb = dyb = dyaw = 0.0
        dt = 0.0

        with self.lock:
            if self.prev_pose_at_cone is not None and self.prev_t_cone is not None:
                x_p, y_p, yaw_p = self.prev_pose_at_cone
                dt = float(t_c - self.prev_t_cone)

                dx_map = x_o - x_p
                dy_map = y_o - y_p
                dyaw = wrap(yaw_o - yaw_p)

                # map -> body using previous yaw (consistent with your SLAM node)
                c = math.cos(-yaw_p)
                s = math.sin(-yaw_p)
                dxb = c * dx_map - s * dy_map
                dyb = s * dx_map + c * dy_map

            # update state
            self.prev_pose_at_cone = (x_o, y_o, yaw_o)
            self.prev_t_cone = t_c

            self.latest_cones = ConeFrame(t=t_c, cones=cones)

            # time series
            self.ts_t.append(t_c)
            self.ts_dt.append(dt)
            self.ts_latency.append(latency)

            self.ts_dxb.append(float(dxb))
            self.ts_dyb.append(float(dyb))
            self.ts_dyaw.append(float(dyaw))

    def _push_timing_only(self, t_c: float, latency: float):
        dt = 0.0
        if self.prev_t_cone is not None:
            dt = float(t_c - self.prev_t_cone)
        self.prev_t_cone = t_c

        self.ts_t.append(t_c)
        self.ts_dt.append(dt)
        self.ts_latency.append(latency)
        self.ts_dxb.append(0.0)
        self.ts_dyb.append(0.0)
        self.ts_dyaw.append(0.0)


def run_plot(node: Step1FrontendDebugPlotter):
    plt.ion()
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.0], width_ratios=[1.4, 1.0])

    ax_ego = fig.add_subplot(gs[0, 0])
    ax_inc = fig.add_subplot(gs[0, 1])
    ax_timing = fig.add_subplot(gs[1, :])

    fig.suptitle("Step 1 — Front-End Prediction + Timing Sanity (Matplotlib)")

    # Setup ego plot
    ax_ego.set_title("Ego-frame Cones (bx, by)")
    ax_ego.set_xlabel("bx [m] (forward)")
    ax_ego.set_ylabel("by [m] (left)")
    ax_ego.set_xlim(-2.0, node.ego_xlim)
    ax_ego.set_ylim(-node.ego_ylim, node.ego_ylim)
    ax_ego.grid(True, alpha=0.3)
    ax_ego.axhline(0.0, linewidth=1.0)
    ax_ego.axvline(0.0, linewidth=1.0)

    # Corridor overlay
    rect = plt.Rectangle((0.0, -0.5 * node.corr_w), node.corr_len, node.corr_w,
                         fill=False, linewidth=1.5)
    ax_ego.add_patch(rect)

    # Scatter handles per class
    scat_blue = ax_ego.scatter([], [], s=25, label="blue (0)")
    scat_yel = ax_ego.scatter([], [], s=25, label="yellow (1)")
    scat_org = ax_ego.scatter([], [], s=25, label="orange (2)")
    scat_big = ax_ego.scatter([], [], s=25, label="big_orange (3)")
    scat_unk = ax_ego.scatter([], [], s=25, label="unknown")

    ax_ego.legend(loc="upper right", fontsize=9)

    # Increments plot
    ax_inc.set_title("Body-frame increments at cone stamps")
    ax_inc.set_xlabel("frame index (recent)")
    ax_inc.set_ylabel("increment")
    ax_inc.grid(True, alpha=0.3)
    line_dxb, = ax_inc.plot([], [], label="Δx_body [m]")
    line_dyb, = ax_inc.plot([], [], label="Δy_body [m]")
    line_dyaw, = ax_inc.plot([], [], label="Δyaw [rad]")
    ax_inc.legend(loc="upper right", fontsize=9)

    # Timing plot
    ax_timing.set_title("Timing")
    ax_timing.set_xlabel("frame index (recent)")
    ax_timing.set_ylabel("seconds")
    ax_timing.grid(True, alpha=0.3)
    line_dt, = ax_timing.plot([], [], label="dt_cones [s]")
    line_lat, = ax_timing.plot([], [], label="latency now-stamp [s]")
    ax_timing.legend(loc="upper right", fontsize=9)

    # text HUD
    hud = ax_ego.text(
        0.02, 0.98, "", transform=ax_ego.transAxes,
        va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.15)
    )

    def update():
        with node.lock:
            cf = node.latest_cones
            t_hist = list(node.ts_t)
            dxb_hist = list(node.ts_dxb)
            dyb_hist = list(node.ts_dyb)
            dyaw_hist = list(node.ts_dyaw)
            dt_hist = list(node.ts_dt)
            lat_hist = list(node.ts_latency)

        # Panel A: cones scatter
        if cf is None:
            bx = by = cls = np.array([])
            t_c = None
        else:
            cones = cf.cones
            t_c = cf.t
            if cones.size == 0:
                bx = by = cls = np.array([])
            else:
                bx = cones[:, 0]
                by = cones[:, 1]
                cls = cones[:, 2].astype(int)

        def set_scatter(scat, mask):
            if bx.size == 0:
                scat.set_offsets(np.zeros((0, 2)))
            else:
                pts = np.stack([bx[mask], by[mask]], axis=1) if np.any(mask) else np.zeros((0, 2))
                scat.set_offsets(pts)

        if bx.size == 0:
            scat_blue.set_offsets(np.zeros((0, 2)))
            scat_yel.set_offsets(np.zeros((0, 2)))
            scat_org.set_offsets(np.zeros((0, 2)))
            scat_big.set_offsets(np.zeros((0, 2)))
            scat_unk.set_offsets(np.zeros((0, 2)))
        else:
            set_scatter(scat_blue, cls == 0)
            set_scatter(scat_yel, cls == 1)
            set_scatter(scat_org, cls == 2)
            set_scatter(scat_big, cls == 3)
            set_scatter(scat_unk, (cls < 0) | (cls > 3))

        # Panel B: increments
        n = len(dxb_hist)
        xidx = list(range(max(0, n - node.max_frames), n))
        # Use last node.max_frames
        dxb_y = dxb_hist[-len(xidx):] if xidx else []
        dyb_y = dyb_hist[-len(xidx):] if xidx else []
        dyaw_y = dyaw_hist[-len(xidx):] if xidx else []

        line_dxb.set_data(xidx, dxb_y)
        line_dyb.set_data(xidx, dyb_y)
        line_dyaw.set_data(xidx, dyaw_y)

        if xidx:
            ymin = min(min(dxb_y), min(dyb_y), min(dyaw_y))
            ymax = max(max(dxb_y), max(dyb_y), max(dyaw_y))
            pad = 0.1 * max(1e-6, (ymax - ymin))
            ax_inc.set_xlim(xidx[0], xidx[-1])
            ax_inc.set_ylim(ymin - pad, ymax + pad)

        # Timing plot
        dt_y = dt_hist[-len(xidx):] if xidx else []
        lat_y = lat_hist[-len(xidx):] if xidx else []
        line_dt.set_data(xidx, dt_y)
        line_lat.set_data(xidx, lat_y)

        if xidx and dt_y and lat_y:
            ymin = min(min(dt_y), min(lat_y))
            ymax = max(max(dt_y), max(lat_y), node.warn_latency)
            pad = 0.1 * max(1e-6, (ymax - ymin))
            ax_timing.set_xlim(xidx[0], xidx[-1])
            ax_timing.set_ylim(max(0.0, ymin - pad), ymax + pad)

        # HUD
        if xidx and lat_y:
            last_lat = float(lat_y[-1])
            last_dt = float(dt_y[-1]) if dt_y else 0.0
            warn = "WARN" if last_lat > node.warn_latency else "OK"
            hud.set_text(
                f"cones: {int(bx.size)} | latency: {last_lat:.3f}s [{warn}] | dt_cones: {last_dt:.3f}s\n"
                f"corridor: L={node.corr_len:.1f}m W={node.corr_w:.1f}m"
            )
        else:
            hud.set_text("waiting for data...")

        fig.canvas.draw_idle()

    # Main plot loop
    try:
        while rclpy.ok():
            update()
            plt.pause(0.03)  # ~33 Hz plot refresh
    except KeyboardInterrupt:
        pass


def main():
    rclpy.init()
    node = Step1FrontendDebugPlotter()

    # Spin ROS in a background thread so matplotlib can own the main thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        run_plot(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
