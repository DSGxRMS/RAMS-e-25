#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2D Cone Viewer Node (with metrics)

Subscribes:
  - /cones (sensor_msgs/PointCloud2), centroids from cone_extractor_2d

Displays:
  - Top-down 2D scatter plot (X forward, Y lateral)
  - HUD with:
      - messages received
      - cones in latest message
      - average input interval (between /cones messages)
      - input rate (Hz)
      - viewer FPS
"""

import sys
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

# GUI
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg


class ConeViewer2D(Node):
    def __init__(self):
        super().__init__("cone_viewer_2d")

        # -------- Parameters --------
        self.declare_parameter("input_topic", "/cones")

        # Plot bounds (in meters, sensor frame)
        # X forward, Y left/right
        self.declare_parameter("x_min", 0.0)
        self.declare_parameter("x_max", 40.0)
        self.declare_parameter("y_min", -10.0)
        self.declare_parameter("y_max", 10.0)

        # Viewer update rate (Hz)
        self.declare_parameter("refresh_hz", 20.0)

        # Read parameters
        self.input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        self.x_min = self.get_parameter("x_min").get_parameter_value().double_value
        self.x_max = self.get_parameter("x_max").get_parameter_value().double_value
        self.y_min = self.get_parameter("y_min").get_parameter_value().double_value
        self.y_max = self.get_parameter("y_max").get_parameter_value().double_value

        self.refresh_hz = self.get_parameter("refresh_hz").get_parameter_value().double_value
        if self.refresh_hz <= 1.0:
            self.refresh_hz = 10.0
            self.get_logger().warn("refresh_hz <= 1. Using 10 Hz instead.")

        # -------- ROS subscriber --------
        self.sub = self.create_subscription(
            PointCloud2,
            self.input_topic,
            self.pc_callback,
            qos_profile_sensor_data,
        )
        self.get_logger().info(f"2D viewer subscribing to: {self.input_topic}")

        # Latest cone centroids (Nx3, z should be 0)
        self._centroids = np.empty((0, 3), dtype=np.float32)

        # Metrics
        self._recv = 0
        self._last_msg_time = None
        self._msg_dt_sum = 0.0
        self._msg_dt_count = 0

        # Viewer FPS tracking
        self._frames = 0
        self._last_fps_time = time.time()
        self._last_fps_value = 0.0

        # -------- Qt / pyqtgraph setup --------
        self.app = QtWidgets.QApplication(sys.argv)

        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle("Cone Viewer 2D — top-down")
        self.win.resize(800, 600)

        central = QtWidgets.QWidget(self.win)
        self.win.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)

        # Plot widget
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "X (forward) [m]")
        self.plot.setLabel("left", "Y (lateral) [m]")
        self.plot.setXRange(self.x_min, self.x_max)
        self.plot.setYRange(self.y_min, self.y_max)
        self.plot.setAspectLocked(False)
        self.plot.showGrid(x=True, y=True, alpha=0.3)

        # Origin lines for reference
        self._x_axis = self.plot.addLine(y=0.0, pen=pg.mkPen(style=QtCore.Qt.DashLine))
        self._y_axis = self.plot.addLine(x=0.0, pen=pg.mkPen(style=QtCore.Qt.DashLine))

        layout.addWidget(self.plot, 1)

        # Scatter item for cones
        self.scatter = pg.ScatterPlotItem(size=10)
        self.plot.addItem(self.scatter)

        # HUD label
        self.lbl = QtWidgets.QLabel("Msgs:0   Cones:0   Avg dt:-- s   In rate:-- Hz   FPS:--")
        layout.addWidget(self.lbl, 0)

        # Timer: integrate ROS spin + GUI update
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._tick)
        interval_ms = int(1000.0 / self.refresh_hz)
        self.timer.start(interval_ms)

    # ---------- ROS callback ----------
    def pc_callback(self, msg: PointCloud2):
        """Store latest centroids from /cones and update message timing stats."""
        recv_time = time.time()

        # Convert to array
        try:
            P = self._pc2_to_xyz_array(msg)
        except Exception as e:
            self.get_logger().warn(f"Error converting cones PC2: {e}")
            return

        self._centroids = P
        self._recv += 1

        # Update average input interval (time between /cones messages)
        if self._last_msg_time is not None:
            dt = recv_time - self._last_msg_time
            if dt > 0:
                self._msg_dt_sum += dt
                self._msg_dt_count += 1
        self._last_msg_time = recv_time

    # ---------- Qt timer tick ----------
    def _tick(self):
        """
        Called at fixed rate.
        - spin_once to process ROS callbacks
        - update scatter plot from latest centroids
        - update HUD (metrics)
        """
        # Process any pending ROS callbacks (including /cones)
        rclpy.spin_once(self, timeout_sec=0.0)

        P = self._centroids
        if P.size == 0:
            xs = []
            ys = []
            n_cones = 0
        else:
            xs = P[:, 0]  # x forward
            ys = P[:, 1]  # y lateral
            n_cones = P.shape[0]

        # Update scatter plot
        self.scatter.setData(xs, ys)

        # Viewer FPS calculation
        self._frames += 1
        now = time.time()
        fps_interval = now - self._last_fps_time
        if fps_interval >= 1.0:
            self._last_fps_value = self._frames / fps_interval
            self._frames = 0
            self._last_fps_time = now

        # Average input interval and rate
        if self._msg_dt_count > 0:
            avg_dt = self._msg_dt_sum / self._msg_dt_count  # seconds
            in_rate = 1.0 / avg_dt if avg_dt > 0 else 0.0
        else:
            avg_dt = 0.0
            in_rate = 0.0

        # HUD text
        avg_dt_ms = avg_dt * 1000.0 if avg_dt > 0 else 0.0
        self.lbl.setText(
            f"Msgs:{self._recv:5d}   "
            f"Cones:{n_cones:3d}   "
            f"Avg dt in:{avg_dt:.3f} s ({avg_dt_ms:.1f} ms)   "
            f"In rate:{in_rate:5.1f} Hz   "
            f"FPS:{self._last_fps_value:4.1f}"
        )

        # Window title with FPS
        self.win.setWindowTitle(f"Cone Viewer 2D — FPS {self._last_fps_value:4.1f}")

    # ---------- Helper: PC2 -> NumPy ----------
    @staticmethod
    def _pc2_to_xyz_array(msg: PointCloud2) -> np.ndarray:
        """Convert PointCloud2 (xyz) to Nx3 float32 array."""
        gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        pts = np.fromiter((c for p in gen for c in p), dtype=np.float32)
        if pts.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        return pts.reshape(-1, 3)


def main(args=None):
    rclpy.init(args=args)
    node = ConeViewer2D()
    node.win.show()
    try:
        sys.exit(node.app.exec())
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
