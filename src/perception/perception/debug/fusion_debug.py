#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

# Same colour convention as before
COLORS = {
    0: "gold",        # yellow cone
    1: "blue",        # blue cone
    2: "darkorange",  # orange cone
    3: "red",         # big orange
    4: "gray",        # unknown / fused default
}


class FusedConesPlotter(Node):
    def __init__(self):
        super().__init__("fused_cones_plotter")

        self.sub = self.create_subscription(
            PointCloud2,
            "/perception/cones_fused",
            self.cb_fused,
            qos_profile_sensor_data,
        )

        self.get_logger().info("[FusedConesPlotter] Subscribed to /cones_fused")

        self._lock = threading.Lock()
        self._cones = []  # list of (x, y, class_id)

        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(6, 8))
        self.fig.canvas.manager.set_window_title("Fused Cones Map")

        # Update timer
        self.timer = self.create_timer(0.1, self.update_plot)

    # -------- ROS callback --------
    def cb_fused(self, msg: PointCloud2):
        cones = []
        try:
            for x, y, _, cls_id in pc2.read_points(
                msg, field_names=("x", "y", "z", "class_id"), skip_nans=True
            ):
                cones.append((float(x), float(y), int(cls_id)))
        except Exception as e:
            self.get_logger().warn(f"[FusedConesPlotter] Error parsing PointCloud2: {e}")
            return

        with self._lock:
            self._cones = cones

    # -------- Plot update --------
    def update_plot(self):
        with self._lock:
            cones = list(self._cones)

        self.ax.cla()
        self.ax.set_title(f"Fused Cones (N={len(cones)})")
        self.ax.set_xlabel("Lateral (m)")   # y (left/right)
        self.ax.set_ylabel("Forward (m)")   # x (forward)

        # Forward up, lateral horizontal, same as earlier
        self.ax.set_xlim(8, -8)    # left/right
        self.ax.set_ylim(-2, 25)   # forward

        self.ax.grid(True, alpha=0.3)

        # Ego vehicle at origin
        self.ax.plot(0.0, 0.0, '^', color='lime', markersize=10, label='Ego')

        for x, y, cls_id in cones:
            color = COLORS.get(cls_id, "black")
            # Plot as (y, x) -> forward on vertical axis
            self.ax.plot(y, x, 'o', color=color, markersize=6)

        self.ax.legend(loc="upper right")
        self.fig.canvas.draw_idle()
        plt.pause(0.001)


def main(args=None):
    rclpy.init(args=args)
    node = FusedConesPlotter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
