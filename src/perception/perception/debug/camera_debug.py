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

# Class → color mapping (same convention as before)
COLORS = {
    0: "gold",        # yellow cone
    1: "blue",        # blue cone
    2: "darkorange",  # orange cone
    3: "red",         # big orange
    4: "gray",        # other/unknown
}


class ConesColourPlotter(Node):
    def __init__(self):
        super().__init__("cones_colour_plotter")

        self.sub = self.create_subscription(
            PointCloud2,
            "/cones_colour",
            self.cb_cones,
            qos_profile_sensor_data,
        )

        self.get_logger().info("[ConesColourPlotter] Subscribed to /cones_colour")

        # Shared state: last received cones [(x, y, class_id), ...]
        self._lock = threading.Lock()
        self._cones = []

        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(6, 8))
        self.fig.canvas.manager.set_window_title("Cones Colour Map")

        # Plot update timer (10 Hz)
        self.timer = self.create_timer(0.1, self.update_plot)

    # --------- ROS callback ---------
    def cb_cones(self, msg: PointCloud2):
        cones = []
        try:
            # Expect fields: x, y, z, class_id
            for x, y, _, cls_id in pc2.read_points(
                msg,
                field_names=("x", "y", "z", "class_id"),
                skip_nans=True,
            ):
                cones.append((float(x), float(y), int(cls_id)))
        except Exception as e:
            self.get_logger().warn(f"[ConesColourPlotter] Error parsing PointCloud2: {e}")
            return

        with self._lock:
            self._cones = cones

    # --------- Plot update ---------
    def update_plot(self):
        with self._lock:
            cones = list(self._cones)

        self.ax.cla()
        self.ax.set_title(f"Cones (N={len(cones)})")
        self.ax.set_xlabel("Lateral (m)")   # y (left/right)
        self.ax.set_ylabel("Forward (m)")   # x (forward)

        # Same orientation as earlier: forward up, left/right horizontal
        # Flip x-limits so left is positive to the right on the plot if you want:
        self.ax.set_xlim(8, -8)     # lateral: left/right
        self.ax.set_ylim(-2, 25)    # forward range

        self.ax.grid(True, alpha=0.3)

        # Ego vehicle at origin
        self.ax.plot(0.0, 0.0, '^', color='lime', markersize=10, label='Ego')

        for x, y, cls_id in cones:
            color = COLORS.get(cls_id, "black")
            # Plot as (y, x) to keep forward on vertical axis
            self.ax.plot(y, x, 'o', color=color, markersize=6)

        # Simple legend stub (just ego)
        self.ax.legend(loc="upper right")
        self.fig.canvas.draw_idle()
        plt.pause(0.001)


def main(args=None):
    rclpy.init(args=args)
    node = ConesColourPlotter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
