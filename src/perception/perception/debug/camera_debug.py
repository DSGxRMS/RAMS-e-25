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

# Class → color mapping
COLORS = {
    0: "gold",        # yellow cone
    1: "blue",        # blue cone
    2: "darkorange",  # orange cone
    3: "red",         # big orange
    4: "gray",        # other/unknown
}

CLASS_NAMES = {
    0: "yellow",
    1: "blue",
    2: "orange",
    3: "big_orange",
    4: "unknown",
}


class ConesColourPlotter(Node):
    def __init__(self):
        super().__init__("cones_colour_plotter")

        self.declare_parameter("topic", "/perception/cones_stereo")
        self.declare_parameter("frame_mode", "auto")  # auto | optical | ros
        self.declare_parameter("xlim", 8.0)
        self.declare_parameter("ylim_forward", 25.0)

        self.topic = self.get_parameter("topic").value
        self.frame_mode = str(self.get_parameter("frame_mode").value).lower()
        self.xlim = float(self.get_parameter("xlim").value)
        self.ylim_forward = float(self.get_parameter("ylim_forward").value)

        self.sub = self.create_subscription(
            PointCloud2,
            self.topic,
            self.cb_cones,
            qos_profile_sensor_data,
        )

        self.get_logger().info(f"[ConesColourPlotter] Subscribed to {self.topic}")

        self._lock = threading.Lock()
        self._cones = []      # raw tuples: (x, y, z, class_id)
        self._frame_id = ""   # last frame_id

        # Matplotlib
        self.fig, self.ax = plt.subplots(figsize=(6, 8))
        self.fig.canvas.manager.set_window_title("Cones Colour Map")

        self.timer = self.create_timer(0.1, self.update_plot)

    def cb_cones(self, msg: PointCloud2):
        cones = []
        try:
            for x, y, z, cls_id in pc2.read_points(
                msg, field_names=("x", "y", "z", "class_id"), skip_nans=True
            ):
                cones.append((float(x), float(y), float(z), int(cls_id)))
        except Exception as e:
            self.get_logger().warn(f"[ConesColourPlotter] Error parsing PointCloud2: {e}")
            return

        with self._lock:
            self._cones = cones
            self._frame_id = msg.header.frame_id

    def _resolve_mode(self, frame_id: str) -> str:
        if self.frame_mode in ("optical", "ros"):
            return self.frame_mode
        # auto
        if "optical" in frame_id:
            return "optical"
        return "ros"

    def update_plot(self):
        with self._lock:
            cones = list(self._cones)
            frame_id = self._frame_id

        mode = self._resolve_mode(frame_id)

        self.ax.cla()
        self.ax.set_title(f"Cones (N={len(cones)})  frame={frame_id}  mode={mode}")

        # We always plot: x-axis = lateral (left +), y-axis = forward (up)
        self.ax.set_xlabel("Lateral (m)  (left +)")
        self.ax.set_ylabel("Forward (m)")

        self.ax.set_xlim(self.xlim, -self.xlim)
        self.ax.set_ylim(-2, self.ylim_forward)
        self.ax.grid(True, alpha=0.3)

        # Ego marker at origin
        self.ax.plot(0.0, 0.0, '^', color='lime', markersize=10, label='Ego')

        # Plot points
        # mode=optical means incoming is camera optical: X right, Y down, Z forward
        # So: forward = Z, lateral(left+) = -X
        # mode=ros means incoming is x forward, y left, z up
        # So: forward = x, lateral = y
        for x, y, z, cls_id in cones:
            if mode == "optical":
                forward = z
                lateral = -x
            else:
                forward = x
                lateral = y

            color = COLORS.get(cls_id, "black")
            self.ax.plot(lateral, forward, 'o', color=color, markersize=6)

        # Add a compact legend for classes actually present
        present = sorted({c[3] for c in cones})
        for cid in present:
            self.ax.plot([], [], 'o', color=COLORS.get(cid, "black"), label=CLASS_NAMES.get(cid, str(cid)))

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
