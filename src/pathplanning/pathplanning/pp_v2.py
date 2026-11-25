# #!/usr/bin/env python3
# # skidpad_debug.py
# #
# # Live debug plotter for /path_points (nav_msgs/Path) + /slam/map_cones.
# # - Plots cones and car pose.
# # - Plots received path points as scatter (NOT joined).
# # - Fits and plots a B-spline through the received poses.
# # - Computes Delaunay triangulation for Blue, Yellow, and a virtual Colourless cone.
# import math
# import threading
# from typing import List, Tuple, Optional

# import rclpy
# from rclpy.node import Node
# from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy

# from nav_msgs.msg import Path
# from geometry_msgs.msg import PoseStamped
# from eufs_msgs.msg import ConeArrayWithCovariance

# import numpy as np

# # Matplotlib
# import matplotlib.pyplot as plt

# # SciPy spline fitting and spatial logic (preferred)
# try:
#     from scipy.interpolate import splprep, splev
#     from scipy.spatial import Delaunay
#     _HAS_SCIPY = True
# except Exception:
#     _HAS_SCIPY = False


# def yaw_from_quat(qx, qy, qz, qw) -> float:
#     """Return yaw (rad) from quaternion."""
#     siny_cosp = 2.0 * (qw * qz + qx * qy)
#     cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
#     return math.atan2(siny_cosp, cosy_cosp)


# class PPDebug(Node):
#     def __init__(self):
#         super().__init__("pp_debug")

#         # ---------------- Parameters ----------------
#         self.declare_parameter("topics.path_in", "/path_points")
#         self.declare_parameter("topics.map_in", "/slam/map_cones")

#         self.declare_parameter("plot.hz", 20.0)
        
#         # Track dimensions
#         self.declare_parameter("track_width", 3.0)                # Expected track width in meters

#         # Spline params
#         self.declare_parameter("spline.include_car_pose", True)   # include pose[0] in spline fit
#         self.declare_parameter("spline.smoothing", 0.5)           # s in splprep (0 = interpolate)
#         self.declare_parameter("spline.num_samples", 200)

#         # Plotting params
#         self.declare_parameter("plot.window_lock", False)         # if True, do not auto-rescale
#         self.declare_parameter("plot.padding_m", 5.0)

#         # QoS
#         self.declare_parameter("qos.best_effort", True)
#         self.declare_parameter("qos.depth", 50)

#         gp = self.get_parameter
#         self.path_topic = str(gp("topics.path_in").value)
#         self.map_topic = str(gp("topics.map_in").value)

#         self.plot_hz = float(gp("plot.hz").value)
#         self.plot_period = 1.0 / max(1e-6, self.plot_hz)
        
#         self.track_width = float(gp("track_width").value)

#         self.include_car_pose = bool(gp("spline.include_car_pose").value)
#         self.spline_s = float(gp("spline.smoothing").value)
#         self.spline_num = int(gp("spline.num_samples").value)

#         self.window_lock = bool(gp("plot.window_lock").value)
#         self.padding_m = float(gp("plot.padding_m").value)

#         best_effort = bool(gp("qos.best_effort").value)
#         depth = int(gp("qos.depth").value)

#         qos = QoSProfile(
#             history=QoSHistoryPolicy.KEEP_LAST,
#             depth=depth,
#             reliability=QoSReliabilityPolicy.BEST_EFFORT if best_effort else QoSReliabilityPolicy.RELIABLE,
#             durability=QoSDurabilityPolicy.VOLATILE,
#         )

#         # ---------------- State ----------------
#         self.lock = threading.Lock()

#         # Latest path message info
#         self.frame_id: str = "map"
#         self.car_pose: Optional[Tuple[float, float, float]] = None  # (x,y,yaw)
#         self.path_points: List[Tuple[float, float]] = []            # remaining points (excluding car pose)

#         # Cones
#         self.blue: List[Tuple[float, float]] = []
#         self.yellow: List[Tuple[float, float]] = []
#         self.orange: List[Tuple[float, float]] = []
#         self.big: List[Tuple[float, float]] = []

#         # ---------------- Subscriptions ----------------
#         self.create_subscription(Path, self.path_topic, self.cb_path, qos)
#         self.create_subscription(ConeArrayWithCovariance, self.map_topic, self.cb_map, qos)

#         # ---------------- Plot setup ----------------
#         plt.ion()
#         self.fig, self.ax = plt.subplots()
#         self.ax.set_title("pp_debug: /path_points B-spline + cones + midpoints")
#         self.ax.set_aspect("equal", adjustable="box")
#         self.ax.grid(True)

#         # Persistent artists (initialized empty)
#         self.scat_blue = self.ax.scatter([], [], color='blue', label="blue cones")
#         self.scat_yellow = self.ax.scatter([], [], color='gold', label="yellow cones")
#         self.scat_orange = self.ax.scatter([], [], color='orange', label="orange cones")
#         self.scat_big = self.ax.scatter([], [], color='orangered', s=80, label="big orange cones")
        
#         # New scatter for the colourless cone and Delaunay midpoints
#         self.scat_colourless = self.ax.scatter([], [], color='gray', marker='^', s=80, label="colourless cone")
#         self.scat_midpoints = self.ax.scatter([], [], color='magenta', marker='x', label="Delaunay midpoints")

#         self.scat_pts = self.ax.scatter([], [], color='black', s=10, label="path points (raw)")

#         self.car_dot, = self.ax.plot([], [], marker="o", color='red', linestyle="", label="car")
#         self.car_arrow = None

#         self.spline_line, = self.ax.plot([], [], linewidth=2.0, color='green', label="B-spline")

#         self.ax.legend(loc="upper right", fontsize='small')

#         if not _HAS_SCIPY:
#             self.get_logger().error(
#                 "[pp_debug] SciPy not available. Install python3-scipy / pip scipy. "
#                 "Spline and Delaunay triangulation will NOT be drawn."
#             )

#         self.get_logger().info(
#             f"[pp_debug] Subscribing path={self.path_topic}, cones={self.map_topic}, plot_hz={self.plot_hz}"
#         )

#         # Timer to refresh plot
#         self.create_timer(self.plot_period, self.timer_cb)

#     # ---------------- Callbacks ----------------
#     def cb_path(self, msg: Path):
#         poses = msg.poses
#         if not poses:
#             return

#         with self.lock:
#             self.frame_id = msg.header.frame_id.strip() or "map"

#             # Pose[0] is current car pose
#             p0 = poses[0].pose
#             x0 = float(p0.position.x)
#             y0 = float(p0.position.y)
#             q = p0.orientation
#             yaw0 = yaw_from_quat(q.x, q.y, q.z, q.w)
#             self.car_pose = (x0, y0, yaw0)

#             # Remaining poses are future points
#             pts = []
#             for ps in poses[1:]:
#                 pts.append((float(ps.pose.position.x), float(ps.pose.position.y)))
#             self.path_points = pts

#     def cb_map(self, msg: ConeArrayWithCovariance):
#         blue, yellow, orange, big = [], [], [], []

#         for c in msg.blue_cones:
#             blue.append((float(c.point.x), float(c.point.y)))
#         for c in msg.yellow_cones:
#             yellow.append((float(c.point.x), float(c.point.y)))
#         for c in msg.orange_cones:
#             orange.append((float(c.point.x), float(c.point.y)))
#         for c in msg.big_orange_cones:
#             big.append((float(c.point.x), float(c.point.y)))

#         with self.lock:
#             self.blue = blue
#             self.yellow = yellow
#             self.orange = orange
#             self.big = big

#     # ---------------- Spline helpers ----------------
#     def _fit_bspline(self, pts: np.ndarray) -> Optional[np.ndarray]:
#         if not _HAS_SCIPY:
#             return None

#         n = pts.shape[0]
#         if n < 2:
#             return None

#         k = min(3, n - 1)
#         if k < 1:
#             return None

#         x = pts[:, 0]
#         y = pts[:, 1]

#         try:
#             tck, _u = splprep([x, y], s=self.spline_s, k=k)
#             uu = np.linspace(0.0, 1.0, max(20, self.spline_num))
#             out = splev(uu, tck)
#             curve = np.vstack(out).T  
#             return curve
#         except Exception as e:
#             self.get_logger().warn(f"[pp_debug] Spline fit failed: {e}")
#             return None

#     # ---------------- Plot refresh ----------------
#     def timer_cb(self):
#         with self.lock:
#             car = self.car_pose
#             pts = list(self.path_points)

#             blue = list(self.blue)
#             yellow = list(self.yellow)
#             orange = list(self.orange)
#             big = list(self.big)

#         # Update scatter data for cones
#         def _set_scatter(sc, data: List[Tuple[float, float]]):
#             if not data:
#                 sc.set_offsets(np.empty((0, 2)))
#             else:
#                 sc.set_offsets(np.array(data, dtype=float))

#         _set_scatter(self.scat_blue, blue)
#         _set_scatter(self.scat_yellow, yellow)
#         _set_scatter(self.scat_orange, orange)
#         _set_scatter(self.scat_big, big)

#         # --- Delaunay & Midpoint Logic ---
#         colourless_cone_data = []
#         midpoints_data = []

#         if car is not None:
#             cx, cy, cyaw = car
#             self.car_dot.set_data([cx], [cy])

#             # Update heading arrow
#             if self.car_arrow is not None:
#                 try:
#                     self.car_arrow.remove()
#                 except Exception:
#                     pass

#             arrow_len = 2.0
#             hx = cx + arrow_len * math.cos(cyaw)
#             hy = cy + arrow_len * math.sin(cyaw)
#             self.car_arrow = self.ax.annotate(
#                 "",
#                 xy=(hx, hy),
#                 xytext=(cx, cy),
#                 arrowprops=dict(arrowstyle="->", color="red", linewidth=2.0),
#             )

#             # 1. Place colourless cone (1/4th track width ahead of the car)
#             cc_dist = self.track_width / 4.0
#             cc_x = cx + cc_dist * math.cos(cyaw)
#             cc_y = cy + cc_dist * math.sin(cyaw)
#             colourless_cone_data.append((cc_x, cc_y))

#             # 2. Perform Delaunay Triangulation and Midpoint Logic
#             if _HAS_SCIPY:
#                 tri_pts = []
#                 tri_labels = [] # 0=Blue, 1=Yellow, 2=Colourless

#                 for bx, by in blue:
#                     tri_pts.append([bx, by])
#                     tri_labels.append(0)
#                 for yx, yy in yellow:
#                     tri_pts.append([yx, yy])
#                     tri_labels.append(1)
                
#                 tri_pts.append([cc_x, cc_y])
#                 tri_labels.append(2)

#                 if len(tri_pts) >= 3:
#                     pts_arr = np.array(tri_pts)
#                     lbl_arr = np.array(tri_labels)
#                     try:
#                         tri = Delaunay(pts_arr)
#                         edges = set()
                        
#                         # Extract unique edges from the simplices
#                         for simplex in tri.simplices:
#                             edges.add(tuple(sorted((simplex[0], simplex[1]))))
#                             edges.add(tuple(sorted((simplex[1], simplex[2]))))
#                             edges.add(tuple(sorted((simplex[2], simplex[0]))))
                        
#                         # Calculate midpoints for valid edges
#                         for u, v in edges:
#                             pair = {lbl_arr[u], lbl_arr[v]}
#                             # Keep if edge is: Blue-Yellow {0,1}, Yellow-Colourless {1,2}, Blue-Colourless {0,2}
#                             if pair in [{0, 1}, {1, 2}, {0, 2}]:
#                                 mx = (pts_arr[u][0] + pts_arr[v][0]) / 2.0
#                                 my = (pts_arr[u][1] + pts_arr[v][1]) / 2.0
#                                 midpoints_data.append((mx, my))
                                
#                     except Exception as e:
#                         self.get_logger().debug(f"Delaunay failed (likely collinear points): {e}")

#         else:
#             self.car_dot.set_data([], [])

#         # Update newly added scatter plots
#         _set_scatter(self.scat_colourless, colourless_cone_data)
#         _set_scatter(self.scat_midpoints, midpoints_data)

#         # Raw path points scatter (NOT joined)
#         if pts:
#             self.scat_pts.set_offsets(np.array(pts, dtype=float))
#         else:
#             self.scat_pts.set_offsets(np.empty((0, 2)))

#         # Build spline input points
#         spline_pts_list: List[Tuple[float, float]] = []
#         if self.include_car_pose and car is not None:
#             spline_pts_list.append((car[0], car[1]))
#         spline_pts_list.extend(pts)

#         spline_curve = None
#         if len(spline_pts_list) >= 2:
#             spline_curve = self._fit_bspline(np.array(spline_pts_list, dtype=float))

#         if spline_curve is not None and spline_curve.shape[0] >= 2:
#             self.spline_line.set_data(spline_curve[:, 0], spline_curve[:, 1])
#         else:
#             self.spline_line.set_data([], [])

#         # Auto-scale view (unless locked)
#         if not self.window_lock:
#             all_xy = []
#             if car is not None:
#                 all_xy.append((car[0], car[1]))
#             all_xy.extend(pts)
#             all_xy.extend(blue)
#             all_xy.extend(yellow)
#             all_xy.extend(orange)
#             all_xy.extend(big)
#             all_xy.extend(colourless_cone_data)
#             all_xy.extend(midpoints_data)

#             if all_xy:
#                 arr = np.array(all_xy, dtype=float)
#                 xmin, ymin = arr.min(axis=0)
#                 xmax, ymax = arr.max(axis=0)
#                 pad = self.padding_m
#                 self.ax.set_xlim(xmin - pad, xmax + pad)
#                 self.ax.set_ylim(ymin - pad, ymax + pad)

#         # Refresh
#         self.fig.canvas.draw_idle()
#         plt.pause(0.001)


# def main():
#     rclpy.init()
#     node = PPDebug()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         try:
#             plt.close("all")
#         except Exception:
#             pass
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == "__main__":
#     main() 

# ''' below is another program 

#!/usr/bin/env python3
# skidpad_debug.py
#
# Live debug plotter for /path_points (nav_msgs/Path) + /slam/map_cones.
# - Plots cones and car pose.
# - Plots received path points as scatter (NOT joined).
# - Fits and plots a B-spline through the received poses.
# - Performs Delaunay triangulation with 60-degree FOV ROI filtering,
#   universal edge length bounds, and OBB rectangle suppression.
# - Includes a heading correction for reversed car orientation.

import math
import threading
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from eufs_msgs.msg import ConeArrayWithCovariance

import numpy as np

# Matplotlib
import matplotlib.pyplot as plt

# SciPy spline fitting and spatial logic
try:
    from scipy.interpolate import splprep, splev
    from scipy.spatial import Delaunay
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def yaw_from_quat(qx, qy, qz, qw) -> float:
    """Return yaw (rad) from quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class PPDebug(Node):
    def __init__(self):
        super().__init__("pp_debug")

        # ---------------- Parameters ----------------
        self.declare_parameter("topics.path_in", "/path_points")
        self.declare_parameter("topics.map_in", "/slam/map_cones")

        self.declare_parameter("plot.hz", 20.0)
        
        # Vehicle Orientation Fix
        self.declare_parameter("flip_heading", True)          # Set True if the car's local frame points backwards

        # Track & Spacing parameters
        self.declare_parameter("track_width", 3.0)            # Track width in meters
        self.declare_parameter("cone_spacing", 3.0)           # Spacing between same-colored cones

        # Filtering parameters
        self.declare_parameter("max_edge_length", 4.5)       # Max allowed distance for ANY Delaunay edge
        self.declare_parameter("min_midpoint_dist", 0.6)     # Min allowed distance for ANY Delaunay edge
        self.declare_parameter("max_lookahead_dist", 12.0)   # Max distance from car to process cones
        self.declare_parameter("fov_deg", 60.0)              # Field of view angle in degrees (60° total / ±30°)

        # Spline params
        self.declare_parameter("spline.include_car_pose", True)   # include pose[0] in spline fit
        self.declare_parameter("spline.smoothing", 0.5)           # s in splprep (0 = interpolate)
        self.declare_parameter("spline.num_samples", 200)

        # Plotting params
        self.declare_parameter("plot.window_lock", False)         # if True, do not auto-rescale
        self.declare_parameter("plot.padding_m", 5.0)

        # QoS
        self.declare_parameter("qos.best_effort", True)
        self.declare_parameter("qos.depth", 50)

        gp = self.get_parameter
        self.path_topic = str(gp("topics.path_in").value)
        self.map_topic = str(gp("topics.map_in").value)

        self.plot_hz = float(gp("plot.hz").value)
        self.plot_period = 1.0 / max(1e-6, self.plot_hz)
        
        self.flip_heading = bool(gp("flip_heading").value)

        self.track_width = float(gp("track_width").value)
        self.cone_spacing = float(gp("cone_spacing").value)

        self.max_edge_length = float(gp("max_edge_length").value)
        self.min_midpoint_dist = float(gp("min_midpoint_dist").value)
        self.max_lookahead_dist = float(gp("max_lookahead_dist").value)
        self.fov_deg = float(gp("fov_deg").value)
        self.fov_rad = math.radians(self.fov_deg)

        self.include_car_pose = bool(gp("spline.include_car_pose").value)
        self.spline_s = float(gp("spline.smoothing").value)
        self.spline_num = int(gp("spline.num_samples").value)

        self.window_lock = bool(gp("plot.window_lock").value)
        self.padding_m = float(gp("plot.padding_m").value)

        best_effort = bool(gp("qos.best_effort").value)
        depth = int(gp("qos.depth").value)

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=depth,
            reliability=QoSReliabilityPolicy.BEST_EFFORT if best_effort else QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # ---------------- State ----------------
        self.lock = threading.Lock()

        # Latest path message info
        self.frame_id: str = "map"
        self.car_pose: Optional[Tuple[float, float, float]] = None  # (x,y,yaw)
        self.path_points: List[Tuple[float, float]] = []            # remaining points (excluding car pose)

        # Cones
        self.blue: List[Tuple[float, float]] = []
        self.yellow: List[Tuple[float, float]] = []
        self.orange: List[Tuple[float, float]] = []
        self.big: List[Tuple[float, float]] = []

        # ---------------- Subscriptions ----------------
        self.create_subscription(Path, self.path_topic, self.cb_path, qos)
        self.create_subscription(ConeArrayWithCovariance, self.map_topic, self.cb_map, qos)

        # ---------------- Plot setup ----------------
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("pp_debug: /path_points B-spline + cones + midpoints")
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True)

        # Persistent artists (initialized empty)
        self.scat_blue = self.ax.scatter([], [], color='blue', label="blue cones")
        self.scat_yellow = self.ax.scatter([], [], color='gold', label="yellow cones")
        self.scat_orange = self.ax.scatter([], [], color='orange', label="orange cones")
        self.scat_big = self.ax.scatter([], [], color='orangered', s=80, label="big orange cones")
        
        # Colourless cone and Delaunay midpoints scatter plots
        self.scat_colourless = self.ax.scatter([], [], color='gray', marker='^', s=80, label="colourless cone")
        self.scat_midpoints = self.ax.scatter([], [], color='magenta', marker='x', label="Delaunay midpoints")

        self.scat_pts = self.ax.scatter([], [], color='black', s=10, label="path points (raw)")

        self.car_dot, = self.ax.plot([], [], marker="o", color='red', linestyle="", label="car")
        self.car_arrow = None

        self.spline_line, = self.ax.plot([], [], linewidth=2.0, color='green', label="B-spline")

        self.ax.legend(loc="upper right", fontsize='small')

        if not _HAS_SCIPY:
            self.get_logger().error(
                "[pp_debug] SciPy not available. Install python3-scipy / pip scipy. "
                "Spline and Delaunay triangulation will NOT be drawn."
            )

        self.get_logger().info(
            f"[pp_debug] Subscribing path={self.path_topic}, cones={self.map_topic}, plot_hz={self.plot_hz}"
        )

        # Timer to refresh plot
        self.create_timer(self.plot_period, self.timer_cb)

    # ---------------- Callbacks ----------------
    def cb_path(self, msg: Path):
        poses = msg.poses
        if not poses:
            return

        with self.lock:
            self.frame_id = msg.header.frame_id.strip() or "map"

            # Pose[0] is current car pose
            p0 = poses[0].pose
            x0 = float(p0.position.x)
            y0 = float(p0.position.y)
            q = p0.orientation
            yaw0 = yaw_from_quat(q.x, q.y, q.z, q.w)
            self.car_pose = (x0, y0, yaw0)

            # Remaining poses are future points
            pts = []
            for ps in poses[1:]:
                pts.append((float(ps.pose.position.x), float(ps.pose.position.y)))
            self.path_points = pts

    def cb_map(self, msg: ConeArrayWithCovariance):
        blue, yellow, orange, big = [], [], [], []

        for c in msg.blue_cones:
            blue.append((float(c.point.x), float(c.point.y)))
        for c in msg.yellow_cones:
            yellow.append((float(c.point.x), float(c.point.y)))
        for c in msg.orange_cones:
            orange.append((float(c.point.x), float(c.point.y)))
        for c in msg.big_orange_cones:
            big.append((float(c.point.x), float(c.point.y)))

        with self.lock:
            self.blue = blue
            self.yellow = yellow
            self.orange = orange
            self.big = big

    # ---------------- Spline helpers ----------------
    def _fit_bspline(self, pts: np.ndarray) -> Optional[np.ndarray]:
        if not _HAS_SCIPY:
            return None

        n = pts.shape[0]
        if n < 2:
            return None

        k = min(3, n - 1)
        if k < 1:
            return None

        x = pts[:, 0]
        y = pts[:, 1]

        try:
            tck, _u = splprep([x, y], s=self.spline_s, k=k)
            uu = np.linspace(0.0, 1.0, max(20, self.spline_num))
            out = splev(uu, tck)
            curve = np.vstack(out).T  
            return curve
        except Exception as e:
            self.get_logger().warn(f"[pp_debug] Spline fit failed: {e}")
            return None

    # ---------------- Plot refresh ----------------
    def timer_cb(self):
        with self.lock:
            car = self.car_pose
            pts = list(self.path_points)

            blue = list(self.blue)
            yellow = list(self.yellow)
            orange = list(self.orange)
            big = list(self.big)

        def _set_scatter(sc, data: List[Tuple[float, float]]):
            if not data:
                sc.set_offsets(np.empty((0, 2)))
            else:
                sc.set_offsets(np.array(data, dtype=float))

        _set_scatter(self.scat_blue, blue)
        _set_scatter(self.scat_yellow, yellow)
        _set_scatter(self.scat_orange, orange)
        _set_scatter(self.scat_big, big)

        colourless_cone_data = []
        midpoints_data = []

        if car is not None:
            cx, cy, cyaw = car
            self.car_dot.set_data([cx], [cy])
            
            # ---------------- Forward Heading Correction ----------------
            # If the vehicle's pose publisher inherently reports facing backward, 
            # add 180 degrees (math.pi) to establish true forward motion.
            forward_yaw = cyaw + math.pi if self.flip_heading else cyaw

            # Heading arrow (now uses corrected forward_yaw)
            if self.car_arrow is not None:
                try:
                    self.car_arrow.remove()
                except Exception:
                    pass

            arrow_len = 2.0
            hx = cx + arrow_len * math.cos(forward_yaw)
            hy = cy + arrow_len * math.sin(forward_yaw)
            self.car_arrow = self.ax.annotate(
                "",
                xy=(hx, hy),
                xytext=(cx, cy),
                arrowprops=dict(arrowstyle="->", color="red", linewidth=2.0),
            )

            # 1. Place colourless cone (1/4th track width ahead of true forward heading)
            cc_dist = self.track_width / 4.0
            cc_x = cx + cc_dist * math.cos(forward_yaw)
            cc_y = cy + cc_dist * math.sin(forward_yaw)
            colourless_cone_data.append((cc_x, cc_y))

            # 2. Perform Delaunay Triangulation & Custom Filtering
            if _HAS_SCIPY:
                tri_pts = []
                tri_labels = []  # 0=Blue, 1=Yellow, 2=Colourless

                # ROI filter: Lookahead Distance + 60-Degree FOV Check (uses forward_yaw)
                def _in_roi(px, py):
                    dx = px - cx
                    dy = py - cy
                    dist = math.hypot(dx, dy)
                    if dist > self.max_lookahead_dist:
                        return False
                    
                    angle_to_pt = math.atan2(dy, dx)
                    angle_diff = math.atan2(math.sin(angle_to_pt - forward_yaw), math.cos(angle_to_pt - forward_yaw))
                    return abs(angle_diff) <= (self.fov_rad / 2.0)

                for bx, by in blue:
                    if _in_roi(bx, by):
                        tri_pts.append([bx, by])
                        tri_labels.append(0)
                for yx, yy in yellow:
                    if _in_roi(yx, yy):
                        tri_pts.append([yx, yy])
                        tri_labels.append(1)

                tri_pts.append([cc_x, cc_y])
                tri_labels.append(2)

                if len(tri_pts) >= 3:
                    pts_arr = np.array(tri_pts)
                    lbl_arr = np.array(tri_labels)
                    try:
                        tri = Delaunay(pts_arr)
                        edges = set()

                        for simplex in tri.simplices:
                            edges.add(tuple(sorted((simplex[0], simplex[1]))))
                            edges.add(tuple(sorted((simplex[1], simplex[2]))))
                            edges.add(tuple(sorted((simplex[2], simplex[0]))))

                        by_midpoints = []
                        colorless_candidates = []

                        # Universal Edge Filtering Pass
                        for u, v in edges:
                            p1, p2 = pts_arr[u], pts_arr[v]
                            edge_len = math.hypot(p1[0] - p2[0], p1[1] - p2[1])

                            # Apply spacing / length bounds universally across all edges
                            if edge_len > self.max_edge_length or edge_len < self.min_midpoint_dist:
                                continue

                            mx, my = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
                            pair = {lbl_arr[u], lbl_arr[v]}

                            if pair == {0, 1}:  # Blue-Yellow
                                p_blue = p1 if lbl_arr[u] == 0 else p2
                                p_yellow = p2 if lbl_arr[u] == 0 else p1
                                by_midpoints.append({
                                    'center': np.array([mx, my]),
                                    'p_blue': np.array(p_blue),
                                    'p_yellow': np.array(p_yellow),
                                    'width': edge_len
                                })
                            elif pair in [{0, 2}, {1, 2}]:  # Colourless + Coloured
                                colorless_candidates.append(np.array([mx, my]))

                        # Rectangle Exclusion Filtering
                        half_length = self.track_width / 2.0
                        half_breadth = (self.cone_spacing / 4.0) / 2.0

                        valid_colorless_midpoints = []

                        for c_pt in colorless_candidates:
                            inside_any_rect = False
                            for by_item in by_midpoints:
                                center = by_item['center']
                                p_b = by_item['p_blue']
                                p_y = by_item['p_yellow']

                                v_across = p_y - p_b
                                track_w = np.linalg.norm(v_across)
                                if track_w == 0:
                                    continue

                                u_across = v_across / track_w
                                u_along = np.array([-u_across[1], u_across[0]])

                                d_vec = c_pt - center
                                dist_across = abs(np.dot(d_vec, u_across))
                                dist_along = abs(np.dot(d_vec, u_along))

                                if dist_across <= half_length and dist_along <= half_breadth:
                                    inside_any_rect = True
                                    break

                            if not inside_any_rect:
                                valid_colorless_midpoints.append(c_pt)

                        # Store validated midpoints
                        for item in by_midpoints:
                            midpoints_data.append((item['center'][0], item['center'][1]))
                        for c_pt in valid_colorless_midpoints:
                            midpoints_data.append((c_pt[0], c_pt[1]))

                    except Exception as e:
                        self.get_logger().debug(f"Delaunay failed: {e}")

        else:
            self.car_dot.set_data([], [])

        _set_scatter(self.scat_colourless, colourless_cone_data)
        _set_scatter(self.scat_midpoints, midpoints_data)

        if pts:
            self.scat_pts.set_offsets(np.array(pts, dtype=float))
        else:
            self.scat_pts.set_offsets(np.empty((0, 2)))

        # Spline visualization
        spline_pts_list: List[Tuple[float, float]] = []
        if self.include_car_pose and car is not None:
            spline_pts_list.append((car[0], car[1]))
        spline_pts_list.extend(pts)

        spline_curve = None
        if len(spline_pts_list) >= 2:
            spline_curve = self._fit_bspline(np.array(spline_pts_list, dtype=float))

        if spline_curve is not None and spline_curve.shape[0] >= 2:
            self.spline_line.set_data(spline_curve[:, 0], spline_curve[:, 1])
        else:
            self.spline_line.set_data([], [])

        # Auto-scale view
        if not self.window_lock:
            all_xy = []
            if car is not None:
                all_xy.append((car[0], car[1]))
            all_xy.extend(pts)
            all_xy.extend(blue)
            all_xy.extend(yellow)
            all_xy.extend(orange)
            all_xy.extend(big)
            all_xy.extend(colourless_cone_data)
            all_xy.extend(midpoints_data)

            if all_xy:
                arr = np.array(all_xy, dtype=float)
                xmin, ymin = arr.min(axis=0)
                xmax, ymax = arr.max(axis=0)
                pad = self.padding_m
                self.ax.set_xlim(xmin - pad, xmax + pad)
                self.ax.set_ylim(ymin - pad, ymax + pad)

        self.fig.canvas.draw_idle()
        plt.pause(0.001)


def main():
    rclpy.init()
    node = PPDebug()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            plt.close("all")
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()