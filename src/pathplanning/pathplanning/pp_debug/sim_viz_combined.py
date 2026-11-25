#!/usr/bin/env python3

"""
pp_viz.py

Combined visualization for EUFS path planning.

GLOBAL / MAP FRAME - LEFT
-------------------------
1. Ground-truth track
2. Car position
3. Car heading
4. Driven trace
5. Selected raw path points
6. Published path
7. B-spline of selected path points
8. Zoom box

LOCAL / CAR FRAME - RIGHT
-------------------------
1. Ground-truth cones
2. FOV
3. Delaunay triangulation
4. Kept Delaunay edges
5. Rejected Delaunay edges
6. Midpoint candidates
7. Selected raw path points
8. Published path
9. B-spline
10. Car at local origin

ROS TOPICS
----------
/ground_truth/track
    Complete track in GLOBAL / MAP frame

/ground_truth/cones
    Visible cones in CAR frame

/path_points
    SELECTED PATH POINTS IN GLOBAL / MAP FRAME

/ground_truth/odom
    Car pose in GLOBAL / MAP frame

IMPORTANT
---------
The selected /path_points are assumed to be GLOBAL coordinates.

Every redraw loop:

    global path
          |
          v
    current car pose
          |
          v
    global -> local transformation
          |
          v
    local raw path points

Therefore the local path moves relative to the car as
the car moves forward.
"""


# ============================================================
# IMPORTS
# ============================================================

import math
import os
import threading

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

import rclpy
from rclpy.node import Node

from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
)

from nav_msgs.msg import Odometry, Path
from eufs_msgs.msg import ConeArrayWithCovariance


# ============================================================
# SCIPY - B-SPLINE
# ============================================================

try:

    from scipy.interpolate import splprep, splev

    _HAS_SPLINE = True

except Exception:

    _HAS_SPLINE = False


# ============================================================
# SCIPY - DELAUNAY
# ============================================================

try:

    from scipy.spatial import Delaunay

    _HAS_DELAUNAY = True

except Exception:

    _HAS_DELAUNAY = False


# ============================================================
# COLORS
# ============================================================

BLUE = "#2b6cb0"
YELLOW = "#d69e2e"
ORANGE = "#dd6b20"

GREEN = "#2f855a"
PURPLE = "#6b46c1"


# ============================================================
# QUATERNION -> YAW
# ============================================================

def yaw_from_quat(qx, qy, qz, qw):

    siny_cosp = 2.0 * (
        qw * qz +
        qx * qy
    )

    cosy_cosp = 1.0 - 2.0 * (
        qy * qy +
        qz * qz
    )

    return math.atan2(
        siny_cosp,
        cosy_cosp
    )


# ============================================================
# MAIN NODE
# ============================================================

class SimViz(Node):

    def __init__(self):

        super().__init__("pp_viz")

        # ====================================================
        # PARAMETERS
        # ====================================================

        self.declare_parameter(
            "track_topic",
            "/ground_truth/track"
        )

        self.declare_parameter(
            "cones_topic",
            "/ground_truth/cones"
        )

        self.declare_parameter(
            "path_topic",
            "/path_points"
        )

        self.declare_parameter(
            "odom_topic",
            "/ground_truth/odom"
        )

        # ----------------------------------------------------
        # Delaunay / FOV
        # ----------------------------------------------------

        self.declare_parameter(
            "max_edge_m",
            6.0
        )

        self.declare_parameter(
            "fov_vertex_m",
            -5.0
        )

        self.declare_parameter(
            "fov_radius_m",
            30.0
        )

        self.declare_parameter(
            "fov_angle_deg",
            60.0
        )

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        self.declare_parameter(
            "zoom_m",
            15.0
        )

        self.declare_parameter(
            "rate_hz",
            10.0
        )

        # ----------------------------------------------------
        # B-SPLINE
        # ----------------------------------------------------

        self.declare_parameter(
            "spline_smoothing",
            0.5
        )

        self.declare_parameter(
            "spline_num_samples",
            200
        )

        # ----------------------------------------------------
        # Recording
        # ----------------------------------------------------

        self.declare_parameter(
            "record",
            False
        )

        self.declare_parameter(
            "gif_path",
            os.path.expanduser(
                "~/sim_planning.gif"
            )
        )

        self.declare_parameter(
            "gif_fps",
            10
        )

        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------

        self.declare_parameter(
            "title",
            "EUFS Path Planning - Global + Local + B-Spline"
        )

        # ====================================================
        # READ PARAMETERS
        # ====================================================

        gp = self.get_parameter

        self.track_topic = str(
            gp("track_topic").value
        )

        self.cones_topic = str(
            gp("cones_topic").value
        )

        self.path_topic = str(
            gp("path_topic").value
        )

        self.odom_topic = str(
            gp("odom_topic").value
        )

        self.max_edge = float(
            gp("max_edge_m").value
        )

        self.fov_vertex = float(
            gp("fov_vertex_m").value
        )

        self.fov_r = float(
            gp("fov_radius_m").value
        )

        self.fov_half = math.radians(
            float(
                gp("fov_angle_deg").value
            ) / 2.0
        )

        self.zoom = float(
            gp("zoom_m").value
        )

        self.rate_hz = float(
            gp("rate_hz").value
        )

        self.spline_s = float(
            gp("spline_smoothing").value
        )

        self.spline_num = int(
            gp("spline_num_samples").value
        )

        self.record = bool(
            gp("record").value
        )

        self.gif_path = str(
            gp("gif_path").value
        )

        self.gif_fps = int(
            gp("gif_fps").value
        )

        # ====================================================
        # ROS QoS
        # ====================================================

        reliable_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        best_effort_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=50,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        # ====================================================
        # DATA STORAGE
        # ====================================================

        self.lock = threading.Lock()

        # ----------------------------------------------------
        # Global ground-truth track
        # ----------------------------------------------------

        self.track = {

            "blue":
                np.zeros((0, 2)),

            "yellow":
                np.zeros((0, 2)),

            "orange":
                np.zeros((0, 2)),

            "big":
                np.zeros((0, 2))
        }

        # ----------------------------------------------------
        # Local cones
        # ----------------------------------------------------

        self.local = {

            "blue":
                np.zeros((0, 2)),

            "yellow":
                np.zeros((0, 2)),

            "orange":
                np.zeros((0, 2)),

            "big":
                np.zeros((0, 2))
        }

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # /path_points are stored as GLOBAL points.
        #
        # These do NOT move in the global frame.
        # ----------------------------------------------------

        self.path_global = np.zeros(
            (0, 2)
        )

        # Current local representation
        self.path_local = np.zeros(
            (0, 2)
        )

        # B-splines
        self.spline_global = np.zeros(
            (0, 2)
        )

        self.spline_local = np.zeros(
            (0, 2)
        )

        # ----------------------------------------------------
        # Car pose
        #
        # x, y, yaw in global frame
        # ----------------------------------------------------

        self.pose = None

        # ----------------------------------------------------
        # Driven trace
        # ----------------------------------------------------

        self.trace = []

        # ----------------------------------------------------
        # GIF frames
        # ----------------------------------------------------

        self.frames = []

        # ====================================================
        # SUBSCRIPTIONS
        # ====================================================

        self.create_subscription(
            ConeArrayWithCovariance,
            self.track_topic,
            self.cb_track,
            reliable_qos
        )

        self.create_subscription(
            ConeArrayWithCovariance,
            self.cones_topic,
            self.cb_cones,
            reliable_qos
        )

        self.create_subscription(
            Path,
            self.path_topic,
            self.cb_path,
            best_effort_qos
        )

        self.create_subscription(
            Odometry,
            self.odom_topic,
            self.cb_odom,
            reliable_qos
        )

        # ====================================================
        # PLOT SETUP
        # ====================================================

        self._setup(
            str(
                gp("title").value
            )
        )

        # ====================================================
        # TIMER
        # ====================================================

        self.create_timer(
            1.0 / max(
                self.rate_hz,
                1e-6
            ),
            self.redraw
        )

        # ====================================================
        # LOGGING
        # ====================================================

        if not _HAS_SPLINE:

            self.get_logger().warn(
                "SciPy spline interpolation unavailable. "
                "B-spline will not be shown."
            )

        if not _HAS_DELAUNAY:

            self.get_logger().warn(
                "SciPy Delaunay unavailable. "
                "Delaunay will not be shown."
            )

        self.get_logger().info(
            "=========================================="
        )

        self.get_logger().info(
            "pp_viz started"
        )

        self.get_logger().info(
            f"path topic : {self.path_topic}"
        )

        self.get_logger().info(
            f"track topic: {self.track_topic}"
        )

        self.get_logger().info(
            f"cones topic: {self.cones_topic}"
        )

        self.get_logger().info(
            f"odom topic : {self.odom_topic}"
        )

        self.get_logger().info(
            "PATH FRAME: GLOBAL / MAP"
        )

        self.get_logger().info(
            "=========================================="
        )


    # ========================================================
    # CONE ARRAY -> NUMPY
    # ========================================================

    @staticmethod
    def _arr(cones):

        if not cones:

            return np.zeros(
                (0, 2)
            )

        return np.array(
            [
                [
                    float(c.point.x),
                    float(c.point.y)
                ]
                for c in cones
            ],
            dtype=float
        )


    # ========================================================
    # TRACK CALLBACK
    # ========================================================

    def cb_track(
        self,
        msg
    ):

        with self.lock:

            self.track = {

                "blue":
                    self._arr(
                        msg.blue_cones
                    ),

                "yellow":
                    self._arr(
                        msg.yellow_cones
                    ),

                "orange":
                    self._arr(
                        msg.orange_cones
                    ),

                "big":
                    self._arr(
                        msg.big_orange_cones
                    )
            }


    # ========================================================
    # LOCAL CONES CALLBACK
    # ========================================================

    def cb_cones(
        self,
        msg
    ):

        with self.lock:

            self.local = {

                "blue":
                    self._arr(
                        msg.blue_cones
                    ),

                "yellow":
                    self._arr(
                        msg.yellow_cones
                    ),

                "orange":
                    self._arr(
                        msg.orange_cones
                    ),

                "big":
                    self._arr(
                        msg.big_orange_cones
                    )
            }


    # ========================================================
    # PATH CALLBACK
    # ========================================================

    def cb_path(
        self,
        msg
    ):

        if msg.poses:

            pts = np.array(
                [
                    [
                        float(
                            p.pose.position.x
                        ),
                        float(
                            p.pose.position.y
                        )
                    ]
                    for p in msg.poses
                ],
                dtype=float
            )

        else:

            pts = np.zeros(
                (0, 2)
            )

        with self.lock:

            # IMPORTANT:
            #
            # Store the planner's selected points
            # as GLOBAL coordinates.
            #
            # We DO NOT convert them here.
            #
            # Their local position is recalculated
            # every redraw using the latest car pose.

            self.path_global = pts


    # ========================================================
    # ODOM CALLBACK
    # ========================================================

    def cb_odom(
        self,
        msg
    ):

        q = msg.pose.pose.orientation

        x = float(
            msg.pose.pose.position.x
        )

        y = float(
            msg.pose.pose.position.y
        )

        yaw = yaw_from_quat(
            q.x,
            q.y,
            q.z,
            q.w
        )

        pose = (
            x,
            y,
            yaw
        )

        with self.lock:

            self.pose = pose

            # Add point to driven trace
            # only if vehicle moved > 0.2 m

            if (

                not self.trace

                or

                (
                    (x - self.trace[-1][0]) ** 2
                    +
                    (y - self.trace[-1][1]) ** 2
                ) > 0.04

            ):

                self.trace.append(
                    (x, y)
                )


    # ========================================================
    # GLOBAL -> LOCAL
    # ========================================================

    @staticmethod
    def _to_local(
        pts,
        pose
    ):

        """
        Convert GLOBAL / MAP coordinates to
        CAR-LOCAL coordinates.

        Global:
            [x_global, y_global]

        Car pose:
            [car_x, car_y, yaw]

        Local:
            R(-yaw) @
            (p_global - p_car)

        This function is called EVERY redraw loop.
        Therefore the local path changes as the
        car moves.
        """

        if len(pts) == 0:

            return np.zeros(
                (0, 2)
            )

        x, y, yaw = pose

        c = math.cos(
            yaw
        )

        s = math.sin(
            yaw
        )

        # Translate relative to car
        translated = (
            pts
            -
            np.array(
                [x, y],
                dtype=float
            )
        )

        # Rotation by -yaw
        R_inv = np.array(
            [
                [c, s],
                [-s, c]
            ],
            dtype=float
        )

        return (
            translated
            @
            R_inv.T
        )


    # ========================================================
    # LOCAL -> GLOBAL
    # ========================================================

    @staticmethod
    def _to_global(
        pts,
        pose
    ):

        if len(pts) == 0:

            return np.zeros(
                (0, 2)
            )

        x, y, yaw = pose

        c = math.cos(
            yaw
        )

        s = math.sin(
            yaw
        )

        R = np.array(
            [
                [c, -s],
                [s, c]
            ],
            dtype=float
        )

        return (
            pts
            @
            R.T
            +
            np.array(
                [x, y],
                dtype=float
            )
        )


    # ========================================================
    # FOV
    # ========================================================

    def _fov_local(
        self,
        cones
    ):

        """
        Filter cones inside the planner FOV.

        Car:
            (0,0)

        Heading:
            +x
        """

        pts = []
        classes = []

        for name in (
            "blue",
            "yellow",
            "orange",
            "big"
        ):

            for lx, ly in cones[name]:

                vx = (
                    lx
                    -
                    self.fov_vertex
                )

                vy = ly

                r = math.hypot(
                    vx,
                    vy
                )

                if r > self.fov_r:

                    continue

                if r < 1e-3:

                    continue

                angle = math.atan2(
                    vy,
                    vx
                )

                if (
                    abs(angle)
                    <=
                    self.fov_half
                ):

                    pts.append(
                        (lx, ly)
                    )

                    classes.append(
                        name
                    )

        if pts:

            pts = np.array(
                pts,
                dtype=float
            )

        else:

            pts = np.zeros(
                (0, 2)
            )

        return (
            pts,
            classes
        )


    # ========================================================
    # DELAUNAY
    # ========================================================

    def _triangulate(
        self,
        pts,
        classes
    ):

        if (
            len(pts) < 3
            or
            not _HAS_DELAUNAY
        ):

            return (
                [],
                [],
                np.zeros(
                    (0, 2)
                )
            )

        try:

            simplices = (
                Delaunay(
                    pts
                ).simplices
            )

        except Exception:

            return (
                [],
                [],
                np.zeros(
                    (0, 2)
                )
            )

        # ----------------------------------------------------
        # UNIQUE EDGES
        # ----------------------------------------------------

        edges = set()

        for tri in simplices:

            i = int(
                tri[0]
            )

            j = int(
                tri[1]
            )

            k = int(
                tri[2]
            )

            for a, b in (
                (i, j),
                (j, k),
                (k, i)
            ):

                edges.add(
                    (
                        min(a, b),
                        max(a, b)
                    )
                )

        # ----------------------------------------------------
        # FILTER
        # ----------------------------------------------------

        orange_like = {
            "orange",
            "big"
        }

        blue_yellow = {
            "blue",
            "yellow"
        }

        kept = []
        rejected = []
        mids = []

        for i, j in edges:

            p1 = pts[i]
            p2 = pts[j]

            c1 = classes[i]
            c2 = classes[j]

            segment = (
                p1,
                p2
            )

            # Edge length
            too_long = (
                np.sum(
                    (p1 - p2) ** 2
                )
                >=
                self.max_edge ** 2
            )

            # Same blue
            same_blue = (
                c1 == "blue"
                and
                c2 == "blue"
            )

            # Same yellow
            same_yellow = (
                c1 == "yellow"
                and
                c2 == "yellow"
            )

            same_colour = (
                same_blue
                or
                same_yellow
            )

            # Orange <-> blue/yellow
            cross_colour = (

                (
                    c1 in orange_like
                    and
                    c2 in blue_yellow
                )

                or

                (
                    c2 in orange_like
                    and
                    c1 in blue_yellow
                )
            )

            if (
                too_long
                or
                same_colour
                or
                cross_colour
            ):

                rejected.append(
                    segment
                )

            else:

                kept.append(
                    segment
                )

                # Midpoint
                if not (
                    c1 == "big"
                    and
                    c2 == "big"
                ):

                    mids.append(
                        (p1 + p2)
                        /
                        2.0
                    )

        if mids:

            mids = np.array(
                mids,
                dtype=float
            )

        else:

            mids = np.zeros(
                (0, 2)
            )

        return (
            kept,
            rejected,
            mids
        )


    # ========================================================
    # B-SPLINE
    # ========================================================

    def _fit_bspline(
        self,
        pts
    ):

        """
        Fit B-spline through the selected points.

        This does NOT change the planner path.

        It is only visualization.
        """

        if not _HAS_SPLINE:

            return np.zeros(
                (0, 2)
            )

        if len(pts) < 2:

            return np.zeros(
                (0, 2)
            )

        # ----------------------------------------------------
        # Remove consecutive duplicate points
        # ----------------------------------------------------

        clean = [
            np.asarray(
                pts[0],
                dtype=float
            )
        ]

        for p in pts[1:]:

            p = np.asarray(
                p,
                dtype=float
            )

            if (
                np.linalg.norm(
                    p
                    -
                    clean[-1]
                )
                >
                1e-8
            ):

                clean.append(
                    p
                )

        pts = np.asarray(
            clean,
            dtype=float
        )

        if len(pts) < 2:

            return np.zeros(
                (0, 2)
            )

        # ----------------------------------------------------
        # Spline degree
        # ----------------------------------------------------

        k = min(
            3,
            len(pts) - 1
        )

        try:

            x = pts[:, 0]
            y = pts[:, 1]

            tck, _ = splprep(
                [x, y],
                s=self.spline_s,
                k=k
            )

            u = np.linspace(
                0.0,
                1.0,
                max(
                    20,
                    self.spline_num
                )
            )

            x_new, y_new = splev(
                u,
                tck
            )

            return np.column_stack(
                (
                    x_new,
                    y_new
                )
            )

        except Exception as e:

            self.get_logger().warn(
                f"B-spline failed: {e}"
            )

            return np.zeros(
                (0, 2)
            )


    # ========================================================
    # SCATTER UPDATE
    # ========================================================

    @staticmethod
    def _set_scatter(
        scatter,
        pts
    ):

        if len(pts):

            scatter.set_offsets(
                pts
            )

        else:

            scatter.set_offsets(
                np.zeros(
                    (0, 2)
                )
            )


    # ========================================================
    # PLOT SETUP
    # ========================================================

    def _setup(
        self,
        title
    ):

        plt.ion()

        self.fig = plt.figure(
            figsize=(15, 7)
        )

        gs = self.fig.add_gridspec(
            1,
            2,
            width_ratios=[
                1.25,
                1.0
            ],
            wspace=0.20
        )

        self.ax1 = self.fig.add_subplot(
            gs[0]
        )

        self.ax2 = self.fig.add_subplot(
            gs[1]
        )

        # ----------------------------------------------------
        # AXES
        # ----------------------------------------------------

        for ax in (
            self.ax1,
            self.ax2
        ):

            ax.set_aspect(
                "equal",
                adjustable="box"
            )

            ax.grid(
                alpha=0.15,
                linewidth=0.5
            )

            ax.set_xlabel(
                "x [m]"
            )

            ax.set_ylabel(
                "y [m]"
            )

        self.ax1.set_title(
            "GLOBAL / MAP FRAME"
        )

        self.ax2.set_title(
            "LOCAL / CAR FRAME"
        )

        # ====================================================
        # LEFT PANEL
        # ====================================================

        # Blue cones
        self.t_blue = self.ax1.scatter(
            [],
            [],
            s=12,
            c=BLUE,
            zorder=3,
            label="blue cones"
        )

        # Yellow cones
        self.t_yellow = self.ax1.scatter(
            [],
            [],
            s=12,
            c=YELLOW,
            zorder=3,
            label="yellow cones"
        )

        # Orange cones
        self.t_orange = self.ax1.scatter(
            [],
            [],
            s=15,
            c=ORANGE,
            zorder=3,
            label="orange cones"
        )

        # Driven trace
        self.t_trace, = self.ax1.plot(
            [],
            [],
            "--",
            color="0.45",
            linewidth=1.1,
            zorder=4,
            label="driven trace"
        )

        # Global published path
        self.t_path, = self.ax1.plot(
            [],
            [],
            "-",
            color=GREEN,
            linewidth=2.2,
            zorder=5,
            label="published path"
        )

        # Global raw points
        self.t_raw = self.ax1.scatter(
            [],
            [],
            s=30,
            c=GREEN,
            marker="o",
            edgecolors="black",
            linewidths=0.5,
            zorder=7,
            label="raw path points"
        )

        # Global B-spline
        self.t_spline, = self.ax1.plot(
            [],
            [],
            "--",
            color=PURPLE,
            linewidth=2.0,
            zorder=6,
            label="B-spline"
        )

        # Car
        self.t_car, = self.ax1.plot(
            [],
            [],
            "s",
            color="black",
            markersize=8,
            zorder=8,
            label="car"
        )

        # Car arrow
        self.t_car_arrow = None

        # Zoom box
        self.t_box, = self.ax1.plot(
            [],
            [],
            "-",
            color="0.35",
            linewidth=0.9,
            zorder=6
        )

        self.ax1.legend(
            fontsize=7,
            loc="upper right"
        )

        # ====================================================
        # RIGHT PANEL
        # ====================================================

        # Kept edges
        self.e_keep = LineCollection(
            [],
            colors="#2d3748",
            linewidths=1.5,
            zorder=3
        )

        # Rejected edges
        self.e_rej = LineCollection(
            [],
            colors="#a0aec0",
            linewidths=0.8,
            linestyles=":",
            zorder=2
        )

        self.ax2.add_collection(
            self.e_keep
        )

        self.ax2.add_collection(
            self.e_rej
        )

        # Blue cones
        self.l_blue = self.ax2.scatter(
            [],
            [],
            s=45,
            c=BLUE,
            zorder=5,
            edgecolors="black",
            linewidths=0.3
        )

        # Yellow cones
        self.l_yellow = self.ax2.scatter(
            [],
            [],
            s=45,
            c=YELLOW,
            zorder=5,
            edgecolors="black",
            linewidths=0.3
        )

        # Orange cones
        self.l_orange = self.ax2.scatter(
            [],
            [],
            s=55,
            c=ORANGE,
            zorder=5,
            edgecolors="black",
            linewidths=0.3
        )

        # Midpoint candidates
        self.l_mid = self.ax2.scatter(
            [],
            [],
            s=55,
            c=PURPLE,
            marker="x",
            linewidths=1.8,
            zorder=6
        )

        # Local published path
        self.l_path, = self.ax2.plot(
            [],
            [],
            "-",
            color=GREEN,
            linewidth=2.2,
            zorder=7
        )

        # Local raw points
        self.l_raw = self.ax2.scatter(
            [],
            [],
            s=30,
            c=GREEN,
            marker="o",
            edgecolors="black",
            linewidths=0.5,
            zorder=9
        )

        # Local B-spline
        self.l_spline, = self.ax2.plot(
            [],
            [],
            "--",
            color=PURPLE,
            linewidth=1.8,
            zorder=8
        )

        # FOV
        self.l_fov, = self.ax2.plot(
            [],
            [],
            "-",
            color="#3182ce",
            linewidth=0.8,
            alpha=0.55,
            zorder=1
        )

        # Car at origin
        self.l_car, = self.ax2.plot(
            [0],
            [0],
            "s",
            color="black",
            markersize=9,
            zorder=10
        )

        # Information text
        self.info = self.ax2.text(
            0.02,
            0.98,
            "",
            transform=self.ax2.transAxes,
            va="top",
            fontsize=8.5,
            bbox=dict(
                boxstyle="round",
                fc="white",
                ec="0.75",
                alpha=0.9
            )
        )

        # ----------------------------------------------------
        # LEGEND
        # ----------------------------------------------------

        self.ax2.legend(
            handles=[

                Line2D(
                    [],
                    [],
                    color="#2d3748",
                    linewidth=1.5,
                    label="kept Delaunay edge"
                ),

                Line2D(
                    [],
                    [],
                    color="#a0aec0",
                    linewidth=1.0,
                    linestyle=":",
                    label="rejected Delaunay edge"
                ),

                Line2D(
                    [],
                    [],
                    color="#3182ce",
                    linewidth=0.9,
                    label="FOV"
                ),

                Line2D(
                    [],
                    [],
                    color=PURPLE,
                    marker="x",
                    linestyle="",
                    label="midpoint candidate"
                ),

                Line2D(
                    [],
                    [],
                    color=GREEN,
                    linewidth=2.0,
                    label="published path"
                ),

                Line2D(
                    [],
                    [],
                    color=GREEN,
                    marker="o",
                    linestyle="",
                    label="raw path points"
                ),

                Line2D(
                    [],
                    [],
                    color=PURPLE,
                    linestyle="--",
                    linewidth=1.8,
                    label="B-spline"
                )

            ],
            fontsize=7,
            loc="lower right",
            framealpha=0.9
        )

        self.fig.suptitle(
            title
        )

        self.fig.tight_layout()

        self.fig.show()


    # ========================================================
    # REDRAW
    # ========================================================

    def redraw(self):

        # ====================================================
        # COPY DATA
        # ====================================================

        with self.lock:

            pose = self.pose

            track = {
                k: v.copy()
                for k, v in self.track.items()
            }

            local = {
                k: v.copy()
                for k, v in self.local.items()
            }

            # GLOBAL selected points
            path_global = (
                self.path_global.copy()
            )

            trace = (
                np.array(
                    self.trace,
                    dtype=float
                )
                if self.trace
                else
                np.zeros(
                    (0, 2)
                )
            )

        # Need car pose
        if pose is None:

            return

        # ====================================================
        # CURRENT CAR POSE
        # ====================================================

        cx, cy, yaw = pose

        # ====================================================
        # IMPORTANT:
        #
        # RECALCULATE LOCAL PATH EVERY LOOP
        # ====================================================

        path_local = self._to_local(
            path_global,
            pose
        )

        # ====================================================
        # DELAUNAY / FOV
        # ====================================================

        fov_pts, fov_classes = (
            self._fov_local(
                local
            )
        )

        kept_edges, rejected_edges, midpoint_candidates = (
            self._triangulate(
                fov_pts,
                fov_classes
            )
        )

        # ====================================================
        # B-SPLINE
        # ====================================================

        # Global B-spline
        spline_global = (
            self._fit_bspline(
                path_global
            )
        )

        # Local B-spline
        #
        # IMPORTANT:
        # Fit again after converting to local.
        #
        # Therefore it follows the car.

        spline_local = (
            self._fit_bspline(
                path_local
            )
        )

        # ====================================================
        # SAVE CURRENT VISUALIZATION DATA
        # ====================================================

        with self.lock:

            self.path_local = (
                path_local.copy()
            )

            self.spline_global = (
                spline_global.copy()
            )

            self.spline_local = (
                spline_local.copy()
            )

        # ====================================================
        # LEFT PANEL
        #
        # GLOBAL / MAP
        # ====================================================

        # ----------------------------------------------------
        # Blue cones
        # ----------------------------------------------------

        self._set_scatter(
            self.t_blue,
            track["blue"]
        )

        # ----------------------------------------------------
        # Yellow cones
        # ----------------------------------------------------

        self._set_scatter(
            self.t_yellow,
            track["yellow"]
        )

        # ----------------------------------------------------
        # Orange + big orange
        # ----------------------------------------------------

        if (
            len(track["orange"])
            or
            len(track["big"])
        ):

            orange_global = np.vstack(
                (
                    track["orange"],
                    track["big"]
                )
            )

        else:

            orange_global = np.zeros(
                (0, 2)
            )

        self._set_scatter(
            self.t_orange,
            orange_global
        )

        # ----------------------------------------------------
        # Driven trace
        # ----------------------------------------------------

        if len(trace):

            self.t_trace.set_data(
                trace[:, 0],
                trace[:, 1]
            )

        else:

            self.t_trace.set_data(
                [],
                []
            )

        # ----------------------------------------------------
        # Global selected path
        # ----------------------------------------------------

        if len(path_global):

            self.t_path.set_data(
                path_global[:, 0],
                path_global[:, 1]
            )

        else:

            self.t_path.set_data(
                [],
                []
            )

        # ----------------------------------------------------
        # Global raw points
        # ----------------------------------------------------

        self._set_scatter(
            self.t_raw,
            path_global
        )

        # ----------------------------------------------------
        # Global B-spline
        # ----------------------------------------------------

        if len(spline_global):

            self.t_spline.set_data(
                spline_global[:, 0],
                spline_global[:, 1]
            )

        else:

            self.t_spline.set_data(
                [],
                []
            )

        # ----------------------------------------------------
        # Car
        # ----------------------------------------------------

        self.t_car.set_data(
            [cx],
            [cy]
        )

        # ----------------------------------------------------
        # Remove previous car arrow
        # ----------------------------------------------------

        if self.t_car_arrow is not None:

            try:

                self.t_car_arrow.remove()

            except Exception:

                pass

        # ----------------------------------------------------
        # New car arrow
        # ----------------------------------------------------

        arrow_length = 2.0

        hx = (
            cx
            +
            arrow_length
            *
            math.cos(yaw)
        )

        hy = (
            cy
            +
            arrow_length
            *
            math.sin(yaw)
        )

        self.t_car_arrow = (
            self.ax1.annotate(
                "",
                xy=(
                    hx,
                    hy
                ),
                xytext=(
                    cx,
                    cy
                ),
                arrowprops=dict(
                    arrowstyle="->",
                    linewidth=2.0
                )
            )
        )

        # ----------------------------------------------------
        # Zoom box
        # ----------------------------------------------------

        self.t_box.set_data(

            [
                cx - self.zoom,
                cx + self.zoom,
                cx + self.zoom,
                cx - self.zoom,
                cx - self.zoom
            ],

            [
                cy - self.zoom,
                cy - self.zoom,
                cy + self.zoom,
                cy + self.zoom,
                cy - self.zoom
            ]
        )

        # ----------------------------------------------------
        # Global axis limits
        # ----------------------------------------------------

        global_objects = []

        for values in track.values():

            if len(values):

                global_objects.append(
                    values
                )

        if len(path_global):

            global_objects.append(
                path_global
            )

        if len(spline_global):

            global_objects.append(
                spline_global
            )

        if len(trace):

            global_objects.append(
                trace
            )

        if global_objects:

            all_global = np.vstack(
                global_objects
            )

            padding = 5.0

            self.ax1.set_xlim(
                all_global[:, 0].min()
                -
                padding,

                all_global[:, 0].max()
                +
                padding
            )

            self.ax1.set_ylim(
                all_global[:, 1].min()
                -
                padding,

                all_global[:, 1].max()
                +
                padding
            )

        # ====================================================
        # RIGHT PANEL
        #
        # LOCAL / CAR FRAME
        # ====================================================

        # ----------------------------------------------------
        # Delaunay
        # ----------------------------------------------------

        self.e_keep.set_segments(
            kept_edges
        )

        self.e_rej.set_segments(
            rejected_edges
        )

        # ----------------------------------------------------
        # Local blue cones
        # ----------------------------------------------------

        self._set_scatter(
            self.l_blue,
            local["blue"]
        )

        # ----------------------------------------------------
        # Local yellow cones
        # ----------------------------------------------------

        self._set_scatter(
            self.l_yellow,
            local["yellow"]
        )

        # ----------------------------------------------------
        # Local orange cones
        # ----------------------------------------------------

        if (
            len(local["orange"])
            or
            len(local["big"])
        ):

            orange_local = np.vstack(
                (
                    local["orange"],
                    local["big"]
                )
            )

        else:

            orange_local = np.zeros(
                (0, 2)
            )

        self._set_scatter(
            self.l_orange,
            orange_local
        )

        # ----------------------------------------------------
        # Midpoints
        # ----------------------------------------------------

        self._set_scatter(
            self.l_mid,
            midpoint_candidates
        )

        # ----------------------------------------------------
        # LOCAL PATH
        #
        # THIS CHANGES EVERY LOOP
        # AS THE CAR MOVES.
        # ----------------------------------------------------

        if len(path_local):

            self.l_path.set_data(
                path_local[:, 0],
                path_local[:, 1]
            )

        else:

            self.l_path.set_data(
                [],
                []
            )

        # ----------------------------------------------------
        # LOCAL RAW POINTS
        # ----------------------------------------------------

        self._set_scatter(
            self.l_raw,
            path_local
        )

        # ----------------------------------------------------
        # LOCAL B-SPLINE
        # ----------------------------------------------------

        if len(spline_local):

            self.l_spline.set_data(
                spline_local[:, 0],
                spline_local[:, 1]
            )

        else:

            self.l_spline.set_data(
                [],
                []
            )

        # ====================================================
        # FOV
        # ====================================================

        apex = np.array(
            [
                self.fov_vertex,
                0.0
            ]
        )

        angles = np.linspace(
            -self.fov_half,
            self.fov_half,
            40
        )

        arc = (
            apex
            +
            self.fov_r
            *
            np.stack(
                [
                    np.cos(angles),
                    np.sin(angles)
                ],
                axis=1
            )
        )

        fov_polygon = np.vstack(
            (
                apex,
                arc,
                apex
            )
        )

        self.l_fov.set_data(
            fov_polygon[:, 0],
            fov_polygon[:, 1]
        )

        # ====================================================
        # LOCAL CAR
        # ====================================================

        self.l_car.set_data(
            [0],
            [0]
        )

        # ====================================================
        # LOCAL VIEW
        # ====================================================

        self.ax2.set_xlim(
            -self.zoom * 0.4,
            self.zoom * 1.6
        )

        self.ax2.set_ylim(
            -self.zoom,
            self.zoom
        )

        # ====================================================
        # INFORMATION
        # ====================================================

        self.info.set_text(

            f"car global: "
            f"({cx:.2f}, {cy:.2f})\n"

            f"yaw: "
            f"{math.degrees(yaw):.1f} deg\n"

            f"cones seen: "
            f"{sum(len(v) for v in local.values())}\n"

            f"cones in FOV: "
            f"{len(fov_pts)}\n"

            f"Delaunay kept: "
            f"{len(kept_edges)}\n"

            f"Delaunay rejected: "
            f"{len(rejected_edges)}\n"

            f"midpoint candidates: "
            f"{len(midpoint_candidates)}\n"

            f"global path points: "
            f"{len(path_global)}\n"

            f"local path points: "
            f"{len(path_local)}\n"

            f"B-spline samples: "
            f"{len(spline_local)}"
        )

        # ====================================================
        # REFRESH
        # ====================================================

        self.fig.canvas.draw_idle()

        self.fig.canvas.flush_events()

        # ====================================================
        # RECORD
        # ====================================================

        if self.record:

            self.fig.canvas.draw()

            width, height = (
                self.fig.canvas.get_width_height()
            )

            if hasattr(
                self.fig.canvas,
                "tostring_rgb"
            ):

                buf = np.frombuffer(
                    self.fig.canvas.tostring_rgb(),
                    dtype=np.uint8
                )

                frame = buf.reshape(
                    height,
                    width,
                    3
                ).copy()

            else:

                buf = np.asarray(
                    self.fig.canvas.buffer_rgba(),
                    dtype=np.uint8
                )

                frame = (
                    buf.reshape(
                        height,
                        width,
                        4
                    )[:, :, :3]
                    .copy()
                )

            self.frames.append(
                frame
            )


    # ========================================================
    # SAVE GIF
    # ========================================================

    def save_gif(self):

        if (
            not self.record
            or
            not self.frames
        ):

            if self.record:

                self.get_logger().warn(
                    "No frames captured."
                )

            return

        try:

            from PIL import Image

        except ImportError:

            self.get_logger().error(
                "Pillow is not installed."
            )

            return

        images = [
            Image.fromarray(
                frame
            )
            for frame in self.frames
        ]

        images[0].save(
            self.gif_path,
            save_all=True,
            append_images=images[1:],
            duration=int(
                1000 /
                max(
                    self.gif_fps,
                    1
                )
            ),
            loop=0
        )

        self.get_logger().info(
            f"Saved GIF to: {self.gif_path}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    rclpy.init()

    node = SimViz()

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    finally:

        node.save_gif()

        node.destroy_node()

        rclpy.shutdown()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()