#!/usr/bin/env python3

# ================================================================
# Ground-Truth Cone Path Planner + Live Visualisation
#
# INPUTS:
#   /ground_truth/odom
#       -> car pose in global/map frame
#
#   /ground_truth/cones
#       -> cones in CAR frame
#          x = forward
#          y = left
#
# The ground-truth cone coordinates are converted:
#
#       car frame -> global/map frame
#
# and then the original planner operates in global coordinates.
#
# NORMAL MODE:
#   - FOV filtering
#   - Delaunay / k-NN
#   - Original edge constraints
#   - Edge midpoint candidates
#   - Big-orange special
#   - Candidate spacing filtering
#   - Greedy nearest-neighbour path
#
# FALLBACK MODE:
#   Triggered when:
#
#       blue_count <= 1
#       OR
#       yellow_count <= 1
#
#   1. Count blue and yellow cones.
#   2. Select colour with larger count.
#   3. Order dominant-colour cones geometrically.
#   4. Form alternate pairs:
#
#          C0-C2
#          C1-C3
#          C2-C4
#          ...
#
#   5. Each pair forms the MINOR AXIS.
#   6. Midpoint = ellipse centre.
#   7. Major axis is perpendicular.
#   8. Semi-major-axis length = 1 m.
#   9. Construct both major-axis ends.
#  10. Choose the end closer to the car.
#  11. That point becomes a virtual cone.
#  12. Add virtual cones to the dominant-colour set.
#  13. Run Delaunay / k-NN.
#
# PLOT:
#   LEFT:
#       Global/map frame
#       Ground-truth cones
#       Car
#       Delaunay/fallback edges
#       Ellipse construction
#       Virtual cones
#       Candidates
#       Published path
#
#   RIGHT:
#       Local car frame
#       x = forward
#       y = left
#       Raw GT cones
#       FOV
#       Delaunay/fallback edges
#       Ellipse construction
#       Virtual cones
#       Candidates
#       Path
#
# ================================================================

import math
import threading
from typing import List, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
)

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from eufs_msgs.msg import ConeArrayWithCovariance


# ================================================================
# OPTIONAL SCIPY
# ================================================================

try:
    from scipy.spatial import Delaunay, QhullError

    _HAS_SCIPY = True

except ImportError:

    Delaunay = None
    QhullError = Exception
    _HAS_SCIPY = False


# ================================================================
# MATPLOTLIB
# ================================================================

try:

    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    _HAS_MATPLOTLIB = True

except ImportError:

    plt = None
    LineCollection = None
    _HAS_MATPLOTLIB = False


# ================================================================
# COLOURS FOR PLOT
# ================================================================

BLUE_COLOR = "#2563eb"
YELLOW_COLOR = "#eab308"
ORANGE_COLOR = "#f97316"
BIG_COLOR = "#dc2626"
VIRTUAL_COLOR = "#d946ef"
CANDIDATE_COLOR = "#111827"
PATH_COLOR = "#16a34a"
EDGE_COLOR = "#475569"
REJECTED_COLOR = "#94a3b8"
ELLIPSE_COLOR = "#7c3aed"
CAR_COLOR = "#000000"
FOV_COLOR = "#0284c7"


# ================================================================
# QUATERNION -> YAW
# ================================================================

def yaw_from_quat(
    qx,
    qy,
    qz,
    qw
):

    siny_cosp = 2.0 * (
        qw * qz
        +
        qx * qy
    )

    cosy_cosp = 1.0 - 2.0 * (
        qy * qy
        +
        qz * qz
    )

    return math.atan2(
        siny_cosp,
        cosy_cosp
    )


# ================================================================
# NODE
# ================================================================

class SlamPathSectorVisualizer(Node):

    def __init__(self):

        super().__init__(
            "slam_path_sector_visualiser"
        )

        # ============================================================
        # PARAMETERS
        # ============================================================

        # ------------------------------------------------------------
        # Ground-truth inputs
        # ------------------------------------------------------------

        self.declare_parameter(
            "topics.odom_in",
            "/ground_truth/odom"
        )

        self.declare_parameter(
            "topics.map_in",
            "/ground_truth/cones"
        )

        self.declare_parameter(
            "topics.path_out",
            "/path_points"
        )

        # ------------------------------------------------------------
        # QoS
        # ------------------------------------------------------------

        self.declare_parameter(
            "qos.depth",
            10
        )

        # ------------------------------------------------------------
        # FOV
        # ------------------------------------------------------------

        self.declare_parameter(
            "fov.vertex_offset_m",
            -5.0
        )

        self.declare_parameter(
            "fov.radius_m",
            30.0
        )

        self.declare_parameter(
            "fov.angle_deg",
            60.0
        )

        # ------------------------------------------------------------
        # Candidate filtering
        # ------------------------------------------------------------

        self.declare_parameter(
            "candidates.min_spacing_m",
            1.0
        )

        self.declare_parameter(
            "candidates.min_car_dist_m",
            2.0
        )

        # ------------------------------------------------------------
        # Greedy
        # ------------------------------------------------------------

        self.declare_parameter(
            "greedy.max_hop_m",
            7.0
        )

        # ------------------------------------------------------------
        # Ellipse
        # ------------------------------------------------------------

        self.declare_parameter(
            "ellipse.major_axis_half_length_m",
            1.0
        )

        # ------------------------------------------------------------
        # Rate
        # ------------------------------------------------------------

        self.declare_parameter(
            "publish.rate_hz",
            20.0
        )

        # ============================================================
        # READ PARAMETERS
        # ============================================================

        gp = self.get_parameter

        odom_topic = str(
            gp(
                "topics.odom_in"
            ).value
        )

        cone_topic = str(
            gp(
                "topics.map_in"
            ).value
        )

        path_topic = str(
            gp(
                "topics.path_out"
            ).value
        )

        depth = int(
            gp(
                "qos.depth"
            ).value
        )

        # FOV

        self.vertex_offset = float(
            gp(
                "fov.vertex_offset_m"
            ).value
        )

        self.fov_radius = float(
            gp(
                "fov.radius_m"
            ).value
        )

        self.fov_angle_deg = float(
            gp(
                "fov.angle_deg"
            ).value
        )

        self.fov_half_rad = math.radians(
            self.fov_angle_deg * 0.5
        )

        # Candidate

        self.min_spacing = float(
            gp(
                "candidates.min_spacing_m"
            ).value
        )

        self.min_car_dist = float(
            gp(
                "candidates.min_car_dist_m"
            ).value
        )

        # Greedy

        self.max_hop = float(
            gp(
                "greedy.max_hop_m"
            ).value
        )

        # Ellipse

        self.ellipse_major_half_length = float(
            gp(
                "ellipse.major_axis_half_length_m"
            ).value
        )

        # Rate

        self.publish_rate_hz = float(
            gp(
                "publish.rate_hz"
            ).value
        )

        # ============================================================
        # QOS
        # ============================================================

        # Ground-truth topics use RELIABLE,
        # matching the working ground-truth visualizer.

        sub_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=depth,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # Path publisher remains BEST_EFFORT.

        pub_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=depth,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # ============================================================
        # SUBSCRIPTIONS
        # ============================================================

        self.create_subscription(
            Odometry,
            odom_topic,
            self.cb_odom,
            sub_qos
        )

        self.create_subscription(
            ConeArrayWithCovariance,
            cone_topic,
            self.cb_map,
            sub_qos
        )

        # ============================================================
        # PUBLISHER
        # ============================================================

        self.path_pub = self.create_publisher(
            Path,
            path_topic,
            pub_qos
        )

        # ============================================================
        # SHARED STATE
        # ============================================================

        self.data_lock = threading.Lock()

        # Global car pose

        self.car_x = None
        self.car_y = None
        self.car_yaw = None

        # Ground-truth cones ARRIVE IN CAR FRAME

        self.blue_local = []
        self.yellow_local = []
        self.orange_local = []
        self.big_local = []

        # ============================================================
        # PLOT STATE
        # ============================================================

        self.plot_lock = threading.Lock()

        self.plot_data = {
            "car_x": 0.0,
            "car_y": 0.0,
            "car_yaw": 0.0,

            # Raw GT local points
            "raw_local_points": [],
            "raw_local_classes": [],

            # Raw global points
            "raw_global_points": [],
            "raw_global_classes": [],

            # Planner graph
            "graph_points": [],
            "graph_classes": [],
            "graph_edges": [],

            # Ellipse construction
            "ellipse_data": [],

            # Virtual cones
            "virtual_cones": [],

            # Candidates
            "candidate_points": [],

            # Path
            "path_points": [],

            # State
            "dominant_colour": None,
            "mode": "WAITING",
        }

        # ============================================================
        # TIMER
        # ============================================================

        period = 1.0 / max(
            self.publish_rate_hz,
            1e-6
        )

        self.timer = self.create_timer(
            period,
            self._tick
        )

        # ============================================================
        # STARTUP INFO
        # ============================================================

        self.get_logger().info(
            "===================================================="
        )

        self.get_logger().info(
            "GROUND-TRUTH PATH PLANNER STARTED"
        )

        self.get_logger().info(
            f"odom topic : {odom_topic}"
        )

        self.get_logger().info(
            f"cone topic : {cone_topic}"
        )

        self.get_logger().info(
            f"path topic : {path_topic}"
        )

        self.get_logger().info(
            "Ground-truth cones are treated as CAR-FRAME data."
        )

        self.get_logger().info(
            f"ellipse half major axis = "
            f"{self.ellipse_major_half_length:.2f} m"
        )

        self.get_logger().info(
            "===================================================="
        )

        if not _HAS_SCIPY:

            self.get_logger().warn(
                "SciPy unavailable. "
                "Using k-NN instead of Delaunay."
            )

        if not _HAS_MATPLOTLIB:

            self.get_logger().error(
                "Matplotlib unavailable. "
                "Planner will still run without plot."
            )

    # ================================================================
    # ODOM CALLBACK
    # ================================================================

    def cb_odom(
        self,
        msg: Odometry
    ):

        x = float(
            msg.pose.pose.position.x
        )

        y = float(
            msg.pose.pose.position.y
        )

        q = msg.pose.pose.orientation

        yaw = yaw_from_quat(
            q.x,
            q.y,
            q.z,
            q.w
        )

        with self.data_lock:

            self.car_x = x
            self.car_y = y
            self.car_yaw = yaw

    # ================================================================
    # GROUND-TRUTH CONE CALLBACK
    # ================================================================

    def cb_map(
        self,
        msg: ConeArrayWithCovariance
    ):

        blue = []
        yellow = []
        orange = []
        big = []

        # ------------------------------------------------------------
        # BLUE
        # ------------------------------------------------------------

        for c in msg.blue_cones:

            blue.append(
                (
                    float(c.point.x),
                    float(c.point.y)
                )
            )

        # ------------------------------------------------------------
        # YELLOW
        # ------------------------------------------------------------

        for c in msg.yellow_cones:

            yellow.append(
                (
                    float(c.point.x),
                    float(c.point.y)
                )
            )

        # ------------------------------------------------------------
        # ORANGE
        # ------------------------------------------------------------

        for c in msg.orange_cones:

            orange.append(
                (
                    float(c.point.x),
                    float(c.point.y)
                )
            )

        # ------------------------------------------------------------
        # BIG ORANGE
        # ------------------------------------------------------------

        for c in msg.big_orange_cones:

            big.append(
                (
                    float(c.point.x),
                    float(c.point.y)
                )
            )

        # ------------------------------------------------------------
        # SAVE AS LOCAL
        # ------------------------------------------------------------

        with self.data_lock:

            self.blue_local = blue
            self.yellow_local = yellow
            self.orange_local = orange
            self.big_local = big

    # ================================================================
    # WORLD -> LOCAL
    # ================================================================

    def _world_to_local(
        self,
        car_x,
        car_y,
        car_yaw,
        px,
        py
    ):

        """
        Global/map -> car frame.

        Local:
            x = forward
            y = left
        """

        c = math.cos(
            car_yaw
        )

        s = math.sin(
            car_yaw
        )

        dx = px - car_x
        dy = py - car_y

        lx = (
            c * dx
            +
            s * dy
        )

        ly = (
            -s * dx
            +
            c * dy
        )

        return lx, ly

    # ================================================================
    # LOCAL -> WORLD
    # ================================================================

    def _local_to_world(
        self,
        car_x,
        car_y,
        car_yaw,
        lx,
        ly
    ):

        """
        Car frame -> global/map frame.
        """

        c = math.cos(
            car_yaw
        )

        s = math.sin(
            car_yaw
        )

        wx = (
            car_x
            +
            c * lx
            -
            s * ly
        )

        wy = (
            car_y
            +
            s * lx
            +
            c * ly
        )

        return wx, wy

    # ================================================================
    # LOCAL CONES -> GLOBAL CONES
    # ================================================================

    def _local_cones_to_global(
        self,
        car_x,
        car_y,
        car_yaw,
        cones
    ):

        result = []

        for lx, ly in cones:

            wx, wy = self._local_to_world(
                car_x,
                car_y,
                car_yaw,
                lx,
                ly
            )

            result.append(
                (
                    wx,
                    wy
                )
            )

        return result

    # ================================================================
    # COMPUTE FOV
    # ================================================================

    def _compute_fov_points(
        self,
        car_x,
        car_y,
        car_yaw,
        blue_global,
        yellow_global,
        orange_global,
        big_global
    ):

        points_window = []
        classes_window = []

        R = self.fov_radius
        half_angle = self.fov_half_rad
        vertex_offset = self.vertex_offset

        # ------------------------------------------------------------
        # ADD LIST
        # ------------------------------------------------------------

        def add_list(
            cones,
            class_name
        ):

            for px, py in cones:

                lx, ly = \
                    self._world_to_local(
                        car_x,
                        car_y,
                        car_yaw,
                        px,
                        py
                    )

                vx = (
                    lx
                    -
                    vertex_offset
                )

                vy = ly

                r = math.hypot(
                    vx,
                    vy
                )

                if r > R:
                    continue

                if r < 1e-3:
                    continue

                theta = math.atan2(
                    vy,
                    vx
                )

                if abs(theta) <= half_angle:

                    points_window.append(
                        (
                            px,
                            py
                        )
                    )

                    classes_window.append(
                        class_name
                    )

        # ------------------------------------------------------------

        add_list(
            blue_global,
            "blue"
        )

        add_list(
            yellow_global,
            "yellow"
        )

        add_list(
            orange_global,
            "orange"
        )

        add_list(
            big_global,
            "big"
        )

        return (
            points_window,
            classes_window
        )

    # ================================================================
    # RAW GRAPH
    # ================================================================

    def _raw_graph_edges(
        self,
        points
    ):

        N = len(
            points
        )

        if N < 2:

            return []

        pts = np.asarray(
            points,
            dtype=float
        )

        edges_set = set()

        # ------------------------------------------------------------
        # DELAUNAY
        # ------------------------------------------------------------

        if (
            _HAS_SCIPY
            and
            N >= 3
        ):

            try:

                tri = Delaunay(
                    pts
                )

                for simplex in tri.simplices:

                    i, j, k = map(
                        int,
                        simplex
                    )

                    for a, b in (
                        (i, j),
                        (j, k),
                        (k, i)
                    ):

                        if a > b:

                            a, b = \
                                b, a

                        edges_set.add(
                            (
                                a,
                                b
                            )
                        )

            except QhullError:

                pass

        # ------------------------------------------------------------
        # KNN FALLBACK
        # ------------------------------------------------------------

        if not edges_set:

            k = min(
                3,
                N - 1
            )

            for i in range(N):

                d2 = np.sum(
                    (
                        pts
                        -
                        pts[i]
                    ) ** 2,
                    axis=1
                )

                d2[i] = \
                    float("inf")

                nearest = np.argsort(
                    d2
                )[:k]

                for j in nearest:

                    a = i
                    b = int(j)

                    if a > b:

                        a, b = \
                            b, a

                    edges_set.add(
                        (
                            a,
                            b
                        )
                    )

        return list(
            edges_set
        )

    # ================================================================
    # NORMAL GRAPH
    # ================================================================

    def _build_edges(
        self,
        points_window,
        classes_window
    ):

        if len(
            points_window
        ) < 2:

            return []

        raw_edges = \
            self._raw_graph_edges(
                points_window
            )

        orange_like = {
            "orange",
            "big"
        }

        blue_yellow = {
            "blue",
            "yellow"
        }

        max_len2 = \
            6.0 * 6.0

        edges = []

        for i, j in raw_edges:

            ci = classes_window[i]
            cj = classes_window[j]

            x1, y1 = \
                points_window[i]

            x2, y2 = \
                points_window[j]

            d2 = (
                (x1 - x2) ** 2
                +
                (y1 - y2) ** 2
            )

            # Max edge length

            if d2 >= max_len2:

                continue

            # Blue-blue

            if (
                ci == "blue"
                and
                cj == "blue"
            ):

                continue

            # Yellow-yellow

            if (
                ci == "yellow"
                and
                cj == "yellow"
            ):

                continue

            # Orange-like <-> blue/yellow

            if (

                (
                    ci in orange_like
                    and
                    cj in blue_yellow
                )

                or

                (
                    cj in orange_like
                    and
                    ci in blue_yellow
                )

            ):

                continue

            edges.append(
                (
                    i,
                    j
                )
            )

        return edges

    # ================================================================
    # ORDER DOMINANT COLOUR
    # ================================================================

    def _order_points_along_principal_axis(
        self,
        points
    ):

        """
        Gives a geometrically consistent ordering:

            C0, C1, C2, C3, ...

        using the principal spatial axis.
        """

        if len(
            points
        ) <= 2:

            return list(
                points
            )

        pts = np.asarray(
            points,
            dtype=float
        )

        center = np.mean(
            pts,
            axis=0
        )

        centered = (
            pts
            -
            center
        )

        covariance = (
            centered.T
            @
            centered
        )

        eigenvalues, eigenvectors = \
            np.linalg.eigh(
                covariance
            )

        principal_dir = \
            eigenvectors[
                :,
                int(
                    np.argmax(
                        eigenvalues
                    )
                )
            ]

        projections = (
            centered
            @
            principal_dir
        )

        order = np.argsort(
            projections
        )

        ordered = [

            (
                float(
                    pts[i, 0]
                ),

                float(
                    pts[i, 1]
                )
            )

            for i in order
        ]

        return ordered

    # ================================================================
    # BUILD ELLIPSE VIRTUAL CONES
    # ================================================================

    def _build_ellipse_virtual_cones(
        self,
        dominant_points,
        car_x,
        car_y
    ):

        """
        For ordered cones:

            C0 C1 C2 C3 ...

        generate alternate pairs:

            C0-C2
            C1-C3
            C2-C4
            ...

        For every pair:

            minor axis = C_i -> C_(i+2)

            centre = midpoint

            major direction =
                perpendicular to minor direction

            semi-major axis = 1 m

        Both ends are tested.

        The end closer to the car becomes
        the virtual cone.
        """

        ellipse_data = []
        virtual_cones = []

        if len(
            dominant_points
        ) < 3:

            return (
                virtual_cones,
                ellipse_data
            )

        ordered = \
            self._order_points_along_principal_axis(
                dominant_points
            )

        car = np.array(
            [
                car_x,
                car_y
            ],
            dtype=float
        )

        half_major = \
            self.ellipse_major_half_length

        # ------------------------------------------------------------
        # ALTERNATE PAIRS
        # ------------------------------------------------------------

        for i in range(
            len(ordered) - 2
        ):

            p1 = np.array(
                ordered[i],
                dtype=float
            )

            p2 = np.array(
                ordered[i + 2],
                dtype=float
            )

            # --------------------------------------------------------
            # MINOR AXIS
            # --------------------------------------------------------

            minor_vec = (
                p2
                -
                p1
            )

            minor_length = float(
                np.linalg.norm(
                    minor_vec
                )
            )

            if minor_length < 1e-6:

                continue

            minor_dir = (
                minor_vec
                /
                minor_length
            )

            # --------------------------------------------------------
            # MAJOR AXIS
            # --------------------------------------------------------

            major_dir = np.array(
                [
                    -minor_dir[1],
                    minor_dir[0]
                ],
                dtype=float
            )

            # --------------------------------------------------------
            # CENTRE
            # --------------------------------------------------------

            center = (
                0.5
                *
                (
                    p1
                    +
                    p2
                )
            )

            # --------------------------------------------------------
            # BOTH MAJOR AXIS ENDS
            # --------------------------------------------------------

            v_plus = (
                center
                +
                half_major
                *
                major_dir
            )

            v_minus = (
                center
                -
                half_major
                *
                major_dir
            )

            # --------------------------------------------------------
            # DISTANCE TO CAR
            # --------------------------------------------------------

            d_plus2 = float(
                np.sum(
                    (
                        v_plus
                        -
                        car
                    ) ** 2
                )
            )

            d_minus2 = float(
                np.sum(
                    (
                        v_minus
                        -
                        car
                    ) ** 2
                )
            )

            # --------------------------------------------------------
            # SELECT CLOSER END
            # --------------------------------------------------------

            if d_plus2 <= d_minus2:

                selected = v_plus
                selected_side = 1

            else:

                selected = v_minus
                selected_side = -1

            virtual = (
                float(
                    selected[0]
                ),

                float(
                    selected[1]
                )
            )

            virtual_cones.append(
                virtual
            )

            # --------------------------------------------------------
            # SAVE ELLIPSE FOR VISUALISATION
            # --------------------------------------------------------

            ellipse_data.append(

                {
                    "p1": (
                        float(
                            p1[0]
                        ),
                        float(
                            p1[1]
                        )
                    ),

                    "p2": (
                        float(
                            p2[0]
                        ),
                        float(
                            p2[1]
                        )
                    ),

                    "center": (
                        float(
                            center[0]
                        ),
                        float(
                            center[1]
                        )
                    ),

                    "v_plus": (
                        float(
                            v_plus[0]
                        ),
                        float(
                            v_plus[1]
                        )
                    ),

                    "v_minus": (
                        float(
                            v_minus[0]
                        ),
                        float(
                            v_minus[1]
                        )
                    ),

                    "selected": virtual,

                    "selected_side":
                        selected_side,
                }
            )

        return (
            virtual_cones,
            ellipse_data
        )

    # ================================================================
    # FALLBACK GRAPH
    # ================================================================

    def _fallback_graph(
        self,
        points_window,
        classes_window,
        car_x,
        car_y
    ):

        # ------------------------------------------------------------
        # BLUE
        # ------------------------------------------------------------

        blue_points = [

            points_window[i]

            for i, cls in enumerate(
                classes_window
            )

            if cls == "blue"
        ]

        # ------------------------------------------------------------
        # YELLOW
        # ------------------------------------------------------------

        yellow_points = [

            points_window[i]

            for i, cls in enumerate(
                classes_window
            )

            if cls == "yellow"
        ]

        blue_count = len(
            blue_points
        )

        yellow_count = len(
            yellow_points
        )

        # ------------------------------------------------------------
        # DOMINANT COLOUR
        # ------------------------------------------------------------

        if blue_count > yellow_count:

            dominant_colour = "blue"
            dominant_points = blue_points

        elif yellow_count > blue_count:

            dominant_colour = "yellow"
            dominant_points = yellow_points

        else:

            # Equal counts:
            # there is no larger colour.

            return (
                [],
                [],
                [],
                None,
                [],
                []
            )

        # ------------------------------------------------------------
        # VIRTUAL CONES
        # ------------------------------------------------------------

        (
            virtual_cones,
            ellipse_data
        ) = self._build_ellipse_virtual_cones(
            dominant_points,
            car_x,
            car_y
        )

        # ------------------------------------------------------------
        # GRAPH POINTS
        # ------------------------------------------------------------

        graph_points = list(
            dominant_points
        )

        graph_points.extend(
            virtual_cones
        )

        graph_classes = (

            [dominant_colour]
            *
            len(
                dominant_points
            )

        )

        graph_classes.extend(

            ["virtual"]
            *
            len(
                virtual_cones
            )

        )

        # ------------------------------------------------------------
        # GRAPH
        # ------------------------------------------------------------

        raw_edges = \
            self._raw_graph_edges(
                graph_points
            )

        max_len2 = \
            6.0 * 6.0

        edges = []

        for i, j in raw_edges:

            x1, y1 = \
                graph_points[i]

            x2, y2 = \
                graph_points[j]

            d2 = (
                (x1 - x2) ** 2
                +
                (y1 - y2) ** 2
            )

            if d2 >= max_len2:

                continue

            # IMPORTANT:
            #
            # In fallback mode all normal cones are the SAME
            # dominant colour, so we DO NOT reject same-colour
            # edges here.

            edges.append(
                (
                    i,
                    j
                )
            )

        return (
            graph_points,
            graph_classes,
            edges,
            dominant_colour,
            virtual_cones,
            ellipse_data
        )

    # ================================================================
    # BIG ORANGE SPECIAL
    # ================================================================

    def _big_cone_candidate(
        self,
        points_window,
        classes_window
    ):

        big_points = [

            points_window[i]

            for i, cls in enumerate(
                classes_window
            )

            if cls == "big"
        ]

        n = len(
            big_points
        )

        if n < 2:

            return []

        # ------------------------------------------------------------
        # TWO BIG CONES
        # ------------------------------------------------------------

        if n == 2:

            (
                x1,
                y1
            ), (
                x2,
                y2
            ) = big_points

            return [

                (
                    0.5 * (
                        x1 + x2
                    ),

                    0.5 * (
                        y1 + y2
                    ),

                    True
                )

            ]

        # ------------------------------------------------------------
        # THREE OR MORE
        # ------------------------------------------------------------

        xs = [
            p[0]
            for p in big_points
        ]

        ys = [
            p[1]
            for p in big_points
        ]

        return [

            (
                sum(xs) / n,
                sum(ys) / n,
                True
            )

        ]

    # ================================================================
    # CANDIDATE FILTER
    # ================================================================

    def _filter_centroids_min_spacing(
        self,
        centroids_raw,
        car_x,
        car_y,
        min_dist
    ):

        if not centroids_raw:

            return []

        pts = np.asarray(
            [
                [
                    c[0],
                    c[1]
                ]
                for c in centroids_raw
            ],
            dtype=float
        )

        car = np.asarray(
            [
                car_x,
                car_y
            ],
            dtype=float
        )

        d2_car = np.sum(
            (
                pts
                -
                car
            ) ** 2,
            axis=1
        )

        # Farthest-first

        order = np.argsort(
            -d2_car
        )

        kept_pts = []
        kept = []

        for idx in order:

            p = pts[idx]

            # --------------------------------------------------------
            # TOO CLOSE TO CAR
            # --------------------------------------------------------

            if (
                np.linalg.norm(
                    p - car
                )
                <
                self.min_car_dist
            ):

                continue

            # --------------------------------------------------------
            # FIRST POINT
            # --------------------------------------------------------

            if not kept_pts:

                kept_pts.append(
                    p
                )

                kept.append(

                    (
                        float(
                            p[0]
                        ),

                        float(
                            p[1]
                        ),

                        bool(
                            centroids_raw[
                                idx
                            ][2]
                        )
                    )

                )

                continue

            # --------------------------------------------------------
            # SPACING
            # --------------------------------------------------------

            too_close = False

            for kp in kept_pts:

                if (
                    np.linalg.norm(
                        p - kp
                    )
                    <
                    min_dist
                ):

                    too_close = True
                    break

            if too_close:

                continue

            kept_pts.append(
                p
            )

            kept.append(

                (
                    float(
                        p[0]
                    ),

                    float(
                        p[1]
                    ),

                    bool(
                        centroids_raw[
                            idx
                        ][2]
                    )
                )

            )

        return kept

    # ================================================================
    # GREEDY PATH
    # ================================================================

    def _build_greedy_path(
        self,
        car_x,
        car_y,
        candidate_points
    ):

        if not candidate_points:

            return []

        pts = np.asarray(
            candidate_points,
            dtype=float
        )

        N = pts.shape[0]

        used = np.zeros(
            N,
            dtype=bool
        )

        current = np.asarray(
            [
                car_x,
                car_y
            ],
            dtype=float
        )

        max_step2 = (
            self.max_hop
            *
            self.max_hop
        )

        order = []

        for _ in range(N):

            diff = (
                pts
                -
                current
            )

            d2 = np.sum(
                diff * diff,
                axis=1
            )

            d2[used] = \
                float("inf")

            d2[
                d2 > max_step2
            ] = float("inf")

            idx = int(
                np.argmin(
                    d2
                )
            )

            if not math.isfinite(
                float(
                    d2[idx]
                )
            ):

                break

            used[idx] = True

            order.append(
                idx
            )

            current = pts[
                idx
            ]

        return [

            tuple(
                pts[i]
            )

            for i in order
        ]

    # ================================================================
    # PUBLISH PATH
    # ================================================================

    def _publish_path(
        self,
        car_x,
        car_y,
        path_points
    ):

        msg = Path()

        msg.header.stamp = \
            self.get_clock().now().to_msg()

        # The planner operates in global/map coordinates.

        msg.header.frame_id = "map"

        poses = []

        # ------------------------------------------------------------
        # CAR FIRST
        # ------------------------------------------------------------

        car_pose = PoseStamped()

        car_pose.header = \
            msg.header

        car_pose.pose.position.x = \
            float(car_x)

        car_pose.pose.position.y = \
            float(car_y)

        car_pose.pose.position.z = 0.0

        car_pose.pose.orientation.w = 1.0

        poses.append(
            car_pose
        )

        # ------------------------------------------------------------
        # PATH POINTS
        # ------------------------------------------------------------

        for x, y in path_points:

            ps = PoseStamped()

            ps.header = \
                msg.header

            ps.pose.position.x = \
                float(x)

            ps.pose.position.y = \
                float(y)

            ps.pose.position.z = 0.0

            ps.pose.orientation.w = 1.0

            poses.append(
                ps
            )

        msg.poses = poses

        self.path_pub.publish(
            msg
        )

    # ================================================================
    # PLOT STATE UPDATE
    # ================================================================

    def _update_plot_state(
        self,
        car_x,
        car_y,
        car_yaw,
        raw_local_points,
        raw_local_classes,
        raw_global_points,
        raw_global_classes,
        graph_points,
        graph_classes,
        graph_edges,
        ellipse_data,
        virtual_cones,
        candidate_points,
        path_points,
        dominant_colour,
        mode
    ):

        with self.plot_lock:

            self.plot_data = {

                "car_x":
                    float(car_x),

                "car_y":
                    float(car_y),

                "car_yaw":
                    float(car_yaw),

                "raw_local_points":
                    list(raw_local_points),

                "raw_local_classes":
                    list(raw_local_classes),

                "raw_global_points":
                    list(raw_global_points),

                "raw_global_classes":
                    list(raw_global_classes),

                "graph_points":
                    list(graph_points),

                "graph_classes":
                    list(graph_classes),

                "graph_edges":
                    list(graph_edges),

                "ellipse_data":
                    list(ellipse_data),

                "virtual_cones":
                    list(virtual_cones),

                "candidate_points":
                    list(candidate_points),

                "path_points":
                    list(path_points),

                "dominant_colour":
                    dominant_colour,

                "mode":
                    mode,
            }

    # ================================================================
    # MAIN PLANNER TICK
    # ================================================================

    def _tick(self):

        # ============================================================
        # COPY SHARED DATA
        # ============================================================

        with self.data_lock:

            car_x = self.car_x
            car_y = self.car_y
            car_yaw = self.car_yaw

            blue_local = list(
                self.blue_local
            )

            yellow_local = list(
                self.yellow_local
            )

            orange_local = list(
                self.orange_local
            )

            big_local = list(
                self.big_local
            )

        # ============================================================
        # WAIT FOR ODOM
        # ============================================================

        if (
            car_x is None
            or
            car_y is None
            or
            car_yaw is None
        ):

            return

        # ============================================================
        # LOCAL GROUND-TRUTH CONES -> GLOBAL
        # ============================================================

        blue_global = \
            self._local_cones_to_global(
                car_x,
                car_y,
                car_yaw,
                blue_local
            )

        yellow_global = \
            self._local_cones_to_global(
                car_x,
                car_y,
                car_yaw,
                yellow_local
            )

        orange_global = \
            self._local_cones_to_global(
                car_x,
                car_y,
                car_yaw,
                orange_local
            )

        big_global = \
            self._local_cones_to_global(
                car_x,
                car_y,
                car_yaw,
                big_local
            )

        # ============================================================
        # ALL RAW GLOBAL POINTS
        # ============================================================

        raw_global_points = []

        raw_global_classes = []

        # Blue

        for p in blue_global:

            raw_global_points.append(
                p
            )

            raw_global_classes.append(
                "blue"
            )

        # Yellow

        for p in yellow_global:

            raw_global_points.append(
                p
            )

            raw_global_classes.append(
                "yellow"
            )

        # Orange

        for p in orange_global:

            raw_global_points.append(
                p
            )

            raw_global_classes.append(
                "orange"
            )

        # Big

        for p in big_global:

            raw_global_points.append(
                p
            )

            raw_global_classes.append(
                "big"
            )

        # ============================================================
        # ALL RAW LOCAL POINTS
        # ============================================================

        raw_local_points = []

        raw_local_classes = []

        for p in blue_local:

            raw_local_points.append(
                p
            )

            raw_local_classes.append(
                "blue"
            )

        for p in yellow_local:

            raw_local_points.append(
                p
            )

            raw_local_classes.append(
                "yellow"
            )

        for p in orange_local:

            raw_local_points.append(
                p
            )

            raw_local_classes.append(
                "orange"
            )

        for p in big_local:

            raw_local_points.append(
                p
            )

            raw_local_classes.append(
                "big"
            )

        # ============================================================
        # FOV
        # ============================================================

        (
            points_window,
            classes_window
        ) = self._compute_fov_points(

            car_x,
            car_y,
            car_yaw,

            blue_global,
            yellow_global,
            orange_global,
            big_global
        )

        # ============================================================
        # TOO FEW
        # ============================================================

        if len(
            points_window
        ) < 2:

            self._publish_path(
                car_x,
                car_y,
                []
            )

            self._update_plot_state(

                car_x,
                car_y,
                car_yaw,

                raw_local_points,
                raw_local_classes,

                raw_global_points,
                raw_global_classes,

                [],
                [],
                [],

                [],
                [],

                [],
                [],

                None,

                "WAITING FOR / TOO FEW FOV POINTS"
            )

            return

        # ============================================================
        # BLUE/YELLOW COUNTS
        # ============================================================

        blue_count = sum(

            1

            for cls
            in classes_window

            if cls == "blue"
        )

        yellow_count = sum(

            1

            for cls
            in classes_window

            if cls == "yellow"
        )

        # ============================================================
        # FALLBACK CONDITION
        # ============================================================

        fallback_needed = (

            blue_count <= 1
            or
            yellow_count <= 1
        )

        # ============================================================
        # DEFAULT GRAPH DATA
        # ============================================================

        graph_points = []
        graph_classes = []
        graph_edges = []

        ellipse_data = []
        virtual_cones = []

        dominant_colour = None

        # ============================================================
        # FALLBACK
        # ============================================================

        if fallback_needed:

            (
                graph_points,
                graph_classes,
                graph_edges,
                dominant_colour,
                virtual_cones,
                ellipse_data
            ) = self._fallback_graph(

                points_window,
                classes_window,

                car_x,
                car_y
            )

            # --------------------------------------------------------
            # EQUAL BLUE/YELLOW COUNTS
            # --------------------------------------------------------

            if dominant_colour is None:

                # No larger colour exists.
                # Revert to normal graph logic.

                graph_points = \
                    list(
                        points_window
                    )

                graph_classes = \
                    list(
                        classes_window
                    )

                graph_edges = \
                    self._build_edges(
                        points_window,
                        classes_window
                    )

                mode = (
                    "NORMAL - "
                    "NO DOMINANT COLOUR"
                )

            else:

                mode = (
                    "ELLIPSE FALLBACK - "
                    +
                    dominant_colour.upper()
                )

        # ============================================================
        # NORMAL
        # ============================================================

        else:

            graph_points = \
                list(
                    points_window
                )

            graph_classes = \
                list(
                    classes_window
                )

            graph_edges = \
                self._build_edges(
                    points_window,
                    classes_window
                )

            mode = "NORMAL"

        # ============================================================
        # RAW CANDIDATES
        # ============================================================

        centroids_raw = []

        for i, j in graph_edges:

            x1, y1 = \
                graph_points[i]

            x2, y2 = \
                graph_points[j]

            ci = \
                graph_classes[i]

            cj = \
                graph_classes[j]

            # Big-big is handled by the special logic

            if (
                ci == "big"
                and
                cj == "big"
            ):

                continue

            mx = 0.5 * (
                x1 + x2
            )

            my = 0.5 * (
                y1 + y2
            )

            centroids_raw.append(

                (
                    mx,
                    my,
                    False
                )
            )

        # ============================================================
        # BIG SPECIAL
        # ============================================================

        big_special = \
            self._big_cone_candidate(

                points_window,
                classes_window
            )

        for bx, by, _ in big_special:

            centroids_raw.append(

                (
                    bx,
                    by,
                    True
                )
            )

        # ============================================================
        # FILTER
        # ============================================================

        centroids = \
            self._filter_centroids_min_spacing(

                centroids_raw,

                car_x,
                car_y,

                self.min_spacing
            )

        # ============================================================
        # CANDIDATE POINTS
        # ============================================================

        candidate_points = [

            (
                x,
                y
            )

            for x, y, _ in centroids
        ]

        # ============================================================
        # GREEDY PATH
        # ============================================================

        path_points = \
            self._build_greedy_path(

                car_x,
                car_y,

                candidate_points
            )

        # ============================================================
        # PUBLISH PATH
        # ============================================================

        self._publish_path(

            car_x,
            car_y,

            path_points
        )

        # ============================================================
        # PLOT STATE
        # ============================================================

        self._update_plot_state(

            car_x,
            car_y,
            car_yaw,

            raw_local_points,
            raw_local_classes,

            raw_global_points,
            raw_global_classes,

            graph_points,
            graph_classes,
            graph_edges,

            ellipse_data,
            virtual_cones,

            candidate_points,
            path_points,

            dominant_colour,

            mode
        )

    # ================================================================
    # INITIALISE PLOT
    # ================================================================

    def init_plot(self):

        if not _HAS_MATPLOTLIB:

            return

        plt.ion()

        self.fig, (
            self.ax_global,
            self.ax_local
        ) = plt.subplots(
            1,
            2,
            figsize=(16, 8)
        )

        try:

            self.fig.canvas.manager.set_window_title(
                "Ground Truth Path Planner - Live Visualisation"
            )

        except Exception:

            pass

        # Global

        self.ax_global.set_title(
            "GLOBAL / MAP FRAME"
        )

        self.ax_global.set_xlabel(
            "X [m]"
        )

        self.ax_global.set_ylabel(
            "Y [m]"
        )

        self.ax_global.grid(
            True,
            alpha=0.25
        )

        self.ax_global.set_aspect(
            "equal",
            adjustable="box"
        )

        # Local

        self.ax_local.set_title(
            "LOCAL / CAR FRAME"
        )

        self.ax_local.set_xlabel(
            "X FORWARD [m]"
        )

        self.ax_local.set_ylabel(
            "Y LEFT [m]"
        )

        self.ax_local.grid(
            True,
            alpha=0.25
        )

        self.ax_local.set_aspect(
            "equal",
            adjustable="box"
        )

        plt.show(
            block=False
        )

    # ================================================================
    # GLOBAL/LOCAL TRANSFORM HELPER
    # ================================================================

    def _global_points_to_local(
        self,
        points,
        car_x,
        car_y,
        car_yaw
    ):

        local = []

        for px, py in points:

            lx, ly = \
                self._world_to_local(

                    car_x,
                    car_y,
                    car_yaw,

                    px,
                    py
                )

            local.append(
                (
                    lx,
                    ly
                )
            )

        return local

    # ================================================================
    # PLOT CONES
    # ================================================================

    def _plot_cones(
        self,
        ax,
        points,
        classes
    ):

        groups = {

            "blue": [],
            "yellow": [],
            "orange": [],
            "big": [],
        }

        for p, cls in zip(
            points,
            classes
        ):

            if cls in groups:

                groups[cls].append(
                    p
                )

        # Blue

        if groups["blue"]:

            ax.scatter(

                [
                    p[0]
                    for p in groups["blue"]
                ],

                [
                    p[1]
                    for p in groups["blue"]
                ],

                s=45,
                c=BLUE_COLOR,
                marker="o",
                label="Blue"
            )

        # Yellow

        if groups["yellow"]:

            ax.scatter(

                [
                    p[0]
                    for p in groups["yellow"]
                ],

                [
                    p[1]
                    for p in groups["yellow"]
                ],

                s=45,
                c=YELLOW_COLOR,
                marker="o",
                label="Yellow"
            )

        # Orange

        if groups["orange"]:

            ax.scatter(

                [
                    p[0]
                    for p in groups["orange"]
                ],

                [
                    p[1]
                    for p in groups["orange"]
                ],

                s=55,
                c=ORANGE_COLOR,
                marker="o",
                label="Orange"
            )

        # Big orange

        if groups["big"]:

            ax.scatter(

                [
                    p[0]
                    for p in groups["big"]
                ],

                [
                    p[1]
                    for p in groups["big"]
                ],

                s=85,
                c=BIG_COLOR,
                marker="s",
                label="Big orange"
            )

    # ================================================================
    # PLOT ELLIPSE
    # ================================================================

    def _plot_ellipse_geometry(
        self,
        ax,
        ellipse_data
    ):

        for e in ellipse_data:

            p1 = e["p1"]
            p2 = e["p2"]
            center = e["center"]
            vp = e["v_plus"]
            vm = e["v_minus"]
            selected = e["selected"]

            # --------------------------------------------------------
            # MINOR AXIS
            # --------------------------------------------------------

            ax.plot(

                [p1[0], p2[0]],
                [p1[1], p2[1]],

                linestyle="--",
                linewidth=1.6,
                color=ELLIPSE_COLOR
            )

            # --------------------------------------------------------
            # CENTRE
            # --------------------------------------------------------

            ax.scatter(

                [center[0]],
                [center[1]],

                s=25,
                marker="+",
                color=ELLIPSE_COLOR
            )

            # --------------------------------------------------------
            # MAJOR AXIS
            # --------------------------------------------------------

            ax.plot(

                [vm[0], vp[0]],
                [vm[1], vp[1]],

                linestyle=":",
                linewidth=1.4,
                color=ELLIPSE_COLOR
            )

            # --------------------------------------------------------
            # BOTH POSSIBLE ENDS
            # --------------------------------------------------------

            ax.scatter(

                [vp[0], vm[0]],
                [vp[1], vm[1]],

                s=38,
                marker="o",
                facecolors="none",
                edgecolors=ELLIPSE_COLOR
            )

            # --------------------------------------------------------
            # SELECTED VIRTUAL CONE
            # --------------------------------------------------------

            ax.scatter(

                [selected[0]],
                [selected[1]],

                s=90,
                marker="D",
                color=VIRTUAL_COLOR,
                edgecolors="black",
                linewidths=0.5
            )

    # ================================================================
    # PLOT CAR
    # ================================================================

    def _plot_car(
        self,
        ax,
        x,
        y,
        yaw
    ):

        ax.scatter(

            [x],
            [y],

            s=150,
            marker="*",
            color=CAR_COLOR,
            label="Car",
            zorder=10
        )

        arrow_length = 2.0

        ax.arrow(

            x,
            y,

            arrow_length * math.cos(yaw),
            arrow_length * math.sin(yaw),

            head_width=0.25,
            head_length=0.35,
            length_includes_head=True,

            color=CAR_COLOR
        )

    # ================================================================
    # PLOT FOV IN LOCAL
    # ================================================================

    def _plot_local_fov(
        self,
        ax
    ):

        apex = np.array(
            [
                self.vertex_offset,
                0.0
            ]
        )

        angles = np.linspace(
            -self.fov_half_rad,
            self.fov_half_rad,
            80
        )

        arc = np.column_stack(

            (
                apex[0]
                +
                self.fov_radius
                *
                np.cos(
                    angles
                ),

                apex[1]
                +
                self.fov_radius
                *
                np.sin(
                    angles
                )
            )
        )

        polygon = np.vstack(

            (
                apex,
                arc,
                apex
            )
        )

        ax.plot(

            polygon[:, 0],
            polygon[:, 1],

            color=FOV_COLOR,
            linewidth=1.0,
            linestyle="-",
            label="FOV"
        )

    # ================================================================
    # UPDATE PLOT
    # ================================================================

    def update_plot(self):

        if not _HAS_MATPLOTLIB:

            return

        # ------------------------------------------------------------
        # COPY DATA
        # ------------------------------------------------------------

        with self.plot_lock:

            d = dict(
                self.plot_data
            )

        car_x = d["car_x"]
        car_y = d["car_y"]
        car_yaw = d["car_yaw"]

        raw_local_points = \
            d["raw_local_points"]

        raw_local_classes = \
            d["raw_local_classes"]

        raw_global_points = \
            d["raw_global_points"]

        raw_global_classes = \
            d["raw_global_classes"]

        graph_points = \
            d["graph_points"]

        graph_classes = \
            d["graph_classes"]

        graph_edges = \
            d["graph_edges"]

        ellipse_data = \
            d["ellipse_data"]

        virtual_cones = \
            d["virtual_cones"]

        candidate_points = \
            d["candidate_points"]

        path_points = \
            d["path_points"]

        dominant_colour = \
            d["dominant_colour"]

        mode = d["mode"]

        # ============================================================
        # CLEAR BOTH AXES
        # ============================================================

        self.ax_global.clear()
        self.ax_local.clear()

        # ============================================================
        # GLOBAL FRAME
        # ============================================================

        ax = self.ax_global

        ax.set_title(
            f"GLOBAL / MAP FRAME\n{mode}"
        )

        ax.set_xlabel(
            "X [m]"
        )

        ax.set_ylabel(
            "Y [m]"
        )

        ax.grid(
            True,
            alpha=0.25
        )

        ax.set_aspect(
            "equal",
            adjustable="box"
        )

        # ------------------------------------------------------------
        # RAW GROUND-TRUTH CONES
        # ------------------------------------------------------------

        self._plot_cones(

            ax,

            raw_global_points,
            raw_global_classes
        )

        # ------------------------------------------------------------
        # GRAPH EDGES
        # ------------------------------------------------------------

        for i, j in graph_edges:

            if (
                i >= len(graph_points)
                or
                j >= len(graph_points)
            ):

                continue

            x1, y1 = \
                graph_points[i]

            x2, y2 = \
                graph_points[j]

            ax.plot(

                [x1, x2],
                [y1, y2],

                linestyle="-",
                linewidth=1.0,
                color=EDGE_COLOR
            )

        # ------------------------------------------------------------
        # ELLIPSE GEOMETRY
        # ------------------------------------------------------------

        self._plot_ellipse_geometry(
            ax,
            ellipse_data
        )

        # ------------------------------------------------------------
        # VIRTUAL CONES
        # ------------------------------------------------------------

        if virtual_cones:

            ax.scatter(

                [
                    p[0]
                    for p in virtual_cones
                ],

                [
                    p[1]
                    for p in virtual_cones
                ],

                s=90,
                marker="D",
                color=VIRTUAL_COLOR,
                edgecolors="black",
                label="Virtual cone",
                zorder=8
            )

        # ------------------------------------------------------------
        # CANDIDATES
        # ------------------------------------------------------------

        if candidate_points:

            ax.scatter(

                [
                    p[0]
                    for p in candidate_points
                ],

                [
                    p[1]
                    for p in candidate_points
                ],

                s=60,
                marker="x",
                color=CANDIDATE_COLOR,
                linewidths=1.7,
                label="Candidates",
                zorder=9
            )

        # ------------------------------------------------------------
        # PATH
        # ------------------------------------------------------------

        if path_points:

            path_x = [
                car_x
            ]

            path_y = [
                car_y
            ]

            path_x.extend(
                p[0]
                for p in path_points
            )

            path_y.extend(
                p[1]
                for p in path_points
            )

            ax.plot(

                path_x,
                path_y,

                color=PATH_COLOR,
                linewidth=2.5,
                marker="o",
                markersize=3,
                label="Published path",
                zorder=8
            )

        # ------------------------------------------------------------
        # CAR
        # ------------------------------------------------------------

        self._plot_car(

            ax,

            car_x,
            car_y,
            car_yaw
        )

        # ------------------------------------------------------------
        # GLOBAL AXIS LIMITS
        # ------------------------------------------------------------

        all_global = []

        all_global.extend(
            raw_global_points
        )

        all_global.extend(
            graph_points
        )

        all_global.extend(
            candidate_points
        )

        all_global.extend(
            path_points
        )

        all_global.append(
            (
                car_x,
                car_y
            )
        )

        if all_global:

            arr = np.asarray(
                all_global,
                dtype=float
            )

            margin = 5.0

            xmin = np.min(
                arr[:, 0]
            ) - margin

            xmax = np.max(
                arr[:, 0]
            ) + margin

            ymin = np.min(
                arr[:, 1]
            ) - margin

            ymax = np.max(
                arr[:, 1]
            ) + margin

            # Protect against zero-size axis

            if (
                abs(xmax - xmin)
                <
                1.0
            ):

                xmin -= 5.0
                xmax += 5.0

            if (
                abs(ymax - ymin)
                <
                1.0
            ):

                ymin -= 5.0
                ymax += 5.0

            ax.set_xlim(
                xmin,
                xmax
            )

            ax.set_ylim(
                ymin,
                ymax
            )

        ax.legend(
            fontsize=7,
            loc="best"
        )

        # ============================================================
        # LOCAL FRAME
        # ============================================================

        ax = self.ax_local

        ax.set_title(
            "LOCAL / CAR FRAME"
        )

        ax.set_xlabel(
            "X FORWARD [m]"
        )

        ax.set_ylabel(
            "Y LEFT [m]"
        )

        ax.grid(
            True,
            alpha=0.25
        )

        ax.set_aspect(
            "equal",
            adjustable="box"
        )

        # ------------------------------------------------------------
        # RAW LOCAL GT CONES
        # ------------------------------------------------------------

        self._plot_cones(

            ax,

            raw_local_points,
            raw_local_classes
        )

        # ------------------------------------------------------------
        # GRAPH -> LOCAL
        # ------------------------------------------------------------

        local_graph = \
            self._global_points_to_local(

                graph_points,

                car_x,
                car_y,

                car_yaw
            )

        # ------------------------------------------------------------
        # GRAPH EDGES
        # ------------------------------------------------------------

        for i, j in graph_edges:

            if (
                i >= len(local_graph)
                or
                j >= len(local_graph)
            ):

                continue

            x1, y1 = \
                local_graph[i]

            x2, y2 = \
                local_graph[j]

            ax.plot(

                [x1, x2],
                [y1, y2],

                linestyle="-",
                linewidth=1.0,
                color=EDGE_COLOR
            )

        # ------------------------------------------------------------
        # ELLIPSES -> LOCAL
        # ------------------------------------------------------------

        for e in ellipse_data:

            p1 = self._world_to_local(
                car_x,
                car_y,
                car_yaw,
                e["p1"][0],
                e["p1"][1]
            )

            p2 = self._world_to_local(
                car_x,
                car_y,
                car_yaw,
                e["p2"][0],
                e["p2"][1]
            )

            center = self._world_to_local(
                car_x,
                car_y,
                car_yaw,
                e["center"][0],
                e["center"][1]
            )

            vp = self._world_to_local(
                car_x,
                car_y,
                car_yaw,
                e["v_plus"][0],
                e["v_plus"][1]
            )

            vm = self._world_to_local(
                car_x,
                car_y,
                car_yaw,
                e["v_minus"][0],
                e["v_minus"][1]
            )

            selected = self._world_to_local(
                car_x,
                car_y,
                car_yaw,
                e["selected"][0],
                e["selected"][1]
            )

            # Minor axis

            ax.plot(

                [p1[0], p2[0]],
                [p1[1], p2[1]],

                linestyle="--",
                linewidth=1.6,
                color=ELLIPSE_COLOR
            )

            # Centre

            ax.scatter(

                [center[0]],
                [center[1]],

                marker="+",
                s=30,
                color=ELLIPSE_COLOR
            )

            # Major axis

            ax.plot(

                [vm[0], vp[0]],
                [vm[1], vp[1]],

                linestyle=":",
                linewidth=1.4,
                color=ELLIPSE_COLOR
            )

            # Both ends

            ax.scatter(

                [vp[0], vm[0]],
                [vp[1], vm[1]],

                s=35,
                marker="o",
                facecolors="none",
                edgecolors=ELLIPSE_COLOR
            )

            # Selected virtual cone

            ax.scatter(

                [selected[0]],
                [selected[1]],

                s=90,
                marker="D",
                color=VIRTUAL_COLOR,
                edgecolors="black",
                zorder=8
            )

        # ------------------------------------------------------------
        # VIRTUAL CONES
        # ------------------------------------------------------------

        local_virtual = \
            self._global_points_to_local(

                virtual_cones,

                car_x,
                car_y,

                car_yaw
            )

        if local_virtual:

            ax.scatter(

                [
                    p[0]
                    for p in local_virtual
                ],

                [
                    p[1]
                    for p in local_virtual
                ],

                s=90,
                marker="D",
                color=VIRTUAL_COLOR,
                edgecolors="black",
                label="Virtual cone",
                zorder=9
            )

        # ------------------------------------------------------------
        # CANDIDATES
        # ------------------------------------------------------------

        local_candidates = \
            self._global_points_to_local(

                candidate_points,

                car_x,
                car_y,

                car_yaw
            )

        if local_candidates:

            ax.scatter(

                [
                    p[0]
                    for p in local_candidates
                ],

                [
                    p[1]
                    for p in local_candidates
                ],

                s=60,
                marker="x",
                color=CANDIDATE_COLOR,
                linewidths=1.7,
                label="Candidates",
                zorder=10
            )

        # ------------------------------------------------------------
        # PATH
        # ------------------------------------------------------------

        local_path = \
            self._global_points_to_local(

                path_points,

                car_x,
                car_y,

                car_yaw
            )

        if local_path:

            path_x = [
                0.0
            ]

            path_y = [
                0.0
            ]

            path_x.extend(
                p[0]
                for p in local_path
            )

            path_y.extend(
                p[1]
                for p in local_path
            )

            ax.plot(

                path_x,
                path_y,

                color=PATH_COLOR,
                linewidth=2.5,
                marker="o",
                markersize=3,
                label="Published path",
                zorder=8
            )

        # ------------------------------------------------------------
        # LOCAL CAR
        # ------------------------------------------------------------

        ax.scatter(

            [0.0],
            [0.0],

            s=150,
            marker="*",
            color=CAR_COLOR,
            label="Car",
            zorder=10
        )

        ax.arrow(

            0.0,
            0.0,

            2.0,
            0.0,

            head_width=0.25,
            head_length=0.35,
            length_includes_head=True,

            color=CAR_COLOR
        )

        # ------------------------------------------------------------
        # FOV
        # ------------------------------------------------------------

        self._plot_local_fov(
            ax
        )

        # ------------------------------------------------------------
        # LOCAL AXIS LIMITS
        # ------------------------------------------------------------

        all_local = []

        all_local.extend(
            raw_local_points
        )

        all_local.extend(
            local_graph
        )

        all_local.extend(
            local_candidates
        )

        all_local.extend(
            local_path
        )

        if all_local:

            arr = np.asarray(
                all_local,
                dtype=float
            )

            xmin = np.min(
                arr[:, 0]
            )

            xmax = np.max(
                arr[:, 0]
            )

            ymin = np.min(
                arr[:, 1]
            )

            ymax = np.max(
                arr[:, 1]
            )

            xmin = min(
                xmin - 2.0,
                -5.0
            )

            xmax = max(
                xmax + 2.0,
                8.0
            )

            yabs = max(
                abs(ymin),
                abs(ymax),
                5.0
            )

            ax.set_xlim(
                xmin,
                xmax
            )

            ax.set_ylim(
                -yabs - 1.0,
                yabs + 1.0
            )

        else:

            ax.set_xlim(
                -5.0,
                15.0
            )

            ax.set_ylim(
                -10.0,
                10.0
            )

        ax.legend(
            fontsize=7,
            loc="best"
        )

        # ------------------------------------------------------------
        # STATUS BOX
        # ------------------------------------------------------------

        status = (

            f"Mode: {mode}\n"
            f"Blue: {sum(c == 'blue' for c in raw_local_classes)}\n"
            f"Yellow: {sum(c == 'yellow' for c in raw_local_classes)}\n"
            f"FOV points: {len(graph_points)}\n"
            f"Graph edges: {len(graph_edges)}\n"
            f"Virtual cones: {len(virtual_cones)}\n"
            f"Candidates: {len(candidate_points)}\n"
            f"Path points: {len(path_points)}"
        )

        ax.text(

            0.02,
            0.98,

            status,

            transform=ax.transAxes,

            va="top",

            fontsize=8,

            bbox=dict(

                boxstyle="round",

                facecolor="white",

                edgecolor="0.6",

                alpha=0.85
            )
        )

        # ============================================================
        # REFRESH
        # ============================================================

        self.fig.tight_layout()

        self.fig.canvas.draw_idle()

        self.fig.canvas.flush_events()

    # ================================================================
    # PLOT LOOP
    # ================================================================

    def plot_loop(self):

        if not _HAS_MATPLOTLIB:

            return

        self.init_plot()

        while rclpy.ok():

            if not plt.fignum_exists(
                self.fig.number
            ):

                break

            self.update_plot()

            plt.pause(
                0.05
            )

    # ================================================================
    # MAIN
    # ================================================================


def main():

    rclpy.init()

    node = \
        SlamPathSectorVisualizer()

    # ============================================================
    # ROS EXECUTOR
    # ============================================================

    executor = MultiThreadedExecutor(
        num_threads=2
    )

    executor.add_node(
        node
    )

    ros_thread = threading.Thread(

        target=executor.spin,

        daemon=True
    )

    ros_thread.start()

    try:

        # Matplotlib MUST run in main thread

        if _HAS_MATPLOTLIB:

            node.plot_loop()

        else:

            while rclpy.ok():

                threading.Event().wait(
                    1.0
                )

    except KeyboardInterrupt:

        pass

    finally:

        try:

            executor.shutdown()

        except Exception:

            pass

        try:

            node.destroy_node()

        except Exception:

            pass

        if rclpy.ok():

            rclpy.shutdown()

        ros_thread.join(
            timeout=2.0
        )


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":

    main()