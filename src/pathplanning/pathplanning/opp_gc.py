#another program 


# ...........improved logic placing virtual at the opposite end of the lane 
# ...........challenges: this program not receiving the cone positions fro slam.



#!/usr/bin/env python3

# ================================================================
# SLAM PATH SECTOR VISUALISER
#
# ROS2 planner + live Matplotlib visualisation
#
# NORMAL MODE:
#   - FOV filtering
#   - Delaunay / k-NN
#   - Original edge constraints
#   - Edge midpoints as candidates
#   - Big-orange special candidate
#   - Candidate spacing filtering
#   - Greedy path
#
# FALLBACK MODE:
#   Triggered when BLUE or YELLOW count <= 1.
#
#   1. Count blue and yellow in FOV.
#   2. Select colour with larger count.
#   3. Spatially order dominant-colour cones.
#   4. Form alternate pairs:
#
#         (C0, C2)
#         (C1, C3)
#         (C2, C4)
#         ...
#
#   5. Each pair is the MINOR AXIS.
#   6. Midpoint of pair = ellipse centre.
#   7. Major axis is perpendicular to minor axis.
#   8. Half major-axis length = 1 m.
#   9. Construct both possible major-axis ends.
#  10. Select the end closer to the car.
#  11. Selected point becomes virtual cone.
#  12. Add virtual cones to dominant-colour cones.
#  13. Run Delaunay / k-NN.
#
# PLOT:
#   Global frame + local car frame
#   updated continuously.
#
# ================================================================

import math
import threading
from typing import List, Tuple

import numpy as np

import rclpy
from rclpy.node import Node

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
    from matplotlib.patches import Polygon

    _HAS_MATPLOTLIB = True

except ImportError:
    plt = None
    Polygon = None
    _HAS_MATPLOTLIB = False


# ================================================================
# QUATERNION -> YAW
# ================================================================

def yaw_from_quat(qx, qy, qz, qw):

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

        self.declare_parameter(
            "topics.odom_in",
            "/slam/odom"
        )

        self.declare_parameter(
            "topics.map_in",
            "/slam/map_cones"
        )

        self.declare_parameter(
            "topics.path_out",
            "/path_points"
        )

        self.declare_parameter(
            "qos.best_effort",
            True
        )

        self.declare_parameter(
            "qos.depth",
            50
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
        # CANDIDATES
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
        # GREEDY
        # ------------------------------------------------------------

        self.declare_parameter(
            "greedy.max_hop_m",
            7.0
        )

        # ------------------------------------------------------------
        # ELLIPSE
        # ------------------------------------------------------------

        self.declare_parameter(
            "ellipse.major_axis_half_length_m",
            1.0
        )

        # ------------------------------------------------------------
        # PUBLISH
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
            gp("topics.odom_in").value
        )

        map_topic = str(
            gp("topics.map_in").value
        )

        path_topic = str(
            gp("topics.path_out").value
        )

        best_effort = bool(
            gp("qos.best_effort").value
        )

        depth = int(
            gp("qos.depth").value
        )

        # FOV

        self.vertex_offset = float(
            gp("fov.vertex_offset_m").value
        )

        self.fov_radius = float(
            gp("fov.radius_m").value
        )

        self.fov_angle_deg = float(
            gp("fov.angle_deg").value
        )

        self.fov_half_rad = math.radians(
            self.fov_angle_deg * 0.5
        )

        # Candidate filtering

        self.min_spacing = float(
            gp("candidates.min_spacing_m").value
        )

        self.min_car_dist = float(
            gp("candidates.min_car_dist_m").value
        )

        # Greedy

        self.max_hop = float(
            gp("greedy.max_hop_m").value
        )

        # Ellipse

        self.ellipse_major_half_length = float(
            gp(
                "ellipse.major_axis_half_length_m"
            ).value
        )

        # Publish

        self.publish_rate_hz = float(
            gp("publish.rate_hz").value
        )

        # ============================================================
        # QOS
        # ============================================================

        sub_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=depth,
            reliability=(
                QoSReliabilityPolicy.BEST_EFFORT
                if best_effort
                else QoSReliabilityPolicy.RELIABLE
            ),
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        pub_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=depth,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # ============================================================
        # ROS SUBSCRIBERS
        # ============================================================

        self.create_subscription(
            Odometry,
            odom_topic,
            self.cb_odom,
            sub_qos
        )

        self.create_subscription(
            ConeArrayWithCovariance,
            map_topic,
            self.cb_map,
            sub_qos
        )

        # ============================================================
        # ROS PUBLISHER
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

        self.car_x = None
        self.car_y = None
        self.car_yaw = None

        self.blue_global = []
        self.yellow_global = []
        self.orange_global = []
        self.big_global = []

        # ============================================================
        # DEBUG / PLOT STATE
        # ============================================================

        self.plot_data_lock = threading.Lock()

        self.plot_data = {

            "car_x": 0.0,
            "car_y": 0.0,
            "car_yaw": 0.0,

            "raw_points": [],
            "raw_classes": [],

            "graph_points": [],
            "graph_classes": [],
            "graph_edges": [],

            "virtual_cones": [],

            "candidate_points": [],

            "path_points": [],

            "dominant_colour": None,

            "mode": "WAITING",
        }

        # ============================================================
        # ROS TIMER
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
        # INFO
        # ============================================================

        self.get_logger().info(
            "================================================"
        )

        self.get_logger().info(
            "SLAM PATH SECTOR VISUALISER STARTED"
        )

        self.get_logger().info(
            f"odom: {odom_topic}"
        )

        self.get_logger().info(
            f"map:  {map_topic}"
        )

        self.get_logger().info(
            f"path: {path_topic}"
        )

        self.get_logger().info(
            f"ellipse half major axis: "
            f"{self.ellipse_major_half_length} m"
        )

        self.get_logger().info(
            "================================================"
        )

        if not _HAS_SCIPY:

            self.get_logger().warn(
                "SciPy not available. "
                "Using k-NN graph."
            )

        if not _HAS_MATPLOTLIB:

            self.get_logger().error(
                "Matplotlib is NOT installed. "
                "The ROS planner will run, "
                "but no plot can be displayed."
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
    # CONE MAP CALLBACK
    # ================================================================

    def cb_map(
        self,
        msg: ConeArrayWithCovariance
    ):

        blue = []
        yellow = []
        orange = []
        big = []

        # Blue

        for c in msg.blue_cones:

            blue.append(
                (
                    float(c.point.x),
                    float(c.point.y)
                )
            )

        # Yellow

        for c in msg.yellow_cones:

            yellow.append(
                (
                    float(c.point.x),
                    float(c.point.y)
                )
            )

        # Orange

        for c in msg.orange_cones:

            orange.append(
                (
                    float(c.point.x),
                    float(c.point.y)
                )
            )

        # Big orange

        for c in msg.big_orange_cones:

            big.append(
                (
                    float(c.point.x),
                    float(c.point.y)
                )
            )

        with self.data_lock:

            self.blue_global = blue
            self.yellow_global = yellow
            self.orange_global = orange
            self.big_global = big

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

        R = self.fov_radius

        half_angle = \
            self.fov_half_rad

        vertex_offset = \
            self.vertex_offset

        points_window = []
        classes_window = []

        # ------------------------------------------------------------

        def add_list(
            cones,
            cls_name
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
                        cls_name
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

        N = len(points)

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

                self.get_logger().debug(
                    "Delaunay failed due to "
                    "degenerate geometry; "
                    "using k-NN."
                )

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

                nn_idx = np.argsort(
                    d2
                )[:k]

                for j in nn_idx:

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

        if len(points_window) < 2:
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
    # ORDER POINTS ALONG PRINCIPAL AXIS
    # ================================================================

    def _order_points_along_principal_axis(
        self,
        points
    ):

        if len(points) <= 2:

            return list(points)

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
    # ELLIPSE VIRTUAL CONES
    # ================================================================

    def _build_ellipse_virtual_cones(
        self,
        dominant_points,
        car_x,
        car_y
    ):

        # Need C0, C2 at minimum

        if len(
            dominant_points
        ) < 3:

            return []

        ordered = \
            self._order_points_along_principal_axis(
                dominant_points
            )

        virtual_cones = []

        half_major = \
            self.ellipse_major_half_length

        car = np.array(
            [
                car_x,
                car_y
            ],
            dtype=float
        )

        # ------------------------------------------------------------
        # ALTERNATE PAIRS
        #
        # C0-C2
        # C1-C3
        # C2-C4
        # ...
        # ------------------------------------------------------------

        for i in range(
            len(ordered) - 2
        ):

            p1 = np.asarray(
                ordered[i],
                dtype=float
            )

            p2 = np.asarray(
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
            #
            # perpendicular to minor axis
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

            center = \
                0.5 * (
                    p1
                    +
                    p2
                )

            # --------------------------------------------------------
            # BOTH POSSIBLE ENDS
            # --------------------------------------------------------

            virtual_plus = (
                center
                +
                half_major
                *
                major_dir
            )

            virtual_minus = (
                center
                -
                half_major
                *
                major_dir
            )

            # --------------------------------------------------------
            # DISTANCE FROM CAR
            # --------------------------------------------------------

            d_plus2 = float(
                np.sum(
                    (
                        virtual_plus
                        -
                        car
                    ) ** 2
                )
            )

            d_minus2 = float(
                np.sum(
                    (
                        virtual_minus
                        -
                        car
                    ) ** 2
                )
            )

            # --------------------------------------------------------
            # SELECT CLOSER END
            # --------------------------------------------------------

            if (
                d_plus2
                <=
                d_minus2
            ):

                selected = \
                    virtual_plus

            else:

                selected = \
                    virtual_minus

            virtual_cones.append(
                (
                    float(
                        selected[0]
                    ),

                    float(
                        selected[1]
                    )
                )
            )

        return virtual_cones

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
        # SPLIT COLOURS
        # ------------------------------------------------------------

        blue_points = [

            points_window[i]

            for i, cls
            in enumerate(
                classes_window
            )

            if cls == "blue"

        ]

        yellow_points = [

            points_window[i]

            for i, cls
            in enumerate(
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
        # SELECT DOMINANT COLOUR
        # ------------------------------------------------------------

        if blue_count > yellow_count:

            dominant_colour = \
                "blue"

            dominant_points = \
                blue_points

        elif yellow_count > blue_count:

            dominant_colour = \
                "yellow"

            dominant_points = \
                yellow_points

        else:

            return (
                [],
                [],
                [],
                None,
                []
            )

        # ------------------------------------------------------------
        # ELLIPSE VIRTUAL CONES
        # ------------------------------------------------------------

        virtual_cones = \
            self._build_ellipse_virtual_cones(
                dominant_points,
                car_x,
                car_y
            )

        # ------------------------------------------------------------
        # GRAPH = DOMINANT + VIRTUAL
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

        if len(
            graph_points
        ) < 2:

            return (
                graph_points,
                graph_classes,
                [],
                dominant_colour,
                virtual_cones
            )

        # ------------------------------------------------------------
        # DELAUNAY / KNN
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
            virtual_cones
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

            for i, cls
            in enumerate(
                classes_window
            )

            if cls == "big"

        ]

        n = len(
            big_points
        )

        if n < 2:

            return []

        # Exactly two

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

        # Three or more

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

        pts = np.array(

            [
                [
                    c[0],
                    c[1]
                ]

                for c in centroids_raw
            ],

            dtype=float

        )

        car = np.array(
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

        # Farthest first

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

        current = np.array(
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

        msg.header.frame_id = \
            "map"

        poses = []

        # ------------------------------------------------------------
        # CAR
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
        # PATH
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
    # SAVE DEBUG DATA FOR PLOT
    # ================================================================

    def _update_plot_state(
        self,
        car_x,
        car_y,
        car_yaw,
        raw_points,
        raw_classes,
        graph_points,
        graph_classes,
        graph_edges,
        virtual_cones,
        candidate_points,
        path_points,
        dominant_colour,
        mode
    ):

        with self.plot_data_lock:

            self.plot_data = {

                "car_x": float(
                    car_x
                ),

                "car_y": float(
                    car_y
                ),

                "car_yaw": float(
                    car_yaw
                ),

                "raw_points": list(
                    raw_points
                ),

                "raw_classes": list(
                    raw_classes
                ),

                "graph_points": list(
                    graph_points
                ),

                "graph_classes": list(
                    graph_classes
                ),

                "graph_edges": list(
                    graph_edges
                ),

                "virtual_cones": list(
                    virtual_cones
                ),

                "candidate_points": list(
                    candidate_points
                ),

                "path_points": list(
                    path_points
                ),

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
        # COPY STATE
        # ============================================================

        with self.data_lock:

            car_x = self.car_x
            car_y = self.car_y
            car_yaw = self.car_yaw

            blue_global = list(
                self.blue_global
            )

            yellow_global = list(
                self.yellow_global
            )

            orange_global = list(
                self.orange_global
            )

            big_global = list(
                self.big_global
            )

        # ============================================================
        # NO ODOMETRY
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

                points_window,
                classes_window,

                [],
                [],
                [],

                [],

                [],

                [],

                None,

                "TOO FEW POINTS"
            )

            return

        # ============================================================
        # COUNTS
        # ============================================================

        blue_count = sum(

            cls == "blue"

            for cls in classes_window
        )

        yellow_count = sum(

            cls == "yellow"

            for cls in classes_window
        )

        # ============================================================
        # FALLBACK?
        # ============================================================

        fallback_needed = (

            blue_count <= 1

            or

            yellow_count <= 1
        )

        # ============================================================
        # VARIABLES
        # ============================================================

        graph_points = []
        graph_classes = []
        edges = []

        virtual_cones = []

        dominant_colour = None

        mode = "NORMAL"

        # ============================================================
        # FALLBACK
        # ============================================================

        if fallback_needed:

            (
                graph_points,
                graph_classes,
                edges,
                dominant_colour,
                virtual_cones

            ) = self._fallback_graph(

                points_window,
                classes_window,

                car_x,
                car_y
            )

            # --------------------------------------------------------
            # NO DOMINANT COLOUR
            # --------------------------------------------------------

            if dominant_colour is None:

                # Equal counts -> no larger colour exists.
                #
                # Use normal graph.

                graph_points = \
                    list(points_window)

                graph_classes = \
                    list(classes_window)

                edges = \
                    self._build_edges(

                        points_window,
                        classes_window
                    )

                mode = \
                    "NORMAL (NO DOMINANT COLOUR)"

            else:

                mode = (
                    "ELLIPSE FALLBACK: "
                    +
                    dominant_colour.upper()
                )

        # ============================================================
        # NORMAL
        # ============================================================

        else:

            graph_points = \
                list(points_window)

            graph_classes = \
                list(classes_window)

            edges = \
                self._build_edges(

                    points_window,
                    classes_window
                )

            mode = "NORMAL"

        # ============================================================
        # CANDIDATES FROM GRAPH EDGES
        # ============================================================

        centroids_raw = []

        for i, j in edges:

            x1, y1 = \
                graph_points[i]

            x2, y2 = \
                graph_points[j]

            ci = \
                graph_classes[i]

            cj = \
                graph_classes[j]

            # Big-big handled separately

            if (
                ci == "big"
                and
                cj == "big"
            ):

                continue

            mx = \
                0.5 * (
                    x1 + x2
                )

            my = \
                0.5 * (
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
        # BIG ORANGE SPECIAL
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
        # GREEDY
        # ============================================================

        path_points = \
            self._build_greedy_path(

                car_x,
                car_y,

                candidate_points
            )

        # ============================================================
        # PUBLISH
        # ============================================================

        self._publish_path(

            car_x,
            car_y,

            path_points
        )

        # ============================================================
        # SAVE DATA FOR PLOT
        # ============================================================

        self._update_plot_state(

            car_x,
            car_y,
            car_yaw,

            points_window,
            classes_window,

            graph_points,
            graph_classes,
            edges,

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

            raise RuntimeError(
                "Matplotlib is not installed.\n"
                "Install it using:\n"
                "python3 -m pip install matplotlib"
            )

        plt.ion()

        self.fig, (
            self.ax_global,
            self.ax_local
        ) = plt.subplots(
            1,
            2,
            figsize=(16, 8)
        )

        self.fig.canvas.manager.set_window_title(
            "SLAM Path Planner - Live Visualisation"
        )

        plt.show(
            block=False
        )

        self.get_logger().info(
            "Matplotlib plot initialised."
        )

    # ================================================================
    # PLOT HELPERS
    # ================================================================

    def _plot_points_by_class(
        self,
        ax,
        points,
        classes,
        local=False
    ):

        groups = {

            "blue": [],
            "yellow": [],
            "orange": [],
            "big": [],
            "virtual": [],
        }

        for point, cls in zip(
            points,
            classes
        ):

            if cls not in groups:
                continue

            groups[cls].append(
                point
            )

        # Blue

        if groups["blue"]:

            xs = [
                p[0]
                for p in groups["blue"]
            ]

            ys = [
                p[1]
                for p in groups["blue"]
            ]

            ax.scatter(
                xs,
                ys,
                marker="o",
                s=35,
                c="blue",
                label="Blue"
            )

        # Yellow

        if groups["yellow"]:

            xs = [
                p[0]
                for p in groups["yellow"]
            ]

            ys = [
                p[1]
                for p in groups["yellow"]
            ]

            ax.scatter(
                xs,
                ys,
                marker="o",
                s=35,
                c="gold",
                label="Yellow"
            )

        # Orange

        if groups["orange"]:

            xs = [
                p[0]
                for p in groups["orange"]
            ]

            ys = [
                p[1]
                for p in groups["orange"]
            ]

            ax.scatter(
                xs,
                ys,
                marker="o",
                s=40,
                c="orange",
                label="Orange"
            )

        # Big orange

        if groups["big"]:

            xs = [
                p[0]
                for p in groups["big"]
            ]

            ys = [
                p[1]
                for p in groups["big"]
            ]

            ax.scatter(
                xs,
                ys,
                marker="s",
                s=75,
                c="red",
                label="Big orange"
            )

        # Virtual

        if groups["virtual"]:

            xs = [
                p[0]
                for p in groups["virtual"]
            ]

            ys = [
                p[1]
                for p in groups["virtual"]
            ]

            ax.scatter(
                xs,
                ys,
                marker="D",
                s=75,
                c="magenta",
                label="Virtual"
            )

    # ================================================================
    # LOCAL TRANSFORMATION OF GRAPH
    # ================================================================

    def _transform_global_points_to_local(
        self,
        points,
        car_x,
        car_y,
        car_yaw
    ):

        result = []

        for px, py in points:

            lx, ly = \
                self._world_to_local(

                    car_x,
                    car_y,
                    car_yaw,

                    px,
                    py
                )

            result.append(
                (
                    lx,
                    ly
                )
            )

        return result

    # ================================================================
    # UPDATE PLOT
    # ================================================================

    def update_plot(self):

        if not _HAS_MATPLOTLIB:

            return

        # ------------------------------------------------------------
        # COPY PLOT STATE
        # ------------------------------------------------------------

        with self.plot_data_lock:

            d = {

                key:
                (
                    list(value)
                    if isinstance(
                        value,
                        list
                    )
                    else value
                )

                for key, value
                in self.plot_data.items()
            }

        car_x = d["car_x"]
        car_y = d["car_y"]
        car_yaw = d["car_yaw"]

        raw_points = d["raw_points"]
        raw_classes = d["raw_classes"]

        graph_points = d["graph_points"]
        graph_classes = d["graph_classes"]

        graph_edges = d["graph_edges"]

        virtual_cones = \
            d["virtual_cones"]

        candidate_points = \
            d["candidate_points"]

        path_points = \
            d["path_points"]

        dominant_colour = \
            d["dominant_colour"]

        mode = d["mode"]

        # ------------------------------------------------------------
        # CLEAR
        # ------------------------------------------------------------

        self.ax_global.clear()

        self.ax_local.clear()

        # ============================================================
        # GLOBAL FRAME
        # ============================================================

        ax = self.ax_global

        ax.set_title(
            "GLOBAL FRAME\n"
            +
            mode
        )

        ax.set_xlabel(
            "X [m]"
        )

        ax.set_ylabel(
            "Y [m]"
        )

        ax.grid(
            True
        )

        ax.set_aspect(
            "equal",
            adjustable="box"
        )

        # ------------------------------------------------------------
        # RAW CONES
        # ------------------------------------------------------------

        self._plot_points_by_class(

            ax,

            raw_points,

            raw_classes
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

                linestyle="--",
                linewidth=0.8,
                c="gray"
            )

        # ------------------------------------------------------------
        # VIRTUAL CONES
        # ------------------------------------------------------------

        if virtual_cones:

            xs = [
                p[0]
                for p in virtual_cones
            ]

            ys = [
                p[1]
                for p in virtual_cones
            ]

            ax.scatter(

                xs,
                ys,

                marker="D",
                s=80,
                c="magenta",
                label="Virtual cone"
            )

        # ------------------------------------------------------------
        # CANDIDATES
        # ------------------------------------------------------------

        if candidate_points:

            xs = [
                p[0]
                for p in candidate_points
            ]

            ys = [
                p[1]
                for p in candidate_points
            ]

            ax.scatter(

                xs,
                ys,

                marker="x",
                s=60,
                c="black",
                linewidths=1.5,
                label="Candidates"
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

                linewidth=2.5,
                c="limegreen",
                label="Greedy path"
            )

        # ------------------------------------------------------------
        # CAR
        # ------------------------------------------------------------

        ax.scatter(

            [car_x],
            [car_y],

            marker="*",
            s=160,
            c="black",
            label="Car"
        )

        arrow_len = 2.0

        ax.arrow(

            car_x,
            car_y,

            arrow_len
            *
            math.cos(
                car_yaw
            ),

            arrow_len
            *
            math.sin(
                car_yaw
            ),

            head_width=0.25,
            head_length=0.35,
            length_includes_head=True,
            color="black"
        )

        # ------------------------------------------------------------
        # LEGEND
        # ------------------------------------------------------------

        ax.legend(
            loc="upper right",
            fontsize=8
        )

        # ------------------------------------------------------------
        # DOMINANT COLOUR
        # ------------------------------------------------------------

        if dominant_colour:

            ax.text(

                0.02,
                0.02,

                f"Dominant colour: "
                f"{dominant_colour}",

                transform=ax.transAxes,

                fontsize=10,

                bbox=dict(
                    boxstyle="round",
                    facecolor="white",
                    alpha=0.8
                )
            )

        # ============================================================
        # LOCAL FRAME
        # ============================================================

        ax = self.ax_local

        ax.set_title(
            "LOCAL CAR FRAME"
        )

        ax.set_xlabel(
            "X forward [m]"
        )

        ax.set_ylabel(
            "Y left [m]"
        )

        ax.grid(
            True
        )

        ax.set_aspect(
            "equal",
            adjustable="box"
        )

        # ------------------------------------------------------------
        # RAW POINTS -> LOCAL
        # ------------------------------------------------------------

        local_raw = \
            self._transform_global_points_to_local(

                raw_points,

                car_x,
                car_y,

                car_yaw
            )

        self._plot_points_by_class(

            ax,

            local_raw,

            raw_classes
        )

        # ------------------------------------------------------------
        # GRAPH POINTS -> LOCAL
        # ------------------------------------------------------------

        local_graph = \
            self._transform_global_points_to_local(

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

                linestyle="--",
                linewidth=0.8,
                c="gray"
            )

        # ------------------------------------------------------------
        # VIRTUAL CONES LOCAL
        # ------------------------------------------------------------

        local_virtual = \
            self._transform_global_points_to_local(

                virtual_cones,

                car_x,
                car_y,

                car_yaw
            )

        if local_virtual:

            xs = [
                p[0]
                for p in local_virtual
            ]

            ys = [
                p[1]
                for p in local_virtual
            ]

            ax.scatter(

                xs,
                ys,

                marker="D",
                s=80,
                c="magenta",
                label="Virtual cone"
            )

        # ------------------------------------------------------------
        # CANDIDATES LOCAL
        # ------------------------------------------------------------

        local_candidates = \
            self._transform_global_points_to_local(

                candidate_points,

                car_x,
                car_y,

                car_yaw
            )

        if local_candidates:

            xs = [
                p[0]
                for p in local_candidates
            ]

            ys = [
                p[1]
                for p in local_candidates
            ]

            ax.scatter(

                xs,
                ys,

                marker="x",
                s=60,
                c="black",
                linewidths=1.5,
                label="Candidates"
            )

        # ------------------------------------------------------------
        # PATH LOCAL
        # ------------------------------------------------------------

        local_path = \
            self._transform_global_points_to_local(

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

                linewidth=2.5,
                c="limegreen",
                label="Greedy path"
            )

        # ------------------------------------------------------------
        # CAR
        # ------------------------------------------------------------

        ax.scatter(

            [0.0],
            [0.0],

            marker="*",
            s=160,
            c="black",
            label="Car"
        )

        # Car forward axis

        ax.arrow(

            0.0,
            0.0,

            2.0,
            0.0,

            head_width=0.25,
            head_length=0.35,
            length_includes_head=True,
            color="black"
        )

        # ------------------------------------------------------------
        # LEGEND
        # ------------------------------------------------------------

        ax.legend(
            loc="upper right",
            fontsize=8
        )

        # ------------------------------------------------------------
        # REFRESH
        # ------------------------------------------------------------

        self.fig.tight_layout()

        self.fig.canvas.draw_idle()

        self.fig.canvas.flush_events()

    # ================================================================
    # PLOT LOOP
    # ================================================================

    def plot_loop(self):

        if not _HAS_MATPLOTLIB:

            self.get_logger().error(
                "Cannot start plot loop because "
                "Matplotlib is not installed."
            )

            return

        self.init_plot()

        while rclpy.ok():

            # --------------------------------------------------------
            # WINDOW CLOSED
            # --------------------------------------------------------

            if not plt.fignum_exists(
                self.fig.number
            ):

                break

            # --------------------------------------------------------
            # UPDATE
            # --------------------------------------------------------

            self.update_plot()

            # --------------------------------------------------------
            # KEEP GUI RESPONSIVE
            # --------------------------------------------------------

            plt.pause(
                0.05
            )

        self.get_logger().info(
            "Plot window closed."
        )


# ================================================================
# MAIN
# ================================================================

def main():

    rclpy.init()

    node = \
        SlamPathSectorVisualizer()

    # ============================================================
    # ROS RUNS IN BACKGROUND THREAD
    #
    # This is important because Matplotlib GUI should run in the
    # main thread.
    # ============================================================

    ros_thread = threading.Thread(

        target=rclpy.spin,

        args=(node,),

        daemon=True
    )

    ros_thread.start()

    try:

        # ========================================================
        # MATPLOTLIB IN MAIN THREAD
        # ========================================================

        if _HAS_MATPLOTLIB:

            node.plot_loop()

        else:

            node.get_logger().error(
                "Matplotlib unavailable. "
                "Running ROS node without plot."
            )

            while rclpy.ok():

                threading.Event().wait(
                    1.0
                )

    except KeyboardInterrupt:

        pass

    finally:

        # ========================================================
        # STOP ROS
        # ========================================================

        if rclpy.ok():

            rclpy.shutdown()

        ros_thread.join(
            timeout=2.0
        )

        node.destroy_node()


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":

    main()