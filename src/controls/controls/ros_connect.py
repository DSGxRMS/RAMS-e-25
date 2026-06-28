# control_v2/ros_connect.py

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from nav_msgs.msg import Odometry, Path
from ackermann_msgs.msg import AckermannDriveStamped
from eufs_msgs.msg import ConeArrayWithCovariance   # /slam/map_cones (for one-sided recovery)
import math


class ROSInterface(Node):
    # POSITION/YAW source: /slam/odom_raw, NOT /slam/odom.
    # /slam/odom is published ONLY inside the SLAM cone callback (slam_node.py:534),
    # so it FREEZES whenever cones aren't being processed (e.g. car between cone
    # frames or past the visible cones) -> stale position -> CTE explodes -> the
    # controller drives the real car off the track. /slam/odom_raw is the
    # prediction_node GT relay (out.pose = ground_truth pose, full rate, always
    # live and — per the branch design — 100% accurate).
    def __init__(self, odom_topic="/slam/odom_raw", cmd_topic="/cmd", path_topic="/path_points",
                 gt_odom_topic="/ground_truth/odom", cones_topic="/slam/map_cones"):
        super().__init__('ros_interface')

        self.cx, self.cy, self.yaw, self.speed = 0.0, 0.0, 0.0, 0.0
        self.have_odom = False

        # Cones (for the one-sided-cone RECOVERY: blue-only -> steer right, yellow-only ->
        # steer left). /slam/map_cones is the global SLAM map, so we count only cones that
        # are NEAR + AHEAD of the car (approximates what's currently in view).
        self._cones_blue   = []   # list[(x,y)] map frame
        self._cones_yellow = []
        self.CONE_NEAR_R   = 12.0 # m: radius around car to consider "in view"

        # Ground-truth velocity (SLAM odom twist is always 0; GT carries real twist).
        # We take POSITION/yaw from /slam/odom (PP's frame) but SPEED from GT.
        self.gt_speed = 0.0
        self.have_gt  = False

        # Path buffering / gating
        self.latest_path = None               # last received Path message
        self._last_good_path = []            # last "usable" path as list[(x,y)]
        self._min_path_points = 5            # ignore tiny paths (fixes IndexError in controller)

        # ---- ROLLING-WINDOW accumulation (senior's design) -------------------
        # Maintain a PERSISTENT buffer of path points in the MAP frame. Each PP
        # frame ADDS its (deduped) points; points the car has driven PAST are
        # discarded. A momentary one-sided/bad PP frame then only adds a couple
        # junk points to a mostly-good buffer instead of replacing the whole path
        # -> no hooks, continuous track.
        self._accum = []                     # list[(x,y)] accumulated in map frame
        self.DEDUP_R   = 0.4                 # m: don't add a point within this of an existing one
        self.BEHIND_M  = 1.0                 # m: drop points more than this BEHIND the car
        self.AHEAD_MAX = 25.0                # m: cap how far ahead we keep (bounded buffer)

        # Best Effort QoS (typical for high-rate sensor-ish streams)
        self.qos_best_effort = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        self.create_subscription(Odometry, odom_topic, self._odom_cb, self.qos_best_effort)
        self.create_subscription(Odometry, gt_odom_topic, self._gt_odom_cb, self.qos_best_effort)
        self.create_subscription(Path, path_topic, self._path_cb, self.qos_best_effort)
        self.create_subscription(ConeArrayWithCovariance, cones_topic, self._cones_cb, self.qos_best_effort)

        self.pub = self.create_publisher(AckermannDriveStamped, cmd_topic, 10)

    def _odom_cb(self, msg: Odometry):
        self.cx = float(msg.pose.pose.position.x)
        self.cy = float(msg.pose.pose.position.y)

        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

        # SLAM twist is 0 — keep for fallback only.
        vx, vy = float(msg.twist.twist.linear.x), float(msg.twist.twist.linear.y)
        self.speed = math.hypot(vx, vy)

        self.have_odom = True

    def _gt_odom_cb(self, msg: Odometry):
        # Ground-truth carries the REAL vehicle velocity in its twist.
        # Use FORWARD velocity (body x) only. hypot(vx,vy) would inflate the
        # reading when the car slides sideways, poisoning the planner.
        self.gt_speed = float(msg.twist.twist.linear.x)
        self.have_gt = True

    def _cones_cb(self, msg: ConeArrayWithCovariance):
        self._cones_blue   = [(float(c.point.x), float(c.point.y)) for c in msg.blue_cones]
        self._cones_yellow = [(float(c.point.x), float(c.point.y)) for c in msg.yellow_cones]

    def get_local_cone_counts(self):
        """Count blue/yellow cones that are NEAR and roughly AHEAD of the car
        (approximates what's currently in view, from the global SLAM map).
        Returns (n_blue, n_yellow). Used for one-sided-cone recovery."""
        cx, cy, yaw = self.cx, self.cy, self.yaw
        fx, fy = math.cos(yaw), math.sin(yaw)
        r2 = self.CONE_NEAR_R * self.CONE_NEAR_R

        def count(cones):
            n = 0
            for (px, py) in cones:
                dx, dy = px - cx, py - cy
                if dx * dx + dy * dy > r2:
                    continue
                if dx * fx + dy * fy < -1.0:      # behind the car -> ignore
                    continue
                n += 1
            return n

        return count(self._cones_blue), count(self._cones_yellow)

    def _path_cb(self, msg: Path):
        # Always store the latest message, but only promote to "usable" if it has enough points.
        self.latest_path = msg

        pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        if len(pts) >= self._min_path_points:
            self._last_good_path = pts

        # ---- ROLLING-WINDOW accumulation: ADD new points (deduped) ----
        # Skip the very first PP point (it's the car's own pose, prepended by PP).
        for (px, py) in pts[1:]:
            dup = False
            for (ax, ay) in self._accum:
                if (px - ax) ** 2 + (py - ay) ** 2 < self.DEDUP_R ** 2:
                    dup = True
                    break
            if not dup:
                self._accum.append((px, py))

    def _prune_accum(self):
        """Drop accumulated points the car has driven PAST (behind it) or that are
        absurdly far ahead. Keeps the buffer to the live forward track."""
        if not self._accum:
            return
        cx, cy, cyaw = self.cx, self.cy, self.yaw
        fx, fy = math.cos(cyaw), math.sin(cyaw)        # car forward unit vector
        kept = []
        for (px, py) in self._accum:
            dx, dy = px - cx, py - cy
            along = dx * fx + dy * fy                  # signed distance along heading
            dist  = math.hypot(dx, dy)
            if along < -self.BEHIND_M:                 # behind the car -> discard
                continue
            if dist > self.AHEAD_MAX:                  # too far ahead -> discard
                continue
            kept.append((px, py))
        self._accum = kept

    def get_state(self):
        """Returns (x, y, yaw, speed, have_odom).

        Position/yaw from /slam/odom; speed from /ground_truth/odom (SLAM twist is 0).
        Falls back to the SLAM twist speed if GT has not arrived yet.
        """
        spd = self.gt_speed if self.have_gt else self.speed
        return (self.cx, self.cy, self.yaw, spd, self.have_odom)

    def get_path(self):
        """
        Return the accumulated rolling-window path AHEAD of the car (map frame),
        ordered by arc-length from the car. Points the car has passed are pruned.
        Falls back to the last good raw PP frame if the buffer is too thin.
        """
        self._prune_accum()
        # order the kept points by distance projected along the car heading so the
        # controller gets a monotonically-forward path (not jumbled by insertion order)
        cx, cy, cyaw = self.cx, self.cy, self.yaw
        fx, fy = math.cos(cyaw), math.sin(cyaw)
        scored = sorted(self._accum, key=lambda p: (p[0] - cx) * fx + (p[1] - cy) * fy)
        if len(scored) >= self._min_path_points:
            return scored
        # not enough accumulated yet -> use the latest good raw PP frame
        if self._last_good_path:
            return self._last_good_path
        return []

    def send_command(self, steering, speed=None, accel=None):
        # Removed per-frame logger spam (was slowing sim / lowering RTF)
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = float(steering)
        if accel is not None:
            msg.drive.acceleration = float(accel)
        if speed is not None:
            msg.drive.speed = float(speed)

        self.pub.publish(msg)
