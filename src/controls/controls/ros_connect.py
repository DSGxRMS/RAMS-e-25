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

        # Cones (for the one-sided-cone RECOVERY: blue dominates -> steer right, yellow
        # dominates -> steer left). /slam/map_cones is the global SLAM map; we count only
        # cones inside the SAME sector FOV the PP node plans on, so REC's "which wall do we
        # see" matches what PP actually built its (bad) path from. Matches pp_node_skidpad.py.
        self._cones_blue   = []                  # list[(x,y)] map frame
        self._cones_yellow = []
        self.FOV_VERTEX = -5.0                   # m: FOV vertex 5 m behind car along heading
        self.FOV_RADIUS = 30.0                   # m: sector radius from the vertex
        self.FOV_HALF   = math.radians(45.0)     # half-angle (PP uses fov.angle_deg = 90)

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
        self.MAX_GAP_M = 3.5                 # m: when chaining points into a path, a jump bigger
                                             #    than this ends the path (stray cone / other lane)

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
        """Count blue/yellow cones inside the SAME sector FOV the PP node plans on
        (vertex FOV_VERTEX m behind the car, radius FOV_RADIUS m, +/-FOV_HALF around
        heading). So the recovery's 'which wall is visible' matches what PP actually
        saw when it built the (bad) path. Returns (n_blue, n_yellow)."""
        cx, cy, yaw = self.cx, self.cy, self.yaw
        cs, sn = math.cos(yaw), math.sin(yaw)
        R2 = self.FOV_RADIUS * self.FOV_RADIUS

        def count(cones):
            n = 0
            for (px, py) in cones:
                dx, dy = px - cx, py - cy
                lx =  cs * dx + sn * dy          # forward (car frame)
                ly = -sn * dx + cs * dy          # left
                vx = lx - self.FOV_VERTEX        # relative to the FOV vertex (behind car)
                vy = ly
                rr = vx * vx + vy * vy
                if rr < 1e-6 or rr > R2:          # outside the sector radius
                    continue
                if abs(math.atan2(vy, vx)) > self.FOV_HALF:   # outside the +/-45 deg wedge
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
        ordered ALONG THE TRACK by greedy nearest-neighbour chaining from the car.
        Points the car has passed are pruned. Falls back to the last good raw PP
        frame if the buffer is too thin.
        """
        self._prune_accum()
        pts = list(self._accum)
        if len(pts) >= self._min_path_points:
            # Reconstruct the 1-D track order from the 2-D point cloud by greedy
            # nearest-neighbour starting at the car. The OLD code sorted by projection
            # onto the car heading, which ZIG-ZAGGED on curves (a point far along a bend
            # projects small and interleaves with near points) -> 70-500 m garbage
            # splines no controller can follow. We stop the chain at a big gap so a
            # stray cone / the other lane can't make it jump back across the track.
            cx, cy = self.cx, self.cy
            remaining = pts[:]
            cur = min(remaining, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
            remaining.remove(cur)
            ordered = [cur]
            while remaining:
                lx, ly = ordered[-1]
                nxt = min(remaining, key=lambda p: (p[0] - lx) ** 2 + (p[1] - ly) ** 2)
                if math.hypot(nxt[0] - lx, nxt[1] - ly) > self.MAX_GAP_M:
                    break                       # next point too far -> path ends here
                ordered.append(nxt)
                remaining.remove(nxt)
            if len(ordered) >= self._min_path_points:
                return ordered
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
