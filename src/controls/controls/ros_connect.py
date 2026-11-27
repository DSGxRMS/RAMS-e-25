# control_v2/ros_connect.py

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from nav_msgs.msg import Odometry, Path
from ackermann_msgs.msg import AckermannDriveStamped
from eufs_msgs.msg import ConeArrayWithCovariance, WheelSpeedsStamped
from controls.control_utils import local_closest_index, cross_track_error
from controls.path_reference import OrderedReference
from controls.cone_clearance import SMALL_CONE_RADIUS_M, LARGE_CONE_RADIUS_M
import math
from threading import Lock


class ROSInterface(Node):
    # POSITION/YAW source: /slam/odom_raw, NOT /slam/odom.
    # /slam/odom is published ONLY inside the SLAM cone callback (slam_node.py:534),
    # so it FREEZES whenever cones aren't being processed (e.g. car between cone
    # frames or past the visible cones) -> stale position -> CTE explodes -> the
    # controller drives the real car off the track. /slam/odom_raw is the
    # prediction_node GT relay (out.pose = ground_truth pose, full rate, always
    # live and — per the branch design — 100% accurate).
    def __init__(self, odom_topic="/slam/odom_raw", cmd_topic="/cmd", path_topic="/path_points",
                 gt_odom_topic="/ground_truth/odom", cones_topic="/slam/map_cones",
                 ordered_reference=False, wheel_feedback=False, clearance_feedback=False):
        super().__init__('ros_interface')

        self.cx, self.cy, self.yaw, self.speed = 0.0, 0.0, 0.0, 0.0
        self.have_odom = False
        self._clearance_feedback = clearance_feedback
        self._clearance_snapshot = None
        self._pose_snapshot = (0.0, 0.0, 0.0, None)

        # Local cone counts provide evidence for the heading guard and legacy steering.
        self._cones_blue   = []                  # list[(x,y)] map frame
        self._cones_yellow = []
        self.FOV_VERTEX = -5.0                   # m: FOV vertex 5 m behind car along heading
        self.FOV_RADIUS = 30.0                   # m: sector radius from the vertex
        self.FOV_HALF   = math.radians(45.0)     # half-angle (PP uses fov.angle_deg = 90)

        # --- P16: FRAME-SPLIT GUARD ----------------------------------------
        # Pose comes from /slam/odom_raw (GT) but the cone map comes from
        # /slam/map_cones, which lives in the PF+snap-back CORRECTED frame
        # (the same frame as /slam/odom). The two agree only while the SLAM
        # correction is ~identity. When snap-back fires big (measured 2026-08-25:
        # ~5 m shift mid-run), the reference may be in the wrong place. Watch
        # /slam/odom and invalidate the reference if it diverges from the GT pose.
        self.slam_x = self.slam_y = self.slam_yaw = 0.0
        self._slam_stamp = None                          # sim time of last /slam/odom
        self.FRAME_SPLIT_M = 1.0                         # m: GT vs SLAM divergence limit
        self.FRAME_STALE_S = 0.5                         # s: /slam/odom silence -> no verdict
        # (the /slam/odom subscription itself is created further down, after
        #  self.qos_best_effort is defined)

        # Ground-truth velocity (SLAM odom twist is always 0; GT carries real twist).
        # We take POSITION/yaw from /slam/odom (PP's frame) but SPEED from GT.
        self.gt_speed = 0.0
        self.have_gt  = False
        self.gt_motion_sample = None         # Stamped planar speed for shutdown confirmation.

        # --- P1: GT-speed staleness guard ---------------------------------
        # have_gt latches True forever, so if /ground_truth/odom dies the old
        # code kept trusting gt_speed frozen at its last value, silently.
        # Every GT message is now stamped with SIM time and gated by age:
        #   age <= GT_STALE_WARN_S : fresh -> use gt_speed normally
        #   WARN < age <= ZERO_S   : stale -> warn ONCE, hold last value briefly
        #   age >  GT_STALE_ZERO_S : stop trusting the frozen reading -> 0.0 m/s
        # Reading speed LOW makes the PI decelerate = safe-direction failure.
        # Age uses the SIM clock on purpose: if Gazebo fully pauses, sim time
        # freezes with it, so nothing false-triggers during a pause.
        self._gt_stamp       = None   # sim-time of last /ground_truth/odom msg
        self._gt_warned      = False  # already warned about the current episode
        self._gt_ever_stale  = False  # a stale episode happened (restore log)
        self.GT_STALE_WARN_S = 0.3    # ~57 missed msgs @190 Hz before warning
        self.GT_STALE_ZERO_S = 0.5    # <=2.5 m travelled on a bad reading @5 m/s

        # Path buffering / gating
        self.latest_path = None               # last received Path message
        self.reference = OrderedReference() if ordered_reference else None
        self._last_good_path = []            # last "usable" path as list[(x,y)]
        self._min_path_points = 5            # ignore tiny paths (fixes IndexError in controller)

        # ---- ROLLING-WINDOW accumulation (senior's design) -------------------
        # Maintain a PERSISTENT buffer of path points in the MAP frame. Each PP
        # frame ADDS its (deduped) points; points the car has driven PAST are
        # discarded. A momentary one-sided/bad PP frame then only adds a couple
        # junk points to a mostly-good buffer instead of replacing the whole path
        # -> no hooks, continuous track.
        self._accum = []                     # entries: (x, y)
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
        # P16 frame-split guard: PF pose in the cone-map's frame (BEST_EFFORT, same as everything else)
        self.create_subscription(Odometry, "/slam/odom", self._slam_odom_cb, self.qos_best_effort)
        self.applied_steering = None
        self._wheel_lock = Lock()
        self._wheel_samples = ()
        if wheel_feedback:
            self.create_subscription(WheelSpeedsStamped, "/ground_truth/wheel_speeds",
                                     self._wheel_cb, self.qos_best_effort)

        self.pub = self.create_publisher(AckermannDriveStamped, cmd_topic, 10)

    def _wheel_cb(self, msg):
        angle = float(msg.speeds.steering)
        stamp = msg.header.stamp
        stamp_s = stamp.sec + stamp.nanosec * 1e-9
        if stamp_s > 0 and math.isfinite(angle):
            sample = stamp_s, angle
            with self._wheel_lock:
                self.applied_steering = sample
                self._wheel_samples = (*self._wheel_samples[-63:], sample)

    def steering_samples(self):
        with self._wheel_lock:
            return self._wheel_samples

    def reset_steering_samples(self):
        with self._wheel_lock:
            self._wheel_samples = ()
            self.applied_steering = None

    def _odom_cb(self, msg: Odometry):
        self.cx = float(msg.pose.pose.position.x)
        self.cy = float(msg.pose.pose.position.y)

        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)
        stamp = msg.header.stamp
        self._pose_snapshot = self.cx, self.cy, self.yaw, stamp.sec + stamp.nanosec * 1e-9

        # SLAM twist is 0 — keep for fallback only.
        vx, vy = float(msg.twist.twist.linear.x), float(msg.twist.twist.linear.y)
        self.speed = math.hypot(vx, vy)

        self.have_odom = True

    def _gt_odom_cb(self, msg: Odometry):
        # Ground-truth carries the REAL vehicle velocity in its twist.
        # Use FORWARD velocity (body x) only. hypot(vx,vy) would inflate the
        # reading when the car slides sideways, poisoning the planner.
        linear = msg.twist.twist.linear
        self.gt_speed = float(linear.x)
        stamp = msg.header.stamp
        self.gt_motion_sample = (stamp.sec * 1_000_000_000 + stamp.nanosec,
                                 math.hypot(linear.x, linear.y))
        self.have_gt = True

        # P1: refresh the freshness stamp; if this message ends a stale
        # episode, say so exactly once and re-arm the warning.
        self._gt_stamp = self.get_clock().now()
        if self._gt_ever_stale:
            self._gt_ever_stale = False
            self._gt_warned = False
            print("[ros_connect] GT speed RESTORED (fresh /ground_truth/odom)",
                  flush=True)

    def _cones_cb(self, msg: ConeArrayWithCovariance):
        self._cones_blue   = [(float(c.point.x), float(c.point.y)) for c in msg.blue_cones]
        self._cones_yellow = [(float(c.point.x), float(c.point.y)) for c in msg.yellow_cones]
        if self._clearance_feedback and msg.header.frame_id == "map":
            stamp = msg.header.stamp
            cone_groups = ((msg.blue_cones, SMALL_CONE_RADIUS_M),
                           (msg.yellow_cones, SMALL_CONE_RADIUS_M),
                           (msg.orange_cones, SMALL_CONE_RADIUS_M),
                           (msg.big_orange_cones, LARGE_CONE_RADIUS_M),
                           (getattr(msg, "unknown_color_cones", ()), LARGE_CONE_RADIUS_M))
            points = tuple((c.point.x, c.point.y, radius) for group, radius in cone_groups
                           for c in group if math.isfinite(c.point.x) and math.isfinite(c.point.y))
            stamp_s = stamp.sec + stamp.nanosec * 1e-9
            if self._clearance_snapshot is None or stamp_s >= self._clearance_snapshot[0]:
                self._clearance_snapshot = (stamp_s, points)

    def clearance_cones(self, now):
        snapshot = self._clearance_snapshot
        if snapshot is not None and -.06 <= now - snapshot[0] <= .25:
            return snapshot[1] or None
        return None

    def _slam_odom_cb(self, msg: Odometry):
        """P16: PF pose in the CORRECTED frame -- the same frame the cone map
        lives in. Compared against the GT pose to detect a snap-back frame
        split."""
        self.slam_x = float(msg.pose.pose.position.x)
        self.slam_y = float(msg.pose.pose.position.y)
        self._slam_stamp = self.get_clock().now()

    def frame_ok(self):
        """P16: True while the cone map's frame is aligned with the GT pose we
        plan in. Returns True (cannot judge) when /slam/odom has never arrived
        or went silent -- a dead SLAM is a different failure, not a split."""
        if self._slam_stamp is None:
            return True
        age = (self.get_clock().now() - self._slam_stamp).nanoseconds * 1e-9
        if age > self.FRAME_STALE_S:
            return True
        d = math.hypot(self.slam_x - self.cx, self.slam_y - self.cy)
        return d <= self.FRAME_SPLIT_M

    def frame_split_distance(self):
        """P16: current |GT pose - SLAM pose| (for diagnostics)."""
        return math.hypot(self.slam_x - self.cx, self.slam_y - self.cy)

    def invalidate_buffer(self):
        """P16: the accumulated points (and the raw fallback) are in a frame
        that no longer matches the pose -- drop them all."""
        self._accum.clear()
        self._last_good_path = []
        self._clearance_snapshot = None
        if self.reference is not None:
            self.reference.clear()

    def get_fov_cones(self):
        """Blue/yellow cone POSITIONS (map frame) inside the planning FOV wedge
        (vertex FOV_VERTEX m behind the car, radius FOV_RADIUS m, +/-FOV_HALF
        around heading) — the same wedge the PP node plans on. Returns
        (blue_pts, yellow_pts) as lists[(x, y)]."""
        cx, cy, yaw = self.cx, self.cy, self.yaw
        cs, sn = math.cos(yaw), math.sin(yaw)
        R2 = self.FOV_RADIUS * self.FOV_RADIUS

        def filt(cones):
            out = []
            for (px, py) in cones:
                dx, dy = px - cx, py - cy
                lx =  cs * dx + sn * dy          # forward (car frame)
                ly = -sn * dx + cs * dy          # left
                vx = lx - self.FOV_VERTEX        # relative to the FOV vertex (behind car)
                vy = ly
                rr = vx * vx + vy * vy
                if rr < 1e-6 or rr > R2:          # outside the sector radius
                    continue
                if abs(math.atan2(vy, vx)) > self.FOV_HALF:   # outside the wedge
                    continue
                out.append((px, py))
            return out

        return filt(self._cones_blue), filt(self._cones_yellow)

    def get_local_cone_counts(self):
        """Count blue/yellow cones inside the planning FOV wedge. Returns
        (n_blue, n_yellow)."""
        blue, yellow = self.get_fov_cones()
        return len(blue), len(yellow)

    def _path_cb(self, msg: Path):
        # Always store the latest message, but only promote to "usable" if it has enough points.
        self.latest_path = msg

        pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        if self.reference is not None:
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.reference.accept_planner(pts, msg.header.frame_id, stamp, self._min_path_points)
            return
        if len(pts) >= self._min_path_points:
            self._last_good_path = pts

        # ---- ROLLING-WINDOW accumulation: ADD new points (deduped) ----
        # Skip the very first PP point (it's the car's own pose, prepended by PP).
        for px, py in pts[1:]:
            if not any((px - bx) ** 2 + (py - by) ** 2 < self.DEDUP_R ** 2
                       for bx, by in self._accum):
                self._accum.append((px, py))

    def _prune_accum(self):
        """Drop accumulated points behind the car or too far from it."""
        cx, cy, cyaw = self.cx, self.cy, self.yaw
        fx, fy = math.cos(cyaw), math.sin(cyaw)
        kept = []
        for px, py in self._accum:
            dx, dy = px - cx, py - cy
            if math.hypot(dx, dy) <= self.AHEAD_MAX and dx * fx + dy * fy >= -self.BEHIND_M:
                kept.append((px, py))
        self._accum = kept

    def get_state(self, *, stamped=False):
        """Returns (x, y, yaw, speed, have_odom).

        Position/yaw from /slam/odom_raw; speed from /ground_truth/odom (SLAM
        twist is 0). P1: the GT speed is freshness-gated (GT_STALE_WARN_S /
        GT_STALE_ZERO_S). Falls back to the SLAM twist speed when GT has never
        arrived, and forces 0.0 m/s once GT has been silent too long.
        """
        spd = self.speed                       # SLAM twist fallback (= 0 today)
        if self.have_gt and self._gt_stamp is not None:
            age = (self.get_clock().now() - self._gt_stamp).nanoseconds * 1e-9
            if age <= self.GT_STALE_ZERO_S:
                spd = self.gt_speed            # fresh, or inside hold window
                if age > self.GT_STALE_WARN_S and not self._gt_warned:
                    self._gt_warned = True
                    self._gt_ever_stale = True
                    print(f"[ros_connect] WARNING: GT speed stale {age:.2f}s "
                          f"(holding last value {self.gt_speed:.2f} m/s; "
                          f"forcing 0.0 after {self.GT_STALE_ZERO_S:.1f}s)",
                          flush=True)
            else:
                # Silent too long -> the frozen reading is a lie. Report 0 so
                # the PI decelerates instead of planning from stale motion.
                spd = 0.0
                self._gt_ever_stale = True
        x, y, yaw, pose_stamp = self._pose_snapshot
        state = x, y, yaw, spd, self.have_odom
        return (*state, pose_stamp) if stamped else state

    def get_path(self):
        """
        Return the accumulated rolling-window path AHEAD of the car (map frame),
        ordered ALONG THE TRACK by greedy nearest-neighbour chaining from the car.
        Points the car has passed are pruned. Falls back to the last good raw PP
        frame if the buffer is too thin.
        """
        self._prune_accum()
        pts = self._accum
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

    def crosstrack_to_last_good(self, cx, cy, yaw):
        """Signed crosstrack + heading error from the car to the LAST-GOOD path.
        No longer used by control_node (REC removed in P14) but kept for the
        GRU twin (control_node_gru.py). Returns (e, hd_off, ok)."""
        path = self._last_good_path
        if not path or len(path) < 2:
            return 0.0, 0.0, False
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        idx = local_closest_index((cx, cy), xs, ys, 0, loop=False)
        idx = max(0, min(idx, len(xs) - 1))
        e, theta_ref = cross_track_error(cx, cy, xs, ys, idx, loop=False)
        hd_off = abs(math.atan2(math.sin(yaw - theta_ref), math.cos(yaw - theta_ref)))
        return e, hd_off, True

    def send_command(self, steering, speed=None, accel=None):
        # Removed per-frame logger spam (was slowing sim / lowering RTF)
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = float(steering)
        if accel is not None:
            msg.drive.acceleration = float(accel)
        if speed is not None:
            msg.drive.speed = float(speed)

        published_at = self.get_clock().now()
        msg.header.stamp = published_at.to_msg()
        self.pub.publish(msg)
        return published_at.nanoseconds
