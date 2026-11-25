# control_v2/ros_connect.py

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from nav_msgs.msg import Odometry, Path
from ackermann_msgs.msg import AckermannDriveStamped
from eufs_msgs.msg import ConeArrayWithCovariance, WheelSpeedsStamped
from controls.control_utils import local_closest_index, cross_track_error
from controls.path_reference import OrderedReference
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
                 gt_odom_topic="/ground_truth/odom", cones_topic="/slam/map_cones",
                 ordered_reference=False, wheel_feedback=False):
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

        # --- P14: one-wall pseudo centreline constants ----------------------
        # The ONLY physical assumption: the track centreline sits HALF_TRACK_WIDTH
        # from the visible wall, on the CAR'S side of it (the car drives on the
        # track). Colour is used only to pick the better wall and as tiebreak.
        self.PSEUDO_MIN_WALL     = 3                    # cones needed to trust a wall fit
        self.PSEUDO_MAX_FIT      = 8                    # fit the NEAREST N cones (walls curve)
        self.PSEUDO_HALF_WIDTH_M = 1.5                  # wall -> centreline offset (m)
        self.PSEUDO_MAX_LEN_M    = 8.0                  # bridge length cap (a straight fit
                                                        #   lies in corners — bridge, not map)
        self.PSEUDO_EXTRAP_M     = 2.0                  # extend past the last cone (m)
        self.PSEUDO_STEP_M       = 2.5                  # spacing of generated points (m)
        self.PSEUDO_MAX_SKEW_RAD = math.radians(75.0)   # wall must run roughly ahead
        self.PSEUDO_MIN_AHEAD_M  = 0.0                  # allow cones level with the bumper
                                                        #   (corner walls run BESIDE the nose)
        self.PSEUDO_SHORT_SKEW_RAD = math.radians(60.0) # above this skew the straight fit
                                                        #   is corner-exit guesswork -> the
                                                        #   bridge is capped short
        self.PSEUDO_SHORT_LEN_M  = 4.0                  # short-bridge cap (m)
        # Pseudo points are a BRIDGE, not map: pruned from the rolling buffer the
        # moment the car passes them, drifts >3 m sideways from them, or after
        # 8 s. Without this, stale beside-points hijack the greedy chain and
        # drag the CTE (measured: -11 -> -14 -> -25 m runaway on 2026-08-25).
        # TTL is generous because the passed/sideways rules do the real work:
        # a slow creep must not have its bridge evaporate underneath it.
        self.PSEUDO_PASSED_M = 0.2                      # m: drop once the car passes them
        self.PSEUDO_LAT_MAX_M = 3.0                     # m: drop when > this to the side
        self.PSEUDO_TTL_S = 8.0                         # s: hard expiry
        self.PSEUDO_GRACE_S = 2.0                       # s: fresh bridges are exempt from
                                                        #   the passed/lateral rules. At a
                                                        #   ~90 deg corner the bridge runs
                                                        #   PERPENDICULAR to the nose (that's
                                                        #   what cornering is) -- measuring
                                                        #   staleness against the heading
                                                        #   pruned every fresh bridge within
                                                        #   one cycle (measured 2026-08-25:
                                                        #   419 feeds, +4 pts each, path
                                                        #   never grew, car froze at
                                                        #   (32.7, 3.4) for 50 s).
        self._last_pseudo_feed_t = -1e9                 # bridge_fresh() timestamp
        self.last_pseudo_reject  = "never attempted"    # reason for the last empty return

        # --- P18-b: PP-SILENCE UNLOCK ---------------------------------------
        # The one_sided dominance gate blocks the bridge exactly where BOTH
        # walls stay visible yet PP delivers nothing usable (measured 2026-08-25
        # lap-2 descent: fov 4-5 blue vs 3-4 yellow for the whole section, feeds
        # locked out, buffer eaten to cur=38/40, permanent STOP_PATH_M park).
        # When no REAL PP point has been accepted for PSEUDO_UNLOCK_SILENT_S and
        # the buffer is starving, control_node calls pseudo_wall_path with
        # allow_both_walls=True: the majority rule is dropped (larger wall fits)
        # and a strict straightness gate is added instead.
        self.PSEUDO_UNLOCK_SILENT_S = 3.0    # s without newly accepted real pts
        self.PSEUDO_FIT_DEV_MAX_M = 0.35     # m: max cone-to-line residual in
                                             #   unlocked mode (a bent or mixed
                                             #   "wall" must not hallucinate a
                                             #   centreline across the corridor)
        self._last_real_pt_t = None          # sim time of last real PP point accepted

        # --- P16: FRAME-SPLIT GUARD ----------------------------------------
        # Pose comes from /slam/odom_raw (GT) but the cone map comes from
        # /slam/map_cones, which lives in the PF+snap-back CORRECTED frame
        # (the same frame as /slam/odom). The two agree only while the SLAM
        # correction is ~identity. When snap-back fires big (measured 2026-08-25:
        # ~5 m shift mid-run), controls plans against displaced cones -> feeds
        # nonsense bridges, the buffer prunes empty, the car coasts to a stop
        # 'for no reason' and can drift off track. Guard: watch /slam/odom and
        # if it diverges from the GT pose the whole map is poison -> invalidate.
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
        self._accum = []                     # entries: (x, y, stamp_s, is_pseudo)
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
        if wheel_feedback:
            self.create_subscription(WheelSpeedsStamped, "/ground_truth/wheel_speeds",
                                     self._wheel_cb, self.qos_best_effort)

        self.pub = self.create_publisher(AckermannDriveStamped, cmd_topic, 10)

    def _wheel_cb(self, msg):
        angle = float(msg.speeds.steering)
        stamp = msg.header.stamp
        if math.isfinite(angle):
            self.applied_steering = stamp.sec + stamp.nanosec * 1e-9, angle

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

    def pseudo_wall_path(self, allow_both_walls=False):
        """P14: ONE-WALL PSEUDO CENTRELINE.

        When one wall dominates the FOV, PP publishes nothing usable and the
        rolling buffer runs dry. The centreline is still computable: fit a
        line to the visible wall and offset it half a track width toward the
        CAR'S SIDE of the wall (the car drives on the track, so the centre is
        between the car and the wall, never away from it). Returns
        list[(x, y)] map-frame points (possibly empty); on an empty return,
        self.last_pseudo_reject says why.

        P18-b: with allow_both_walls=True (PP-silence unlock, set by
        control_node when no real point has been accepted for a while), the
        majority rule is dropped -- the LARGER colour fits even when both
        walls are visible -- and a strict straightness gate applies instead
        (every fitted cone must sit within PSEUDO_FIT_DEV_MAX_M of the PCA
        line, so a bent or mixed "wall" cannot hallucinate a line across the
        corridor).

        Wall selection: normally the DOMINANT colour (>= PSEUDO_MIN_WALL cones
        and strictly more than the other). A couple of opposing cones near
        the car do NOT block the fit — the fit uses only this wall's cones.

        Guards (refuse to hallucinate):
          - total-least-squares line (PCA) on the NEAREST PSEUDO_MAX_FIT cones
            only (walls curve; a long straight fit lies)
          - wall direction must run roughly ahead (< PSEUDO_MAX_SKEW_RAD off
            the nose) — a wall across the nose is corner entry, not a wall
          - generated points capped at PSEUDO_MAX_LEN_M: a BRIDGE (~2-3 s of
            driving) so the real pipeline takes over, not a map
          - unlocked mode only: max cone-to-line residual gate (above)
        """
        blue, yellow = self.get_fov_cones()
        nb, ny = len(blue), len(yellow)
        if allow_both_walls:
            # P18-b UNLOCKED MODE: bigger wall wins, majority not required.
            if nb >= self.PSEUDO_MIN_WALL and nb >= ny:
                wall, side = blue, -1.0
            elif ny >= self.PSEUDO_MIN_WALL:
                wall, side = yellow, +1.0
            else:
                self.last_pseudo_reject = (f"unlock: too few wall cones "
                                           f"(nb={nb} ny={ny}, need >="
                                           f"{self.PSEUDO_MIN_WALL})")
                return []
        elif nb >= self.PSEUDO_MIN_WALL and nb > ny:
            wall, side = blue, -1.0            # tiebreak only: blue LEFT -> centre right
        elif ny >= self.PSEUDO_MIN_WALL and ny > nb:
            wall, side = yellow, +1.0          # tiebreak only: yellow RIGHT -> centre left
        else:
            self.last_pseudo_reject = (f"no dominant wall (nb={nb} ny={ny}, "
                                       f"need >={self.PSEUDO_MIN_WALL} and majority)")
            return []

        cx, cy, yaw = self.cx, self.cy, self.yaw
        cs, sn = math.cos(yaw), math.sin(yaw)

        # to car frame (fwd, left), keep cones ahead of the bumper, nearest first
        loc = []
        for (px, py) in wall:
            dx, dy = px - cx, py - cy
            loc.append((cs * dx + sn * dy, -sn * dx + cs * dy))
        loc = sorted((p for p in loc if p[0] > self.PSEUDO_MIN_AHEAD_M),
                     key=lambda p: p[0])
        if len(loc) < self.PSEUDO_MIN_WALL:
            self.last_pseudo_reject = (f"only {len(loc)} wall cones ahead of "
                                       f"the bumper (need >={self.PSEUDO_MIN_WALL})")
            return []
        fit = loc[:self.PSEUDO_MAX_FIT]

        # Total-least-squares line via the closed-form 2x2 PCA direction.
        # NOT y = a*x + b: near corners the wall runs sideways and OLS degenerates.
        n = len(fit)
        mf = sum(p[0] for p in fit) / n
        ml = sum(p[1] for p in fit) / n
        sff = sum((p[0] - mf) ** 2 for p in fit)
        sll = sum((p[1] - ml) ** 2 for p in fit)
        sfl = sum((p[0] - mf) * (p[1] - ml) for p in fit)
        theta = 0.5 * math.atan2(2.0 * sfl, sff - sll)
        d = (math.cos(theta), math.sin(theta))       # principal dir (fwd, left)
        if d[0] < 0.0:                               # point the line FORWARD
            d = (-d[0], -d[1])
        skew = math.atan2(abs(d[1]), d[0])
        if skew > self.PSEUDO_MAX_SKEW_RAD:
            self.last_pseudo_reject = (f"wall skew {math.degrees(skew):.0f} deg "
                                       f"> {math.degrees(self.PSEUDO_MAX_SKEW_RAD):.0f} limit")
            return []

        if allow_both_walls:
            # P18-b straightness gate (unlocked mode only): every fitted cone
            # must sit close to the PCA line, else the "wall" is bending or
            # mixing colours -- exactly the geometry where an offset line can
            # cut across the corridor.
            dev = max(abs((p[0] - mf) * d[1] - (p[1] - ml) * d[0]) for p in fit)
            if dev > self.PSEUDO_FIT_DEV_MAX_M:
                self.last_pseudo_reject = (
                    f"unlock: wall fit bends {dev:.2f} m "
                    f"> {self.PSEUDO_FIT_DEV_MAX_M:.2f} limit")
                return []

        # Offset side: the CENTRELINE IS ON THE CAR'S SIDE of the visible wall
        # (the car drives on the track — the centre is between the car and the
        # wall, never away from it). Colour convention is only the tiebreak for
        # the degenerate car-sits-on-the-wall-line case. This is self-correcting
        # for travel direction: a wall on the right offsets left, and vice versa.
        nrm = (d[1], -d[0])                          # arbitrary perpendicular
        w = -(mf * nrm[0] + ml * nrm[1])             # car (0,0) side vs the line
        if abs(w) < 0.05:
            if side < 0.0:
                nrm = (d[1], -d[0])                  # blue: centre right of wall
            else:
                nrm = (-d[1], d[0])                  # yellow: centre left of wall
        elif w < 0.0:
            nrm = (-nrm[0], -nrm[1])                 # flip: point toward the car

        # sample along the offset line. The bridge STARTS AT THE CAR'S OWN
        # PROJECTION onto the line (not at the wall's near end): the first
        # points then sit within ~1.5-2 m of the car, so the greedy chain
        # (max hop 3.5 m) actually CONNECTS the buffer to the bridge. Without
        # this, feeds at corners landed >3.5 m from the buffer end and the
        # chain silently ignored them (measured 2026-08-25: 32 feeds, path
        # never grew, car froze at the top-right 90 deg corner).
        ts = [(p[0] - mf) * d[0] + (p[1] - ml) * d[1] for p in fit]
        t1 = max(ts) + self.PSEUDO_EXTRAP_M
        t_car = (-mf) * d[0] + (-ml) * d[1]          # car (0,0) projected on the line
        t_start = min(t_car, t1)                     # never start past the far end
        max_len = (self.PSEUDO_MAX_LEN_M if skew <= self.PSEUDO_SHORT_SKEW_RAD
                   else self.PSEUDO_SHORT_LEN_M)     # steep skew = short hop
        pts = []
        t = t_start
        while t <= t1 and (t - t_start) <= max_len:
            pf = mf + t * d[0] + self.PSEUDO_HALF_WIDTH_M * nrm[0]
            pl = ml + t * d[1] + self.PSEUDO_HALF_WIDTH_M * nrm[1]
            if pf > self.PSEUDO_MIN_AHEAD_M:
                pts.append((cx + cs * pf - sn * pl, cy + sn * pf + cs * pl))
            t += self.PSEUDO_STEP_M
        if not pts:
            self.last_pseudo_reject = "all sampled points fell behind the bumper"
        return pts

    def feed_pseudo_points(self, pts):
        """P14: merge pseudo centreline points into the rolling buffer with the
        SAME dedup rule as PP frames. Pseudo entries are tagged and pruned
        aggressively (after a short grace: passed / too-far-sideways / expired)
        so they can never hijack the chain after their moment has passed.
        Returns the number of points actually added."""
        now = self.get_clock().now().nanoseconds * 1e-9
        self._last_pseudo_feed_t = now
        added = 0
        for (px, py) in pts:
            dup = False
            for a in self._accum:
                if (px - a[0]) ** 2 + (py - a[1]) ** 2 < self.DEDUP_R ** 2:
                    dup = True
                    break
            if not dup:
                self._accum.append((px, py, now, True))
                added += 1
        return added

    def bridge_fresh(self, max_age_s=1.5):
        """P14-c: True while a pseudo bridge was fed very recently. While
        fresh, control_node tells MPPI to trust the path despite a large
        heading offset -- the bridge was synthesized from live cones THIS
        cycle, so the one-sided-garbage veto must not eat it."""
        now = self.get_clock().now().nanoseconds * 1e-9
        return (now - self._last_pseudo_feed_t) <= max_age_s

    def seconds_since_real_points(self):
        """P18-b: sim-time seconds since the rolling buffer last accepted a
        REAL PP point. A large value means PP has delivered nothing new for a
        long while (empty / dead-end chains) -- the starvation-unlock signal.
        Uses the SIM clock on purpose: a Gazebo pause freezes both sides, so
        silence cannot accumulate while paused."""
        if self._last_real_pt_t is None:
            return 1e9
        now = self.get_clock().now().nanoseconds * 1e-9
        return max(0.0, now - self._last_real_pt_t)

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
        now = self.get_clock().now().nanoseconds * 1e-9
        added_real = False
        for (px, py) in pts[1:]:
            dup = False
            for a in self._accum:
                if (px - a[0]) ** 2 + (py - a[1]) ** 2 < self.DEDUP_R ** 2:
                    dup = True
                    break
            if not dup:
                self._accum.append((px, py, now, False))
                added_real = True
        if added_real:
            self._last_real_pt_t = now      # P18-b: PP is alive and extending the buffer

    def _prune_accum(self):
        """Drop accumulated points the car has driven PAST (behind it) or that are
        absurdly far ahead. Keeps the buffer to the live forward track.
        PSEUDO entries are pruned much harder (P14): the moment the car passes
        them, drifts too far to their side, or after TTL — a stale pseudo point
        beside the track otherwise hijacks the greedy chain and drags the CTE
        (measured runaway: CTE -11 -> -14 -> -25 m)."""
        if not self._accum:
            return
        cx, cy, cyaw = self.cx, self.cy, self.yaw
        fx, fy = math.cos(cyaw), math.sin(cyaw)        # car forward unit vector
        now = self.get_clock().now().nanoseconds * 1e-9
        kept = []
        for (px, py, ts, pseudo) in self._accum:
            dx, dy = px - cx, py - cy
            along = dx * fx + dy * fy                  # signed distance along heading
            dist  = math.hypot(dx, dy)
            if dist > self.AHEAD_MAX:                  # too far ahead -> discard
                continue
            if pseudo:
                # P14-b/c: pseudo points are a bridge, pruned by their own
                # rules. A FRESH bridge is exempt from the heading-relative
                # rules entirely: at a ~90 deg corner it runs perpendicular
                # to the nose (behind-plane and wide-lateral by definition),
                # and the passed/lateral/behind rules pruned every fresh
                # bridge within one cycle (419-feeds freeze, 2026-08-25).
                age = now - ts
                if age > self.PSEUDO_TTL_S:            # expired -> discard
                    continue
                if age > self.PSEUDO_GRACE_S:
                    if along < self.PSEUDO_PASSED_M:   # consumed -> discard
                        continue
                    lat = abs(-fx * dy + fy * dx)      # |lateral| off the nose
                    if lat > self.PSEUDO_LAT_MAX_M:    # drifted beside -> discard
                        continue
                kept.append((px, py, ts, pseudo))
                continue
            # real PP points: original rules
            if along < -self.BEHIND_M:                 # behind the car -> discard
                continue
            kept.append((px, py, ts, pseudo))
        self._accum = kept

    def get_state(self):
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
        return (self.cx, self.cy, self.yaw, spd, self.have_odom)

    def get_path(self):
        """
        Return the accumulated rolling-window path AHEAD of the car (map frame),
        ordered ALONG THE TRACK by greedy nearest-neighbour chaining from the car.
        Points the car has passed are pruned. Falls back to the last good raw PP
        frame if the buffer is too thin.
        """
        self._prune_accum()
        pts = [(p[0], p[1]) for p in self._accum]
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
