#!/usr/bin/env python3
"""
birdview.py  -  live bird's-eye view of the whole RAMS-e-25 stack (read-only)

Run in its own terminal while Gazebo + bringup (+ controls) are running:
    rams_abh && python3 ~/birdview.py
It only SUBSCRIBES (never publishes), so it cannot change what the car does.

Shows
    cones            /slam/map_cones        blue / yellow / orange / big orange
    car + trail      /slam/odom_raw         grey trail, brighter = faster (cleared on sim reset)
    FOV sector       same sector as the planner (vertex -5 m, 30 m, 60 deg)
    planner path     /path_points           red, what the planner publishes
    controller path  /debug/buffer_path     wide green halo = what the controller actually tracks
                                            (normally right under the red planner path)
    pseudo bridge    /debug/pseudo_path     magenta dashes, controller's straight-line guess
    MPPI rollouts    /debug/rollouts        thin grey, best one darker
    planner insides  /pp/debug_markers + /pp/mode   (PCD node only)
                     traced walls (lines in the cone colour), predicted cones
                     (magenta diamonds), virtual cones (white circles), Delaunay
                     midpoints (cyan dots); NORMAL/FALLBACK modes: Delaunay edges
                     mode + reason + measured track width in the panel
    status panel     speed, steering + accel command (/cmd), topic rates,
                     path length, lap count + lap times, distance driven,
                     closest cone (warns when < 0.6 m)

Keys (click the plot window first)
    f   follow car  <->  whole track
    + / -   zoom in / out (follow mode)
    r   rollouts on / off
    p   planner insides on / off
    t   clear trail + laps
    s   save a PNG screenshot to the current folder
    q   quit
"""
import math
import threading
import time
from collections import deque

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon, Wedge

import json

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from nav_msgs.msg import Odometry, Path
from eufs_msgs.msg import ConeArrayWithCovariance

try:
    from ackermann_msgs.msg import AckermannDriveStamped
except ImportError:
    AckermannDriveStamped = None
try:
    from std_msgs.msg import String
except ImportError:
    String = None
try:
    from visualization_msgs.msg import MarkerArray
except ImportError:
    MarkerArray = None

# ---------------------------------------------------------------- settings
ODOM_TOPIC = "/slam/odom_raw"
FOV_VERTEX, FOV_RADIUS, FOV_ANGLE = -5.0, 30.0, 60.0
FOLLOW_RADIUS = 22.0          # half-width of the view in follow mode [m]
SPEED_MAX = 8.0               # top of the trail colour scale [m/s]
CONE_WARN = 0.6               # [m] car centre to cone: "close to a cone"
LAP_MIN_DIST = 60.0           # [m] driven before a return to the start counts as a lap
LAP_RADIUS = 3.0              # [m]
REDRAW_HZ = 15.0


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def stamp_s(st):
    return st.sec + st.nanosec * 1e-9


class Rate:
    """Messages per second over the last 2 s (wall clock)."""
    def __init__(self):
        self.t = deque(maxlen=200)

    def tick(self):
        self.t.append(time.monotonic())

    def hz(self):
        now = time.monotonic()
        while self.t and now - self.t[0] > 2.0:
            self.t.popleft()
        return len(self.t) / 2.0

    def age(self):
        return time.monotonic() - self.t[-1] if self.t else float("inf")


class BirdViewNode(Node):
    def __init__(self):
        super().__init__("birdview")
        qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=10,
                         reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         durability=QoSDurabilityPolicy.VOLATILE)
        self.lock = threading.Lock()
        self.cones = {k: np.zeros((0, 2)) for k in ("blue", "yellow", "orange", "big")}
        self.car = None                 # (x, y, yaw, speed, sim_t)
        self.trail = []                 # (x, y, speed)
        self.path = np.zeros((0, 2))
        self.buffer = np.zeros((0, 2))
        self.pseudo = np.zeros((0, 2))
        self.rollouts = []
        self.cmd = None                 # (steer, accel)
        self.pp = {}                    # ns -> (N,2) points, from /pp/debug_markers
        self.pp_mode = None             # dict from /pp/mode
        self.rate = {k: Rate() for k in ("odom", "map", "path", "buffer", "cmd", "ppdbg")}
        self.dist = 0.0
        self.lap_start = None           # (x, y, sim_t, dist_at_start)
        self.laps = []

        self.create_subscription(Odometry, ODOM_TOPIC, self.cb_odom, qos)
        self.create_subscription(ConeArrayWithCovariance, "/slam/map_cones", self.cb_map, qos)
        self.create_subscription(Path, "/path_points", self.cb_path, qos)
        self.create_subscription(Path, "/debug/buffer_path", self.cb_buffer, qos)
        self.create_subscription(Path, "/debug/pseudo_path", self.cb_pseudo, qos)
        if AckermannDriveStamped is not None:
            self.create_subscription(AckermannDriveStamped, "/cmd", self.cb_cmd, qos)
        if MarkerArray is not None:
            self.create_subscription(MarkerArray, "/debug/rollouts", self.cb_rollouts, qos)
            self.create_subscription(MarkerArray, "/pp/debug_markers", self.cb_pp_markers, qos)
        if String is not None:
            self.create_subscription(String, "/pp/mode", self.cb_pp_mode, qos)

    # ---------------------------------------------------------- callbacks
    def cb_odom(self, m):
        p = m.pose.pose.position
        v = math.hypot(m.twist.twist.linear.x, m.twist.twist.linear.y)
        t = stamp_s(m.header.stamp)
        with self.lock:
            self.rate["odom"].tick()
            if self.trail:
                lx, ly, _ = self.trail[-1]
                d = math.hypot(p.x - lx, p.y - ly)
                if d < 0.15:
                    self.car = (p.x, p.y, yaw_of(m.pose.pose.orientation), v, t)
                    return
                if d >= 5.0:                     # teleport = sim reset: start a fresh trail + laps
                    self.trail, self.laps, self.dist, self.lap_start = [], [], 0.0, None
                else:
                    self.dist += d
            self.trail.append((p.x, p.y, v))
            if len(self.trail) > 30000:
                del self.trail[:5000]
            self.car = (p.x, p.y, yaw_of(m.pose.pose.orientation), v, t)
            if self.lap_start is None:
                self.lap_start = (p.x, p.y, t, self.dist)
            else:
                sx, sy, st, sd = self.lap_start
                if self.dist - sd > LAP_MIN_DIST and math.hypot(p.x - sx, p.y - sy) < LAP_RADIUS:
                    self.laps.append(t - st)
                    self.lap_start = (sx, sy, t, self.dist)

    def cb_map(self, m):
        def arr(lst):
            return np.array([[c.point.x, c.point.y] for c in lst], float).reshape(-1, 2)
        with self.lock:
            self.rate["map"].tick()
            self.cones = dict(blue=arr(m.blue_cones), yellow=arr(m.yellow_cones),
                              orange=arr(m.orange_cones), big=arr(m.big_orange_cones))

    @staticmethod
    def _xy(m):
        return np.array([[p.pose.position.x, p.pose.position.y] for p in m.poses], float).reshape(-1, 2)

    def cb_path(self, m):
        with self.lock:
            self.rate["path"].tick(); self.path = self._xy(m)

    def cb_buffer(self, m):
        with self.lock:
            self.rate["buffer"].tick(); self.buffer = self._xy(m)

    def cb_pseudo(self, m):
        with self.lock:
            self.pseudo = self._xy(m)

    def cb_cmd(self, m):
        with self.lock:
            self.rate["cmd"].tick()
            self.cmd = (m.drive.steering_angle, m.drive.acceleration)

    def cb_rollouts(self, m):
        lines = []
        for mk in m.markers:
            if len(mk.points) >= 2:
                lines.append(np.array([[q.x, q.y] for q in mk.points], float))
        with self.lock:
            self.rollouts = lines

    def cb_pp_markers(self, m):
        out = {}
        for mk in m.markers:
            if mk.ns and mk.points:
                out[mk.ns] = np.array([[q.x, q.y] for q in mk.points], float)
        with self.lock:
            self.rate["ppdbg"].tick()
            self.pp = out

    def cb_pp_mode(self, m):
        try:
            d = json.loads(m.data)
        except ValueError:
            d = {"mode": str(m.data)}
        with self.lock:
            self.pp_mode = d

    def snapshot(self):
        with self.lock:
            return dict(cones={k: v.copy() for k, v in self.cones.items()}, car=self.car,
                        trail=np.array(self.trail, float).reshape(-1, 3), path=self.path.copy(),
                        buffer=self.buffer.copy(), pseudo=self.pseudo.copy(),
                        rollouts=list(self.rollouts), cmd=self.cmd,
                        pp=dict(self.pp), pp_mode=self.pp_mode, dist=self.dist,
                        laps=list(self.laps), lap_start=self.lap_start,
                        rate={k: (r.hz(), r.age()) for k, r in self.rate.items()})

    def clear_trail(self):
        with self.lock:
            self.trail, self.laps, self.dist, self.lap_start = [], [], 0.0, None


# =================================================================== view
class BirdView:
    def __init__(self, node):
        self.node = node
        self.follow = True
        self.zoom = FOLLOW_RADIUS
        self.show_roll = True
        self.show_pp = True
        self.running = True

        plt.rcParams["toolbar"] = "None"
        self.fig = plt.figure("RAMS bird view", figsize=(13, 8))
        self.ax = self.fig.add_axes([0.0, 0.0, 0.74, 1.0])
        self.hud = self.fig.add_axes([0.74, 0.0, 0.26, 1.0]); self.hud.axis("off")
        ax = self.ax
        ax.set_aspect("equal"); ax.set_facecolor("#3a3f3a")
        ax.tick_params(colors="#bbbbbb", labelsize=7)
        ax.grid(True, color="#4d544d", lw=0.5)
        self.fig.patch.set_facecolor("#262826")

        from matplotlib.colors import LinearSegmentedColormap
        trail_cmap = LinearSegmentedColormap.from_list("trail", ["#6e6e6e", "#ffffff"])
        self.trail_lc = LineCollection([], cmap=trail_cmap, lw=1.8, alpha=0.8, zorder=1)
        self.trail_lc.set_clim(0, SPEED_MAX); ax.add_collection(self.trail_lc)   # grey (slow) .. white (fast)
        self.roll_lc = LineCollection([], colors="#d0d0d0", lw=0.6, alpha=0.35); ax.add_collection(self.roll_lc)
        self.roll_best, = ax.plot([], [], "-", color="#ffffff", lw=1.2, alpha=0.8, label="MPPI best rollout")
        self.fov = Wedge((0, 0), FOV_RADIUS, 0, 1, facecolor="#ffffff", alpha=0.04, edgecolor="#ffffff", lw=0.8)
        ax.add_patch(self.fov)
        st = dict(ls="none", markeredgecolor="black", markeredgewidth=0.4)
        self.c_blue, = ax.plot([], [], "^", color="#2f6bff", ms=6, **st)
        self.c_yel, = ax.plot([], [], "^", color="#ffd400", ms=6, **st)
        self.c_or, = ax.plot([], [], "^", color="#ff8c1a", ms=5, **st)
        self.c_big, = ax.plot([], [], "^", color="#ff5a00", ms=9, **st)
        self.pp_lwall, = ax.plot([], [], "-", lw=2.4, alpha=0.9, zorder=4)
        self.pp_rwall, = ax.plot([], [], "-", lw=2.4, alpha=0.9, zorder=4)
        self.pp_edges = LineCollection([], colors="#cfcfcf", lw=0.6, alpha=0.45, zorder=3); ax.add_collection(self.pp_edges)
        self.pp_mids, = ax.plot([], [], ".", color="#29e6ff", ms=4, zorder=5, label="planner midpoints")
        self.pp_virt, = ax.plot([], [], "o", ms=7, mfc="none", mec="white", mew=1.4, zorder=6, label="virtual cones")
        self.pp_pred, = ax.plot([], [], "D", ms=7, color="#ff4dff", mec="black", mew=0.5, zorder=6, label="predicted cones")
        self.pseudo, = ax.plot([], [], "--", color="#ff4dff", lw=1.5, label="controller pseudo bridge")
        self.buffer, = ax.plot([], [], "-", color="#3dff6e", lw=6.0, alpha=0.45, solid_capstyle="round",
                               label="controller is tracking (halo)")
        self.path, = ax.plot([], [], ".-", color="#ff3030", lw=1.4, ms=3, label="planner /path_points", zorder=7)
        self.car = Polygon(np.zeros((3, 2)), closed=True, facecolor="white", edgecolor="black", lw=1.0, zorder=10)
        ax.add_patch(self.car)
        self.warn = ax.text(0.5, 0.97, "", transform=ax.transAxes, ha="center", va="top",
                            color="white", fontsize=12, weight="bold",
                            bbox=dict(facecolor="#c0001a", edgecolor="none", alpha=0.0))
        leg = ax.legend(loc="lower left", fontsize=7, facecolor="#262826", edgecolor="#555555", labelcolor="#dddddd")
        leg.set_zorder(20)
        cb = self.fig.colorbar(self.trail_lc, ax=self.hud, location="bottom", fraction=0.04, pad=0.02, shrink=0.8)
        cb.set_label("trail speed [m/s] (grey = slow, white = fast)", color="#cccccc", fontsize=8)
        cb.ax.tick_params(colors="#cccccc", labelsize=7)
        self.txt = self.hud.text(0.04, 0.98, "", va="top", ha="left", family="monospace",
                                 fontsize=9.5, color="#e8e8e8", transform=self.hud.transAxes)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("close_event", lambda e: setattr(self, "running", False))

    def on_key(self, e):
        k = e.key
        if k == "f": self.follow = not self.follow
        elif k in ("+", "="): self.zoom = max(6.0, self.zoom / 1.3)
        elif k in ("-", "_"): self.zoom = min(120.0, self.zoom * 1.3)
        elif k == "r": self.show_roll = not self.show_roll
        elif k == "p": self.show_pp = not self.show_pp
        elif k == "t": self.node.clear_trail()
        elif k == "s":
            fn = time.strftime("birdview_%Y%m%d_%H%M%S.png")
            self.fig.savefig(fn, dpi=150, facecolor=self.fig.get_facecolor()); print("saved", fn)
        elif k == "q": self.running = False

    def update(self):
        s = self.node.snapshot()
        C = s["cones"]
        for line, k in ((self.c_blue, "blue"), (self.c_yel, "yellow"), (self.c_or, "orange"), (self.c_big, "big")):
            line.set_data(C[k][:, 0], C[k][:, 1])
        T = s["trail"]
        if len(T) >= 2:
            seg = np.stack([T[:-1, :2], T[1:, :2]], axis=1)
            self.trail_lc.set_segments(seg); self.trail_lc.set_array(T[1:, 2])
        else:
            self.trail_lc.set_segments([])
        for line, k in ((self.path, "path"), (self.buffer, "buffer"), (self.pseudo, "pseudo")):
            line.set_data(s[k][:, 0], s[k][:, 1])
        D = s["pp"] if self.show_pp else {}
        M = s["pp_mode"] or {}
        def ns(name):
            a = D.get(name)
            return (a[:, 0], a[:, 1]) if a is not None and len(a) else ([], [])
        blue_left = M.get("left") == "BLUE"
        self.pp_lwall.set_color("#2f6bff" if blue_left else "#ffd400")
        self.pp_rwall.set_color("#ffd400" if blue_left else "#2f6bff")
        self.pp_lwall.set_data(*ns("left_wall")); self.pp_rwall.set_data(*ns("right_wall"))
        self.pp_mids.set_data(*ns("midpoints")); self.pp_virt.set_data(*ns("virtual"))
        self.pp_pred.set_data(*ns("predicted"))
        E = D.get("edges")
        self.pp_edges.set_segments(E.reshape(-1, 2, 2) if E is not None and len(E) >= 2 and len(E) % 2 == 0 else [])
        R = s["rollouts"] if self.show_roll else []
        self.roll_lc.set_segments(R[1:] if len(R) > 1 else [])
        self.roll_best.set_data(*(R[0].T if R else ([], [])))

        car = s["car"]
        allc = np.vstack([v for v in C.values() if len(v)]) if any(len(v) for v in C.values()) else np.zeros((0, 2))
        near = float("inf")
        if car is not None:
            x, y, yaw, v, _ = car
            c, sn = math.cos(yaw), math.sin(yaw)
            shape = np.array([[1.6, 0.0], [-0.9, 0.75], [-0.9, -0.75]])
            self.car.set_xy(shape @ np.array([[c, sn], [-sn, c]]) + [x, y])
            self.fov.set_center((x + FOV_VERTEX * c, y + FOV_VERTEX * sn))
            self.fov.set_theta1(math.degrees(yaw) - FOV_ANGLE / 2)
            self.fov.set_theta2(math.degrees(yaw) + FOV_ANGLE / 2)
            if len(allc):
                near = float(np.min(np.hypot(allc[:, 0] - x, allc[:, 1] - y)))
            if self.follow:
                self.ax.set_xlim(x - self.zoom, x + self.zoom); self.ax.set_ylim(y - self.zoom, y + self.zoom)
        if (not self.follow or car is None):
            pts = [a for a in (allc, T[:, :2] if len(T) else np.zeros((0, 2))) if len(a)]
            if pts:
                P = np.vstack(pts); lo, hi = P.min(0) - 5, P.max(0) + 5
                ctr, half = (lo + hi) / 2, max(hi - lo) / 2
                self.ax.set_xlim(ctr[0] - half * 1.3, ctr[0] + half * 1.3); self.ax.set_ylim(ctr[1] - half, ctr[1] + half)

        if near < CONE_WARN:
            self.warn.set_text(f"CLOSE TO A CONE  {near:.2f} m"); self.warn.get_bbox_patch().set_alpha(0.85)
        else:
            self.warn.set_text(""); self.warn.get_bbox_patch().set_alpha(0.0)

        # ---- status panel
        r = s["rate"]
        def rt(k):
            hz, age = r[k]
            return f"{hz:5.1f} Hz" if age < 1.0 else ("   none " if age == float("inf") else f" stale {age:3.0f}s")
        L = []
        L.append("RAMS bird view")
        L.append("-" * 26)
        if car is not None:
            L.append(f"speed      {car[3]:5.2f} m/s")
            L.append(f"           {car[3] * 3.6:5.1f} km/h")
        else:
            L.append(f"waiting for {ODOM_TOPIC}")
        if s["cmd"] is not None:
            st, ac = s["cmd"]
            L.append(f"steer cmd  {math.degrees(st):+6.1f} deg")
            L.append(f"accel cmd  {ac:+6.2f} m/s2")
            bar = int(round(np.clip(st / 0.52, -1, 1) * 10))
            L.append("  L " + "".join("#" if (bar > 0 and 10 - bar <= i < 10) or (bar < 0 and 10 < i <= 10 - bar) else ("|" if i == 10 else ".") for i in range(21)) + " R")
        else:
            L.append("no /cmd yet (controls off?)")
        L.append("")
        P = s["path"]
        plen = float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))) if len(P) > 1 else 0.0
        M = s["pp_mode"]
        if M and r["ppdbg"][1] < 1.0:
            L.append(f"planner mode  {M.get('mode', '?')}")
            if M.get("reason"):
                L.append(f"  ({str(M['reason'])[:24]})")
            if M.get("mode") == "PCD":
                L.append(f"track width   {float(M.get('width', 0)):.2f} m")
                L.append(f"left wall     {M.get('left')}")
        else:
            L.append("planner mode  - (no /pp/mode:")
            L.append("  team planner or old PCD file)")
        L.append(f"planner    {len(P):3d} pts {plen:5.1f} m")
        L.append(f"closest cone {near:5.2f} m" if near < float("inf") else "closest cone   -")
        L.append(f"cones in map {sum(len(v) for v in C.values()):4d}")
        L.append("")
        L.append(f"driven     {s['dist']:7.1f} m")
        L.append(f"laps       {len(s['laps']):3d}")
        if s["laps"]:
            L.append(f"last lap   {s['laps'][-1]:6.1f} s")
            L.append(f"best lap   {min(s['laps']):6.1f} s")
        if s["lap_start"] is not None and car is not None:
            L.append(f"this lap   {car[4] - s['lap_start'][2]:6.1f} s")
        L.append("")
        L.append("topic rates")
        L.append(f" odom      {rt('odom')}")
        L.append(f" map       {rt('map')}")
        L.append(f" path      {rt('path')}")
        L.append(f" ctrl buf  {rt('buffer')}")
        L.append(f" cmd       {rt('cmd')}")
        L.append("")
        L.append("keys: f follow/whole  +/- zoom")
        L.append("      r rollouts  p planner insides")
        L.append("      t clear trail")
        L.append("      s screenshot  q quit")
        L.append(f"view: {'FOLLOW' if self.follow else 'WHOLE TRACK'}")
        self.txt.set_text("\n".join(L))


def main():
    if matplotlib.get_backend().lower() == "agg":
        try:
            matplotlib.use("TkAgg")
        except Exception:
            pass
    rclpy.init()
    node = BirdViewNode()
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()
    view = BirdView(node)
    plt.show(block=False)
    period = 1.0 / REDRAW_HZ
    try:
        while view.running and plt.fignum_exists(view.fig.number):
            t0 = time.monotonic()
            view.update()
            view.fig.canvas.draw_idle()
            plt.pause(max(0.001, period - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        plt.close("all")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
