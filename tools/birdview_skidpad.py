#!/usr/bin/env python3
"""
birdview_skidpad.py  -  the bird view with a skidpad scoreboard

    python3 tools/birdview_skidpad.py          (with stack_skidpad.launch.py running)

Everything tools/birdview.py shows, plus a scoreboard on the map:

    circle    try 1      try 2 (timed)
    right     13.11 s    12.46 s
    left      13.28 s    12.94 s
    now       left 2     4.31 s
    cones hit 2   (right 2: 1, exit: 1)
    result    12.70 s + 2 x 0.2 s = 13.10 s

* Circle times: the time between two passes of the car through the timing line at the
  crossing, in simulation time (odometry stamps). The timing line comes from the skidpad
  planner's fitted layout (/pp/mode "pose": crossing point and driving direction), so it only
  starts once the layout is fitted. Each lap is named by the planner's phase right after the
  crossing (right 1, right 2, left 1, left 2).
* Cones hit: read straight from Gazebo's contact stream (gz topic -e .../physics/contacts, the
  same source as the run recorder): a contact between the car (model "eufs") and any model with
  "cone" in its name. One hit = one cone touched; touching the same cone again within 1 s counts
  once. Each hit is booked to the phase the car was in. Needs the `gz` command (Gazebo classic);
  without it the line says so.
* Result: mean of the two timed laps (try 2 of each circle) + 0.2 s per cone hit, the skidpad
  scoring of the FS rules (D 10 penalties: 0.2 s per cone down or out in skidpad). An
  estimate: the officials count cones down/out, the simulator counts touches.

Keys as in the bird view. `k` resets the scoreboard (new run without restarting the view).
"""
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import birdview as bv                                   # noqa: E402

CONE_PENALTY_S = 0.2
SAME_CONE_GAP_S = 1.0
VEHICLE_RE = re.compile(r"(^|::|/)eufs(::|/|$)", re.I)
CONE_RE = re.compile(r"cone", re.I)


def native_contacts(lines):
    """pairs of collision names from `gz topic -e .../physics/contacts` text output"""
    depth, names = 0, {}
    for raw in lines:
        line = raw.strip()
        if line.startswith("contact {"):
            depth, names = 1, {}
            continue
        if not depth:
            continue
        m = re.match(r'(collision[12]):\s*"([^"]+)"', line)
        if m:
            names[m[1]] = m[2]
        if line.endswith("{"):
            depth += 1
        elif line == "}":
            depth -= 1
            if depth == 0 and len(names) == 2:
                yield names["collision1"], names["collision2"]


class SkidpadNode(bv.BirdViewNode):
    def __init__(self):
        super().__init__()
        self.sk_reset()
        self.contact_status = "starting"
        threading.Thread(target=self._contacts, daemon=True).start()

    def sk_reset(self):
        with self.lock:
            self.crossings = []          # (sim_t, phase after the crossing)
            self.prev_lon = None
            self.hits = []               # (cone name, phase, wall time)
            self.last_touch = {}         # cone name -> wall time of the last contact
            self.cur_phase = "entry"

    # ---------------------------------------------------------- timing line
    def cb_odom(self, m):
        super().cb_odom(m)
        with self.lock:
            M = self.pp_mode or {}
            if M.get("phase"):
                self.cur_phase = str(M["phase"])
            pose = M.get("pose")
            car = self.car
            if not pose or car is None or M.get("mode") != "SKIDPAD":
                return
            tx, ty, th = pose
            dx, dy = car[0] - tx, car[1] - ty
            lon = dx * math.cos(th) + dy * math.sin(th)
            lat = -dx * math.sin(th) + dy * math.cos(th)
            if self.prev_lon is not None and self.prev_lon < 0.0 <= lon and abs(lat) < 3.0:
                if not self.crossings or car[4] - self.crossings[-1][0] > 3.0:
                    self.crossings.append([car[4], None])
            self.prev_lon = lon
            # name the lap that just started by the planner's phase once it has switched
            for c in self.crossings:
                if c[1] is None and car[4] - c[0] > 0.5:
                    c[1] = self.cur_phase

    # ---------------------------------------------------------- Gazebo contacts
    def _contacts(self):
        if shutil.which("gz") is None:
            self.contact_status = "gz not found: cone hits not available"
            return
        while True:
            topic = None
            try:
                out = subprocess.run(["gz", "topic", "-l"], text=True, capture_output=True, timeout=3).stdout
                topic = next((l.strip() for l in out.splitlines() if l.strip().endswith("/physics/contacts")), None)
            except Exception:
                pass
            if not topic:
                self.contact_status = "waiting for Gazebo contacts topic"
                time.sleep(2.0)
                continue
            cmd = ["gz", "topic", "-e", topic]
            if shutil.which("stdbuf"):
                cmd = ["stdbuf", "-oL"] + cmd
            self.contact_status = "ok"
            try:
                with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1) as p:
                    for a, b in native_contacts(p.stdout):
                        for veh, cone in ((a, b), (b, a)):
                            if VEHICLE_RE.search(veh) and CONE_RE.search(cone):
                                self._touch(cone.split("::")[1] if "::" in cone else cone)
                                break
            except Exception as exc:
                self.contact_status = f"contacts stream failed ({type(exc).__name__})"
            time.sleep(1.0)

    def _touch(self, cone):
        now = time.monotonic()
        with self.lock:
            last = self.last_touch.get(cone)
            self.last_touch[cone] = now
            if last is not None and now - last < SAME_CONE_GAP_S:
                return
            if any(h[0] == cone for h in self.hits):    # same cone again later: still one cone
                return
            self.hits.append((cone, self.cur_phase, now))

    def scoreboard(self):
        with self.lock:
            cr = [tuple(c) for c in self.crossings]
            hits = list(self.hits)
            car = self.car
            phase = self.cur_phase
        laps = {}
        for (t0, name), (t1, _) in zip(cr[:-1], cr[1:]):
            if name:
                laps[name] = t1 - t0
        cur = None
        if cr and car is not None and phase not in ("entry", "exit", "stopped"):
            cur = (cr[-1][1] or phase, car[4] - cr[-1][0])
        return laps, cur, hits, phase


class SkidpadView(bv.BirdView):
    def __init__(self, node):
        super().__init__(node)
        self.fig.canvas.manager.set_window_title("RAMS bird view - skidpad")
        self.board = self.ax.text(0.99, 0.99, "", transform=self.ax.transAxes, ha="right", va="top",
                                  family="monospace", fontsize=11, color="#ffffff", zorder=30, multialignment="left",
                                  bbox=dict(facecolor="#1b1d1b", edgecolor="#ffb000", alpha=0.88,
                                            boxstyle="round,pad=0.5"))

    def on_key(self, e):
        if e.key == "k":
            self.node.sk_reset()
            return
        super().on_key(e)

    def update(self):
        super().update()
        laps, cur, hits, phase = self.node.scoreboard()
        f = lambda v: f"{v:6.2f} s" if v is not None else "    -   "
        L = ["SKIDPAD          try 1     try 2 (timed)"]
        for side in ("right", "left"):
            L.append(f"{side:<6}         {f(laps.get(side + ' 1'))}  {f(laps.get(side + ' 2'))}")
        if cur is not None:
            L.append(f"now    {cur[0]:<8} {cur[1]:6.2f} s")
        else:
            L.append(f"phase  {phase}")
        st = self.node.contact_status
        if st == "ok":
            by = {}
            for _, ph, _ in hits:
                by[ph] = by.get(ph, 0) + 1
            where = ", ".join(f"{k}: {v}" for k, v in by.items())
            L.append(f"cones hit  {len(hits)}" + (f"   ({where})" if where else ""))
            if hits:
                L.append("  last: " + ", ".join(h[0].replace("_cone", "") for h in hits[-3:]))
        else:
            L.append(f"cones hit  ? ({st})")
        r2, l2 = laps.get("right 2"), laps.get("left 2")
        if r2 is not None and l2 is not None:
            mean = 0.5 * (r2 + l2)
            n = len(hits) if st == "ok" else 0
            L.append(f"result     {mean:.2f} s + {n} x {CONE_PENALTY_S} s = {mean + CONE_PENALTY_S * n:.2f} s")
        L.append("k = reset scoreboard")
        self.board.set_text("\n".join(L))


def main():
    if bv.matplotlib.get_backend().lower() == "agg":
        try:
            bv.matplotlib.use("TkAgg")
        except Exception:
            pass
    bv.rclpy.init()
    node = SkidpadNode()
    threading.Thread(target=bv.rclpy.spin, args=(node,), daemon=True).start()
    view = SkidpadView(node)
    bv.plt.show(block=False)
    period = 1.0 / bv.REDRAW_HZ
    try:
        while view.running and bv.plt.fignum_exists(view.fig.number):
            t0 = time.monotonic()
            view.update()
            view.fig.canvas.draw_idle()
            bv.plt.pause(max(0.001, period - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        bv.plt.close("all")
        node.destroy_node()
        if bv.rclpy.ok():
            bv.rclpy.shutdown()


if __name__ == "__main__":
    main()
