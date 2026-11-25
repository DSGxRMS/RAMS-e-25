#!/usr/bin/env python3
"""Record the ROS and Gazebo evidence needed by report.py."""
import argparse
import json
import math
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path


def finite(value):
    return value if math.isfinite(value) else None


def xy(point):
    return [finite(point.x), finite(point.y)]


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def odom(msg):
    pose, twist = msg.pose.pose, msg.twist.twist
    q = pose.orientation
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                     1 - 2 * (q.y * q.y + q.z * q.z))
    return dict(xy=xy(pose.position), yaw=finite(yaw), vx=finite(twist.linear.x),
                vy=finite(twist.linear.y), yaw_rate=finite(twist.angular.z))


class Archive:
    def __init__(self, path):
        self.file = path.open("x", encoding="utf-8")
        self.start = time.monotonic_ns()
        self.sim = None
        self.lock = threading.Lock()

    def write(self, kind, data, msg=None):
        with self.lock:
            row = dict(kind=kind, t=(time.monotonic_ns() - self.start) / 1e9,
                       wall_ns=time.time_ns(), sim_ns=self.sim, data=data)
            if msg is not None and hasattr(msg, "header"):
                row.update(frame=msg.header.frame_id, stamp_ns=stamp_ns(msg.header.stamp))
            self.file.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            self.file.flush()


def native_contacts(lines):
    """Decode Gazebo Classic Contacts DebugString messages from gz topic -e."""
    names = {}
    depth = 0
    for raw in lines:
        line = raw.strip()
        if line == "contact {":
            depth, names = 1, {}
            continue
        if not depth:
            continue
        match = re.match(r'(collision[12]):\s*"([^"]+)"', line)
        if match:
            names[match[1]] = match[2]
        if line.endswith("{"):
            depth += 1
        elif line == "}":
            depth -= 1
            if depth == 0 and len(names) == 2:
                yield [names["collision1"], names["collision2"]]


class GazeboContacts:
    def __init__(self, archive, topic, vehicle, cone):
        self.archive = archive
        self.topic = topic
        self.vehicle = re.compile(vehicle, re.I)
        self.cone = re.compile(cone, re.I)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        if shutil.which("gz") is None:
            self.archive.write("coverage", {"contact": "gz executable unavailable"})
            return
        while not self.stop.is_set():
            topic = self.topic
            if topic == "auto":
                try:
                    listed = subprocess.run(["gz", "topic", "-l"], text=True,
                                            capture_output=True, timeout=2)
                    topic = next((line.strip() for line in listed.stdout.splitlines()
                                  if line.strip().endswith("/physics/contacts")), None)
                except subprocess.TimeoutExpired:
                    topic = None
            if not topic:
                self.stop.wait(1)
                continue
            if self.stop.is_set():
                break
            command = ["gz", "topic", "-e", topic]
            if shutil.which("stdbuf"):
                command = ["stdbuf", "-oL"] + command
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  text=True, bufsize=1) as process:
                self.process = process
                if self.stop.is_set():
                    process.terminate()
                self.archive.write("coverage", {"contact_topic": topic})
                observed = False
                for pair in native_contacts(process.stdout):
                    if not observed:
                        self.archive.write("coverage", {"native_contact_payload": True})
                        observed = True
                    if any(self.vehicle.search(a) and self.cone.search(b)
                           for a, b in (pair, pair[::-1])):
                        self.archive.write("contact_native", {"pairs": [pair]})
                    if self.stop.is_set():
                        break
                if process.poll() is None:
                    process.terminate()
            self.archive.write("coverage", {"contact_stream_end": topic})
            self.stop.wait(1)

    def close(self):
        self.stop.set()
        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            process.terminate()
        self.thread.join(timeout=4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New run directory")
    parser.add_argument("--gazebo-contacts", default="auto",
                        help="auto, off, or a Gazebo Classic transport topic")
    parser.add_argument("--ros-contacts", help="Optional gazebo_msgs/ContactsState ROS topic")
    parser.add_argument("--vehicle-pattern", default=r"(^|::|/)eufs(::|/|$)")
    parser.add_argument("--cone-pattern", default=r"cone")
    args = parser.parse_args()

    import rclpy
    from ackermann_msgs.msg import AckermannDriveStamped
    from eufs_msgs.msg import ConeArrayWithCovariance
    from nav_msgs.msg import Odometry, Path as PathMsg
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from rosgraph_msgs.msg import Clock
    from std_msgs.msg import String
    from visualization_msgs.msg import MarkerArray

    args.output.mkdir(parents=True, exist_ok=True)
    archive = Archive(args.output / "events.jsonl")
    archive.write("meta", dict(version=1, vehicle_pattern=args.vehicle_pattern,
                              cone_pattern=args.cone_pattern, gazebo_contacts=args.gazebo_contacts,
                              ros_contacts=args.ros_contacts))
    rclpy.init()
    node = rclpy.create_node("mppi_run_audit")
    qos = QoSProfile(depth=300, reliability=ReliabilityPolicy.BEST_EFFORT)

    def sub(cls, topic, kind, extract):
        def save(msg):
            archive.write(kind, extract(msg), msg)
        node.create_subscription(cls, topic, save, qos)

    def clock(msg):
        now = stamp_ns(msg.clock)
        prior = archive.sim
        archive.sim = now
        if prior is None or now < prior:
            archive.write("clock_epoch", {"previous_sim_ns": prior})

    def cones(msg):
        return {name: [xy(c.point) for c in getattr(msg, name + "_cones", [])]
                for name in ("blue", "yellow", "orange", "big_orange", "unknown_color")}

    node.create_subscription(Clock, "/clock", clock, qos)
    for topic, kind in (("/ground_truth/odom", "gt"), ("/slam/odom_raw", "odom_raw"),
                        ("/slam/odom", "odom_slam")):
        sub(Odometry, topic, kind, odom)
    for topic, kind in (("/path_points", "pp"), ("/debug/buffer_path", "buffer"),
                        ("/debug/pseudo_path", "pseudo")):
        sub(PathMsg, topic, kind, lambda m: {"points": [xy(p.pose.position) for p in m.poses]})
    sub(MarkerArray, "/debug/rollouts", "rollouts",
        lambda m: {"stamp_ns": stamp_ns(m.markers[0].header.stamp) if m.markers else None,
                   "paths": [[xy(p) for p in marker.points]
                             for marker in sorted(m.markers, key=lambda item: item.id)
                             if marker.action == 0]})
    sub(AckermannDriveStamped, "/cmd", "cmd",
        lambda m: {k: finite(getattr(m.drive, k))
                   for k in ("speed", "acceleration", "steering_angle", "steering_angle_velocity", "jerk")})
    sub(ConeArrayWithCovariance, "/ground_truth/cones", "gt_cones", cones)
    sub(ConeArrayWithCovariance, "/slam/map_cones", "map_cones", cones)
    sub(String, "/debug/mppi_cycle", "cycle", lambda m: json.loads(m.data))
    if args.ros_contacts:
        from gazebo_msgs.msg import ContactsState
        sub(ContactsState, args.ros_contacts, "contact_ros",
            lambda m: {"pairs": [[s.collision1_name, s.collision2_name] for s in m.states]})

    native = None
    if args.gazebo_contacts != "off":
        native = GazeboContacts(archive, args.gazebo_contacts, args.vehicle_pattern, args.cone_pattern)
    stopping = False

    def request_stop(_signal, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    print(f"Recording {args.output / 'events.jsonl'}  |  Ctrl+C to finish", flush=True)
    try:
        while rclpy.ok() and not stopping:
            rclpy.spin_once(node, timeout_sec=.2)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if native is not None:
            native.close()
        archive.write("end", {})
        archive.file.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
