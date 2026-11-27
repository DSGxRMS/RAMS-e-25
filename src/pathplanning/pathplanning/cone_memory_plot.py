"""Plot cone-memory planning as it happens, in the planner's map frame."""

from collections import deque
import json
import math
import time

import matplotlib
from ackermann_msgs.msg import AckermannDriveStamped
from eufs_msgs.msg import WheelSpeedsStamped
from matplotlib import pyplot as plt
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .cone_memory_display import PlannerView
from .cone_memory_report import PlannerReport


MOTION_AGE_S = 0.35
MOTION_RECEIPT_TIMEOUT_S = 1.0


def message_stamp(message):
    stamp = message.header.stamp
    return stamp.sec + stamp.nanosec * 1e-9


class LivePlannerPlot(Node):
    """Receive planner telemetry without sharing the planning executor."""

    def __init__(self):
        super().__init__('cone_memory_plot')
        self.declare_parameter('topics.debug_in', '/pathplanning/cone_memory_debug')
        self.declare_parameter('plot.radius_m', 20.0)
        self.declare_parameter('plot.history_s', 20.0)
        self.declare_parameter('plot.map_hz', 2.0)
        self.declare_parameter('report.output_dir', 'runs')
        self.declare_parameter('topics.odom_in', '/ground_truth/odom')
        self.declare_parameter('topics.wheel_in', '/ground_truth/wheel_speeds')
        self.declare_parameter('topics.command_in', '/cmd')
        radius = self.get_parameter('plot.radius_m').value
        history = self.get_parameter('plot.history_s').value
        map_hz = self.get_parameter('plot.map_hz').value
        if not all(math.isfinite(value) and value > 0 for value in (radius, history, map_hz)):
            raise ValueError('plot.radius_m, plot.history_s, and plot.map_hz must be positive finite values')
        self.view = PlannerView(radius, history)
        self.map_period = 1.0 / map_hz
        self.last_map_refresh = -math.inf
        self.latest = None
        self.last_message = None
        self.rendered = None
        self.report = None
        self.last_capture_stamp = None
        self.publisher_missing_since = None
        self.motion = {source: deque(maxlen=200) for source in ('odom', 'wheel', 'command')}
        self.motion_received_at = {}
        self.previous_velocity = None
        self.longitudinal_accel = None
        self.debug_topic = self.get_parameter('topics.debug_in').value
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(String, self.debug_topic, self.on_debug, qos)
        for message_type, name, callback in (
                (Odometry, 'odom', self.on_odom),
                (WheelSpeedsStamped, 'wheel', self.on_wheel),
                (AckermannDriveStamped, 'command', self.on_command)):
            self.create_subscription(message_type, self.get_parameter(f'topics.{name}_in').value, callback, qos)

    def record_motion(self, source, stamp, **values):
        history = self.motion[source]
        if history and stamp < history[-1][0]:
            history.clear()
        values = {name: value for name, value in values.items() if value is not None and math.isfinite(value)}
        if values:
            history.append((stamp, values))
            self.motion_received_at[source] = time.monotonic()

    def on_odom(self, message):
        stamp = message_stamp(message)
        velocity = message.twist.twist.linear
        if math.isfinite(velocity.x):
            if self.previous_velocity is None:
                self.previous_velocity = (stamp, velocity.x)
            else:
                previous_stamp, previous_x = self.previous_velocity
                dt = stamp - previous_stamp
                if dt < 0 or dt > MOTION_AGE_S:
                    self.longitudinal_accel = None
                    self.previous_velocity = (stamp, velocity.x)
                elif dt >= 0.005:
                    self.longitudinal_accel = (velocity.x - previous_x) / dt
                    self.previous_velocity = (stamp, velocity.x)
        self.record_motion('odom', stamp, speed_mps=math.hypot(velocity.x, velocity.y),
                           longitudinal_accel_mps2=self.longitudinal_accel)

    def on_wheel(self, message):
        self.record_motion('wheel', message_stamp(message), wheel_steering_rad=message.speeds.steering)

    def on_command(self, message):
        self.record_motion('command', message_stamp(message), command_speed_mps=message.drive.speed,
                           command_steering_rad=message.drive.steering_angle,
                           command_accel_mps2=message.drive.acceleration)

    def live_motion(self, now):
        samples = [history[-1] for source, history in self.motion.items()
                   if history and now - self.motion_received_at[source] <= MOTION_RECEIPT_TIMEOUT_S]
        stamp = max((stamp for stamp, _ in samples), default=None)
        values = {name: value for source_stamp, readings in samples if stamp - source_stamp <= MOTION_AGE_S
                  for name, value in readings.items()}
        return stamp, values

    def capture(self, data):
        stamp = data['stamp']
        if self.last_capture_stamp is not None and stamp < self.last_capture_stamp:
            self.finish_report('simulation time reset')
        if self.report is None:
            self.report = PlannerReport(self.get_parameter('report.output_dir').value)
            self.get_logger().info(f'Recording planner report in {self.report.directory}')
        matched = {}
        for source, history in self.motion.items():
            for sample_stamp, values in reversed(history):
                if sample_stamp <= stamp:
                    if stamp - sample_stamp <= MOTION_AGE_S:
                        matched.update(values)
                        matched[f'{source}_stamp_s'] = sample_stamp
                    break
        self.report.record(data, matched)
        self.last_capture_stamp = stamp

    def finish_report(self, reason):
        if self.report is None:
            return
        report, self.report = self.report, None
        self.last_capture_stamp = None
        self.publisher_missing_since = None
        path = report.finish(reason)
        if path:
            print(f'[planner report] {path}', flush=True)

    def on_debug(self, message):
        try:
            data = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if (isinstance(data, dict) and data.get('frame') == 'map'
                and isinstance(data.get('stamp'), (float, int)) and math.isfinite(data['stamp'])):
            if self.report is None or data['stamp'] < self.last_capture_stamp:
                self.view.reset_run()
            self.latest = data
            self.last_message = time.monotonic()
            self.publisher_missing_since = None
            self.view.record(data)
            self.capture(data)

    def refresh(self):
        if not rclpy.ok():
            self.view.close()
            return
        # Drain command, wheel, odometry, and debug callbacks within a GUI time budget.
        deadline = time.monotonic() + 0.025
        for _ in range(64):
            rclpy.spin_once(self, timeout_sec=0.0)
            if time.monotonic() >= deadline:
                break
        now = time.monotonic()
        if self.report is not None and now - self.last_message >= 1.0:
            if self.count_publishers(self.debug_topic) == 0:
                if self.publisher_missing_since is None:
                    self.publisher_missing_since = now
                elif now - self.publisher_missing_since >= 0.75:
                    self.finish_report('the planner publisher disconnected')
            else:
                self.publisher_missing_since = None
        motion_updated = self.view.motion.update(*self.live_motion(now))
        if now - self.last_map_refresh >= self.map_period:
            self.last_map_refresh = now
            if self.latest is not None and self.latest is not self.rendered:
                self.view.draw(self.latest)
                self.rendered = self.latest
                return
            if (self.last_message is not None and now - self.last_message > 1.0
                    and self.view.detail.get_text() != 'Waiting for planner updates'):
                self.view.mark_waiting()
                return
            if self.view.motion.scale_graph():
                self.view.figure.canvas.draw_idle()
                return
        if motion_updated:
            self.view.motion.redraw()


def main(args=None):
    """Run the live plot on the GUI thread."""
    if matplotlib.get_backend().lower() == 'agg':
        raise SystemExit('The planner plot needs a display and a Matplotlib GUI backend such as TkAgg.')
    rclpy.init(args=args)
    node = None
    try:
        node = LivePlannerPlot()
        timer = node.view.figure.canvas.new_timer(interval=100)
        timer.add_callback(node.refresh)
        timer.start()
        try:
            plt.show()
        finally:
            timer.stop()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.finish_report('the tracker closed')
            finally:
                node.view.close()
                node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
