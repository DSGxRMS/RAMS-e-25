"""ROS adapter for the cone-memory planner and the existing path interface."""

from dataclasses import fields
import json
import math

from eufs_msgs.msg import ConeArrayWithCovariance
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .cone_memory import ConeMemoryPlanner, Config


CONE_COLOURS = ('blue', 'yellow', 'orange')


def seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def xy(cones):
    return [(cone.point.x, cone.point.y) for cone in cones]


class ConeMemoryNode(Node):
    """Publish ordered map-frame samples and explicit planner status."""

    def __init__(self, planner_type=ConeMemoryPlanner, config_type=Config, node_name='pp_cone_memory',
                 *, timer_period=0.05):
        super().__init__(node_name)
        self.declare_parameter('topics.odom_in', '/slam/odom_raw')
        self.declare_parameter('topics.map_in', '/slam/map_cones')
        self.declare_parameter('topics.path_out', '/path_points')
        self.declare_parameter('topics.status_out', '/pathplanning/cone_memory_status')
        self.declare_parameter('topics.debug_out', '/pathplanning/cone_memory_debug')
        self.declare_parameter('map.observation_snapshots', False)
        self.declare_parameter('pose.max_age', 0.35)
        defaults = config_type()
        params = {}
        for field in fields(defaults):
            name = f'cone_memory.{field.name}'
            self.declare_parameter(name, getattr(defaults, field.name))
            params[field.name] = self.get_parameter(name).value
        self.planner = planner_type(config_type(**params))
        self.pose = None
        self.last_tick = None
        self.last_debug = None
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.path_pub = self.create_publisher(Path, self.get_parameter('topics.path_out').value, qos)
        self.status_pub = self.create_publisher(String, self.get_parameter('topics.status_out').value, 5)
        debug_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.debug_pub = self.create_publisher(String, self.get_parameter('topics.debug_out').value, debug_qos)
        self.odom_sub = self.create_subscription(Odometry, self.get_parameter('topics.odom_in').value,
                                                 self.on_odom, qos)
        self.map_sub = self.create_subscription(ConeArrayWithCovariance,
                                                self.get_parameter('topics.map_in').value, self.on_map, qos)
        self.timer = self.create_timer(timer_period, self.tick)

    def on_odom(self, msg):
        """Accept ego poses in the same frame as the supported integration map."""
        if msg.header.frame_id != 'map':
            return
        stamp = seconds(msg.header.stamp)
        pose = msg.pose.pose
        q = pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
        car = (pose.position.x, pose.position.y)
        if all(map(math.isfinite, (*car, yaw))):
            self.pose = (car, yaw, stamp)

    def on_map(self, msg):
        """Feed the received map evidence into the planner."""
        if msg.header.frame_id != 'map':
            return
        stamp = seconds(msg.header.stamp)
        snapshots = self.get_parameter('map.observation_snapshots').value
        self.planner.observe(xy(msg.blue_cones), xy(msg.yellow_cones), stamp,
                             observation_snapshot=snapshots,
                             orange=xy(msg.orange_cones) + xy(msg.big_orange_cones))

    def tick(self):
        """Report the current geometry and validity on the simulation clock."""
        tick = self.get_clock().now()
        now = tick.nanoseconds * 1e-9
        if self.last_tick is not None and now < self.last_tick:
            self.planner.reset()
            self.pose = None
            self.last_debug = None
        self.last_tick = now
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = tick.to_msg()
        status = {'stamp': now, 'state': 'no_valid_path', 'valid': False, 'provisional': False,
                  'remaining_distance_m': 0.0, 'endpoint': None, 'shadow_active': False}
        max_age = self.get_parameter('pose.max_age').value
        result = None
        if self.pose is not None and -0.10 <= now - self.pose[2] <= max_age:
            car, yaw, _ = self.pose
            result = self.planner.plan(car, yaw, now)
            for index, (x, y) in enumerate(result.points):
                pose = PoseStamped()
                pose.header = path.header
                pose.pose.position.x = float(x)
                pose.pose.position.y = float(y)
                start = max(0, index - 1)
                dx, dy = result.points[start + 1] - result.points[start]
                angle = math.atan2(dy, dx)
                pose.pose.orientation.z = math.sin(angle / 2)
                pose.pose.orientation.w = math.cos(angle / 2)
                path.poses.append(pose)
            status.update(state=result.state, valid=bool(len(path.poses)), provisional=result.provisional,
                          remaining_distance_m=result.remaining, visible_counts=result.cone_counts,
                          third_cones=result.anchors)
            if result.planning_mode is not None:
                status['planning_mode'] = result.planning_mode
            if result.probe is not None:
                origin, direction = result.probe
                status.update(probe_origin=origin.tolist(), ab_direction=direction.tolist(),
                              cd_direction=[float(-direction[1]), float(direction[0])])
                scan = self.planner.scan
                status['shadow_active'] = scan is not None and bool(scan.shadow_bands_deg)
            if path.poses:
                status['endpoint'] = result.points[-1].tolist()
            elif self.planner.cache is None:
                status['reason'] = 'no_usable_cache'
            else:
                status['reason'] = 'path_exhausted'
        else:
            status['reason'] = 'pose_unavailable'
        status['remembered_run_point_count'] = len(self.planner.remembered_path_points)
        self.path_pub.publish(path)
        message = String()
        message.data = json.dumps(status)
        self.status_pub.publish(message)
        if self.debug_pub.get_subscription_count() and (self.last_debug is None or now - self.last_debug >= 0.095):
            self.publish_debug(status, result)
            self.last_debug = now

    def publish_debug(self, status, result):
        """Send one map-frame view of what the planner evaluated and published."""
        planner = self.planner
        cfg = planner.cfg
        scan = planner.scan if result is not None else None
        snapshot = dict(status, frame='map', fov_angle_deg=cfg.fov_angle, radius_m=cfg.horizon,
                        startup_pending=not planner.startup_scan_complete,
                        cones=[], remembered_path=[], candidate_path=[], published_path=[], walls=[],
                        remembered_sources=[], published_sources=[], remembered_confirmed=[],
                        remembered_seen_s=[])
        if result is not None:
            car, yaw, _ = self.pose
            now = status['stamp']
            map_recent = planner.last_map_stamp is not None and now - planner.last_map_stamp <= 0.4
            snapshot.update(car=car, yaw=yaw, published_path=result.points.tolist())
            for key, cone in planner.cones.items():
                age = now - cone.seen
                if age <= cfg.max_age and math.hypot(cone.point[0] - car[0], cone.point[1] - car[1]) <= cfg.horizon + 8:
                    snapshot['cones'].append({'xy': cone.point.tolist(), 'colour': CONE_COLOURS[cone.colour],
                                              'in_latest_map': map_recent and key in planner.visible})
            cache = planner.cache
            if cache is not None:
                snapshot.update(remembered_path=cache.center.tolist(), remembered_sources=cache.sources.tolist(),
                                remembered_confirmed=cache.confirmed.tolist(), remembered_seen_s=cache.seen.tolist(),
                                cache_progress_m=planner.progress)
                if len(result.points):
                    sources = result.point_sources
                    if sources is None:
                        sources = cache.sources[-len(result.points):]
                    snapshot['published_sources'] = sources.tolist()
        if scan is not None:
            origin, direction = scan.probe
            bands = scan.shadow_bands_deg
            snapshot.update(scan_stamp=scan.stamp, scan_origin=origin.tolist(), scan_axis=direction.tolist(),
                            scan_car=scan.car.tolist(), startup=scan.startup, tested_yaws=scan.tested_yaws,
                            selected_yaw=scan.selected_yaw, shadow_bands_deg=bands,
                            shadow_bounds_deg=bands[0] if len(bands) == 1 else None,
                            shadow_selected=scan.shadow_selected, walls=[wall.tolist() for wall in scan.walls],
                            shadow_outcome=('selected' if scan.shadow_selected and scan.shadow_outcome != 'merged'
                                            else scan.shadow_outcome))
            for name, cones in (('fov_cones', scan.nominal_cones), ('shadow_cones', scan.shadow_cones)):
                snapshot[name] = [
                    {'xy': cone.point.tolist(), 'colour': CONE_COLOURS[cone.colour],
                     'in_latest_map': available, 'last_seen_s': cone.seen}
                    for cone, available in cones
                ]
            if scan.candidate is not None:
                snapshot['candidate_path'] = scan.candidate.tolist()
            if scan.shadow_candidate is not None:
                snapshot['shadow_candidate_path'] = scan.shadow_candidate.tolist()
            for name in ('shadow_added', 'tls_candidate', 'tls_added'):
                points = getattr(scan, name)
                if points is not None:
                    snapshot[name] = points.tolist()
        message = String()
        message.data = json.dumps(snapshot)
        self.debug_pub.publish(message)


def main(args=None):
    """Run the selected planner as a ROS 2 executable."""
    run_planner(ConeMemoryNode, args)


def run_planner(make_node, args=None):
    """Run a planner adapter with the common ROS lifecycle."""
    rclpy.init(args=args)
    node = None
    try:
        node = make_node()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
