"""Local and global maps and live vehicle instruments for the cone-memory tracker."""

from collections import deque
import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon
from matplotlib.path import Path
from matplotlib.transforms import Affine2D


BACKGROUND, TEXT, MUTED = '#eef3f7', '#162b3b', '#485e72'
CONE_COLOURS = {'blue': '#216ac9', 'yellow': '#d49b13', 'orange': '#e36a22'}
PATH, PROVISIONAL = '#087d62', '#a65027'
FOV, SEARCH, SHADOW_PATH, SHADOW_JOIN = '#167f95', '#9067b1', '#7025a5', '#c62867'
CAR, REMEMBERED = '#152a3c', '#6e7986'
CAR_MARKER = Path([(0.9, 0), (-0.6, 0.52), (-0.38, 0), (-0.6, -0.52), (0.9, 0)], closed=True)
MODE_NAMES = {'straight': 'Cone memory', 'tls_turn': 'TLS turn',
              'shadow_turn': 'Shadow-only turn', 'cone_memory': 'Cone memory'}
TURN_MODES = ('tls_turn', 'shadow_turn')

PANEL, PANEL_EDGE = '#142337', '#344a62'
INSTRUMENT_TEXT, INSTRUMENT_MUTED = '#eaf3fa', '#acbfd1'
MEASURED, COMMAND = '#67e2c2', '#ffcb80'
STEER_SCALE_DEG = 20
LOCAL_MAP_WIDTH = 1.30


def xy_array(points):
    return np.asarray(points, dtype=float).reshape(-1, 2)


def set_line(line, points):
    line.set_data(*xy_array(points).T)


def map_axes(axis, radius):
    axis.set(xlabel='Map X [m]', ylabel='Map Y [m]', xlim=(-radius, radius), ylim=(-radius, radius),
             aspect='equal')
    axis.grid(color='#d6e0ea', linewidth=0.7)
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=9, colors=MUTED)
    for edge in axis.spines.values():
        edge.set_color('#98aabd')


def car_marker(axis):
    marker, = axis.plot([], [], marker=CAR_MARKER, linestyle='none', markersize=18,
                        markerfacecolor=CAR, markeredgecolor='white', markeredgewidth=1.35, zorder=10)
    return marker


def draw_car(marker, data):
    position = data.get('car')
    if position is not None:
        marker.set_marker(Affine2D().rotate(data['yaw']).transform_path(CAR_MARKER))
    set_line(marker, [position] if position is not None else [])


def sector(origin, axis, low, high, radius):
    heading = math.atan2(axis[1], axis[0])
    angles = heading + np.radians(np.linspace(low, high, 50))
    arc = origin + radius * np.column_stack((np.cos(angles), np.sin(angles)))
    return np.vstack((origin, arc, origin))


def shadow_bands(data):
    bounds = data.get('shadow_bounds_deg')
    return data.get('shadow_bands_deg', (bounds,) if bounds else ())


class GlobalMap:
    """Accumulate edges actually present in planner snapshots without bridging gaps."""

    def __init__(self, radius):
        self.radius = radius
        self.figure = plt.figure(figsize=(8.5, 7.2), facecolor=BACKGROUND)
        self.figure.canvas.manager.set_window_title('Cone-memory tracker · Global map')
        self.axis = self.figure.add_axes([0.10, 0.16, 0.84, 0.67])
        map_axes(self.axis, radius)
        self.figure.text(0.09, 0.94, 'Global bird’s-eye view', fontsize=19, weight='bold', color=TEXT)
        self.info = self.figure.text(0.09, 0.89, 'Waiting for planner data', fontsize=10, color=MUTED)
        self.layers = (
            LineCollection([], colors=REMEMBERED, linewidths=1.9, label='Remembered path', zorder=3),
            LineCollection([], colors=SHADOW_PATH, linewidths=2.5, label='Published with shadow active', zorder=4),
        )
        for layer in self.layers:
            self.axis.add_collection(layer)
        self.cones = {name: self.axis.scatter([], [], s=14, color=colour, alpha=0.50, zorder=2)
                      for name, colour in CONE_COLOURS.items()}
        self.car = car_marker(self.axis)
        self.axis.legend(handles=[*self.layers, Line2D([], [], marker=CAR_MARKER, color=CAR,
                                                      markersize=9, linestyle='none', label='Car')],
                         loc='upper center', bbox_to_anchor=(0.5, -0.11), ncol=3, fontsize=9, frameon=False)
        self.closed = False
        self.figure.canvas.mpl_connect('close_event', self.mark_closed)
        self.clear()

    def mark_closed(self, _event):
        self.closed = True

    def clear(self):
        self.edges = ({}, {})
        self.rendered_lengths = [-1, -1]
        self.bounds = None

    def include(self, points):
        if len(points):
            bounds = np.array([points.min(axis=0), points.max(axis=0)])
            if self.bounds is not None:
                bounds = np.array([np.minimum(bounds[0], self.bounds[0]), np.maximum(bounds[1], self.bounds[1])])
            self.bounds = bounds

    def add_edges(self, index, path, eligible):
        # Give near-identical edges the same display key while drawing the original coordinates.
        cells = np.rint(path / 0.05).astype(int)
        for i in np.flatnonzero(eligible):
            first, second = tuple(cells[i]), tuple(cells[i + 1])
            if first != second:
                key = (first, second) if first < second else (second, first)
                if key not in self.edges[index]:
                    self.edges[index][key] = path[i:i + 2].copy()

    def record(self, data):
        cache = xy_array(data.get('remembered_path', ()))
        confirmed = np.asarray(data.get('remembered_confirmed', ()), dtype=bool)
        sources = np.asarray(data.get('remembered_sources', ()), dtype=str)
        if len(confirmed) != len(cache):
            confirmed = np.ones(len(cache), dtype=bool)
        if len(sources) == len(cache):
            confirmed = confirmed | (sources == 'shadow_spline')
        self.add_edges(0, cache, confirmed[:-1] & confirmed[1:])
        self.include(cache[confirmed])
        if data.get('shadow_active', bool(shadow_bands(data))):
            published = xy_array(data.get('published_path', ()))
            self.add_edges(1, published, np.ones(max(len(published) - 1, 0), dtype=bool))
            self.include(published)
        if data.get('car') is not None:
            self.include(xy_array([data['car']]))

    def draw(self, data):
        for index, (layer, edges) in enumerate(zip(self.layers, self.edges)):
            if len(edges) != self.rendered_lengths[index]:
                layer.set_segments(list(edges.values()))
                self.rendered_lengths[index] = len(edges)

        cones = {name: [] for name in self.cones}
        for cone in data.get('cones', ()):
            cones[cone['colour']].append(cone['xy'])
        for name, artist in self.cones.items():
            artist.set_offsets(xy_array(cones[name]))
        if self.bounds is None:
            self.axis.set(xlim=(-self.radius, self.radius), ylim=(-self.radius, self.radius))
        else:
            visible = [self.bounds, *(xy_array(points) for points in cones.values())]
            extent = np.vstack(visible)
            low, high = extent.min(axis=0), extent.max(axis=0)
            center = (low + high) / 2
            half = max(0.5 * np.max(high - low) + 2.5, min(10.0, self.radius))
            self.axis.set(xlim=(center[0] - half, center[0] + half), ylim=(center[1] - half, center[1] + half))
        draw_car(self.car, data)
        count = data.get('remembered_run_point_count')
        label = f'{count:,} remembered points  ·  ' if count is not None else ''
        self.info.set_text(f"{label}{data['stamp']:.1f} s sim")

    def close(self):
        if not self.closed:
            self.closed = True
            plt.close(self.figure)


class MotionPanel:
    """Draw the instruments separately from the slower map refresh."""

    def __init__(self, figure, axes, history_seconds):
        self.canvas = figure.canvas
        self.history_seconds = history_seconds
        self.history = deque(maxlen=2400)
        self.values = {}
        self.stamp = None
        speed, accel, steering = axes
        for axis in axes:
            axis.set(xlim=(0, 1), ylim=(0, 1), facecolor=PANEL, xticks=[], yticks=[])
            for edge in axis.spines.values():
                edge.set_color(PANEL_EDGE)

        self.text(speed, 0.06, 0.83, 'SPEED', weight='bold')
        self.speed = self.text(speed, 0.06, 0.53, '—', size=38, number=True, colour=MEASURED)
        self.text(speed, 0.83, 0.50, 'm/s', align='right')
        self.speed_cmd = self.text(speed, 0.06, 0.17, 'Target  —', colour=INSTRUMENT_MUTED)

        self.text(accel, 0.06, 0.90, 'LONGITUDINAL ACCELERATION', size=9, weight='bold')
        self.accel_value = self.text(accel, 0.06, 0.77, 'Measured  —', size=9, colour=MEASURED)
        self.accel_cmd = self.text(accel, 0.94, 0.77, 'Command  —', size=9, colour=COMMAND, align='right')
        self.graph = accel.inset_axes([0.12, 0.20, 0.81, 0.47], facecolor=PANEL)
        self.graph.set(xlim=(-history_seconds, 0), ylim=(-4, 4), yticks=[-4, 0, 4],
                       xticks=[-history_seconds, -history_seconds / 2, 0])
        self.graph.set_xticklabels([f'−{history_seconds:g}s', f'−{history_seconds / 2:g}s', 'now'])
        self.graph.tick_params(axis='both', labelsize=8, colors=INSTRUMENT_MUTED, length=0, pad=4)
        self.graph.grid(color=PANEL_EDGE, linewidth=0.65)
        self.graph.axhline(0, color='#546a7e', linewidth=0.8)
        for edge in self.graph.spines.values():
            edge.set_color(PANEL_EDGE)
        self.acceleration, = self.graph.plot([], [], color=MEASURED, linewidth=1.5)
        self.acceleration_cmd, = self.graph.plot([], [], color=COMMAND, linewidth=1.3, linestyle='--')

        self.text(steering, 0.06, 0.87, 'STEERING COMMAND', weight='bold')
        self.steer = self.text(steering, 0.06, 0.64, '—', size=28, number=True, colour=COMMAND)
        self.wheel = self.text(steering, 0.94, 0.64, 'Wheel  —', size=9, align='right', colour=INSTRUMENT_MUTED)
        steering.plot([0.09, 0.91], [0.32, 0.32], color=PANEL_EDGE, linewidth=5, solid_capstyle='round')
        steering.plot([0.50, 0.50], [0.26, 0.38], color=INSTRUMENT_MUTED, linewidth=1)
        self.steer_bar, = steering.plot([], [], color=COMMAND, linewidth=5, solid_capstyle='round')
        for x, label, alignment in ((0.09, '−20°', 'left'), (0.50, '0°', 'center'), (0.91, '+20°', 'right')):
            self.text(steering, x, 0.13, label, size=8, align=alignment, colour=INSTRUMENT_MUTED)

        self.artists = {
            speed: (self.speed, self.speed_cmd),
            accel: (self.accel_value, self.accel_cmd, self.acceleration, self.acceleration_cmd),
            steering: (self.steer, self.wheel, self.steer_bar),
        }
        self.backgrounds = {}
        if self.canvas.supports_blit:
            for items in self.artists.values():
                for item in items:
                    item.set_animated(True)
            self.canvas.mpl_connect('draw_event', self.on_draw)

    @staticmethod
    def text(axis, x, y, text, size=10, *, colour=INSTRUMENT_TEXT, weight='normal', number=False, align='left'):
        return axis.text(x, y, text, transform=axis.transAxes, va='center', ha=align, color=colour,
                         fontsize=size, fontweight=weight, family='monospace' if number else 'sans-serif')

    def on_draw(self, _event):
        self.backgrounds = {axis: self.canvas.copy_from_bbox(axis.bbox) for axis in self.artists}
        for items in self.artists.values():
            for item in items:
                item.axes.draw_artist(item)

    def redraw(self):
        if not self.backgrounds:
            self.canvas.draw_idle()
            return
        for axis, items in self.artists.items():
            self.canvas.restore_region(self.backgrounds[axis])
            for item in items:
                item.axes.draw_artist(item)
            self.canvas.blit(axis.bbox)

    def clear(self):
        self.history.clear()
        self.stamp = None
        self.values = {}
        self.update_readings({})
        self.acceleration.set_data([], [])
        self.acceleration_cmd.set_data([], [])

    def update_readings(self, values):
        speed = values.get('speed_mps')
        target = values.get('command_speed_mps')
        measured = values.get('longitudinal_accel_mps2')
        accel = values.get('command_accel_mps2')
        steer = values.get('command_steering_rad')
        wheel = values.get('wheel_steering_rad')
        self.speed.set_text(f'{speed:05.2f}' if speed is not None else '—')
        self.speed_cmd.set_text(f'Target  {target:.2f} m/s' if target is not None else 'Target  —')
        self.accel_value.set_text(f'Measured  {measured:+.2f} m/s²' if measured is not None else 'Measured  —')
        self.accel_cmd.set_text(f'Command  {accel:+.2f} m/s²' if accel is not None else 'Command  —')
        self.steer.set_text(f'{math.degrees(steer):+05.1f}°' if steer is not None else '—')
        self.wheel.set_text(f'Wheel  {math.degrees(wheel):+.1f}°' if wheel is not None else 'Wheel  —')
        if steer is None:
            self.steer_bar.set_data([], [])
        else:
            fraction = np.clip(math.degrees(steer) / STEER_SCALE_DEG, -1, 1)
            self.steer_bar.set_data([0.50, 0.50 + 0.41 * fraction], [0.32, 0.32])

    def update(self, stamp, values):
        if stamp == self.stamp and values == self.values:
            return False
        self.stamp, self.values = stamp, values
        self.update_readings(values)
        if stamp is not None:
            if self.history and stamp < self.history[-1][0]:
                self.history.clear()
            sample = (stamp, values.get('longitudinal_accel_mps2', math.nan),
                      values.get('command_accel_mps2', math.nan))
            if self.history and stamp == self.history[-1][0]:
                self.history[-1] = sample
            else:
                self.history.append(sample)
            while self.history[0][0] < stamp - self.history_seconds:
                self.history.popleft()
            time_s, measured, command = np.asarray(self.history).T
            self.acceleration.set_data(time_s - stamp, measured)
            self.acceleration_cmd.set_data(time_s - stamp, command)
        return True

    def scale_graph(self):
        # Recompute limits with the map draw, when the grid background is refreshed too.
        values = [abs(value) for sample in self.history for value in sample[1:] if math.isfinite(value)]
        limit = max(4, math.ceil(max(values, default=0.0)))
        if self.graph.get_ylim() != (-limit, limit):
            self.graph.set(ylim=(-limit, limit), yticks=[-limit, 0, limit])
            return True
        return False


class PlannerView:
    """Keep the latest published local path and each run's global geometry."""

    def __init__(self, radius, history_seconds):
        self.radius = radius
        self.figure = plt.figure(figsize=(13, 7.8), facecolor=BACKGROUND)
        self.figure.canvas.manager.set_window_title('Cone-memory tracker · Local map and controls')
        grid = self.figure.add_gridspec(3, 2, width_ratios=(2.15, 1), height_ratios=(1, 1.55, 1.05),
                                       left=0.075, right=0.97, bottom=0.16, top=0.81, wspace=0.24, hspace=0.23)
        self.map = self.figure.add_subplot(grid[:, 0])
        map_axes(self.map, radius)
        self.map.set_xlim(-radius * LOCAL_MAP_WIDTH, radius * LOCAL_MAP_WIDTH)
        self.map.set_title('Local bird’s-eye view', loc='left', fontsize=12, weight='bold', color=TEXT, pad=11)
        self.figure.text(0.06, 0.94, 'Cone-memory tracker', fontsize=22, weight='bold', color=TEXT)
        self.state = self.figure.text(0.061, 0.893, 'Waiting for planner data', fontsize=11, color=TEXT)
        self.detail = self.figure.text(0.061, 0.858, '', fontsize=9, color=MUTED)
        self.mode = self.figure.text(0.97, 0.934, 'Waiting for planner', ha='right', va='center',
                                     fontsize=11, weight='bold', color=TEXT,
                                     bbox=dict(boxstyle='round,pad=0.45', facecolor='#dce6ef', edgecolor='none'))
        self.mode_detail = self.figure.text(0.97, 0.892, '', ha='right', fontsize=9, color=MUTED)
        self.motion = MotionPanel(self.figure, [self.figure.add_subplot(grid[i, 1]) for i in range(3)],
                                  history_seconds)
        self.overview = GlobalMap(radius)
        self.figure.canvas.mpl_connect('close_event', lambda _: self.overview.close())

        self.startup = Circle((0, 0), 0, fill=False, edgecolor=FOV, linestyle='--', visible=False)
        self.fov = Polygon([[0, 0]] * 3, facecolor='#167f9517', edgecolor=FOV, linewidth=1.0, visible=False)
        self.shadows = tuple(Polygon([[0, 0]] * 3, facecolor='#9067b127', edgecolor=SEARCH,
                                     linewidth=1.35, linestyle='--', visible=False) for _ in range(2))
        for patch in (self.startup, self.fov, *self.shadows):
            self.map.add_patch(patch)
        self.ab, = self.map.plot([], [], color=FOV, linewidth=1.15)
        self.cd, = self.map.plot([], [], color=FOV, linewidth=1.0, linestyle=':')
        self.ab_label = self.map.text(0, 0, 'AB', color=FOV, fontsize=8, visible=False)
        self.cd_label = self.map.text(0, 0, 'CD', color=FOV, fontsize=8, visible=False)
        self.cones = {}
        for name, colour in CONE_COLOURS.items():
            self.cones[name, True] = self.map.scatter([], [], s=30, facecolors=colour, edgecolors=CAR,
                                                      linewidths=0.45, zorder=4)
            self.cones[name, False] = self.map.scatter([], [], s=26, facecolors='none', edgecolors=colour,
                                                       linewidths=1.0, zorder=3)
        self.reference, = self.map.plot([], [], color=PATH, linewidth=2.1, marker='o', markersize=3, zorder=5)
        self.shadow_path = LineCollection([], colors=SHADOW_PATH, linewidths=2.3, zorder=6)
        self.shadow_join = LineCollection([], colors=SHADOW_JOIN, linewidths=2.3, zorder=6)
        for layer in (self.shadow_path, self.shadow_join):
            self.map.add_collection(layer)
        self.shadow_samples = self.map.scatter([], [], marker='D', s=30, c=SHADOW_PATH,
                                               edgecolors='white', linewidths=0.75, zorder=7)
        self.shadow_join_samples = self.map.scatter([], [], s=25, c=SHADOW_JOIN, edgecolors='white',
                                                    linewidths=0.75, zorder=7)
        self.car = car_marker(self.map)

        self.map.legend(handles=[
            Line2D([], [], color=PATH, marker='o', markersize=3, label='Published path'),
            Line2D([], [], color=PROVISIONAL, linestyle='--', label='Provisional path'),
            Line2D([], [], color=SHADOW_PATH, marker='D', markersize=4, label='Published shadow'),
            Line2D([], [], color=SHADOW_JOIN, marker='o', markersize=4, label='Published spline'),
            Line2D([], [], marker=CAR_MARKER, color=CAR, markersize=9, linestyle='none', label='Car'),
            *(Line2D([], [], marker='o', color='none', markerfacecolor=colour, markeredgecolor=CAR,
                     markeredgewidth=0.45, label=f'{name.title()} cone') for name, colour in CONE_COLOURS.items()),
        ], loc='upper center', bbox_to_anchor=(0.5, -0.11), ncol=4, fontsize=8, frameon=False,
            columnspacing=1.0, handletextpad=0.55)
        self.active_mode = None
        self.last_switch = None

    def reset_run(self):
        self.overview.clear()
        self.motion.clear()
        self.active_mode = self.last_switch = None

    def record(self, data):
        if not self.overview.closed:
            self.overview.record(data)
        if data.get('reason') == 'pose_unavailable':
            return
        mode = data.get('planning_mode', 'cone_memory')
        if any({mode, self.active_mode} == {'straight', turn} for turn in TURN_MODES):
            self.last_switch = (data['stamp'], self.active_mode, mode)
        self.active_mode = mode

    def draw_published(self, data):
        path = xy_array(data.get('published_path', ()))
        set_line(self.reference, path)
        provisional = data.get('provisional')
        self.reference.set(color=PROVISIONAL if provisional else PATH, linestyle='--' if provisional else '-')
        sources = np.asarray(data.get('published_sources', ()), dtype=str)
        if len(sources) != len(path):
            sources = np.repeat('', len(path))
        shadow = sources == 'shadow'
        spline = sources == 'shadow_spline'
        legs = np.stack((path[:-1], path[1:]), axis=1)
        joins = spline[:-1] | spline[1:]
        self.shadow_join.set_segments(legs[joins])
        self.shadow_path.set_segments(legs[shadow[1:] & ~joins])
        self.shadow_join_samples.set_offsets(path[spline])
        self.shadow_samples.set_offsets(path[shadow])

    def draw(self, data):
        car = data.get('car')
        if car is not None:
            x, y = car
            half_width = self.radius * LOCAL_MAP_WIDTH
            self.map.set(xlim=(x - half_width, x + half_width), ylim=(y - self.radius, y + self.radius))
        draw_car(self.car, data)
        cones = {key: [] for key in self.cones}
        for cone in data.get('cones', ()):
            cones[cone['colour'], cone['in_latest_map']].append(cone['xy'])
        for key, artist in self.cones.items():
            artist.set_offsets(xy_array(cones[key]))
        self.draw_published(data)
        self.draw_scan(data)
        self.draw_status(data)
        if not self.overview.closed:
            self.overview.draw(data)
            self.overview.figure.canvas.draw_idle()
        self.motion.scale_graph()
        self.figure.canvas.draw_idle()

    def draw_status(self, data):
        phase = data['state'].replace('_', ' ').title()
        if data.get('startup') or data.get('startup_pending'):
            phase = f'Startup scan  ·  {phase}'
        if data.get('provisional'):
            phase += '  ·  Provisional'
        self.state.set_text(f"{phase}  ·  {len(data.get('published_path', ()))} published points")
        bands = shadow_bands(data)
        detail = 'Open cones mark earlier observations'
        if data.get('reason') == 'pose_unavailable':
            detail = 'Waiting for a fresh car pose'
        elif data.get('startup'):
            detail = f"Startup scan within {data['radius_m']:.0f} m of the car"
        elif bands:
            if len(bands) == 2:
                description = 'Both shadow sectors'
            else:
                low, high = bands[0]
                side = 'Left' if low > 0 else 'Right'
                description = f'{side} shadow {low:+.0f}° to {high:+.0f}°'
            outcome = 'joined' if data.get('shadow_selected') else 'searching'
            detail = f'{description}  ·  {outcome}'
        self.detail.set_text(detail)
        mode = None if data.get('reason') == 'pose_unavailable' else self.active_mode
        self.mode.set_text(MODE_NAMES.get(mode, 'Waiting for planner'))
        self.mode.get_bbox_patch().set_facecolor('#e7dcf1' if self.active_mode in TURN_MODES else '#dce6ef')
        if self.last_switch is not None:
            stamp, previous, current = self.last_switch
            self.mode_detail.set_text(f'{MODE_NAMES[previous]} → {MODE_NAMES[current]} at {stamp:.2f} s sim')
        else:
            self.mode_detail.set_text('')

    def mark_waiting(self):
        self.detail.set_text('Waiting for planner updates')
        self.mode.set_text('Waiting for planner')
        self.mode.get_bbox_patch().set_facecolor('#dce6ef')
        self.figure.canvas.draw_idle()

    def draw_scan(self, data):
        has_scan = 'scan_origin' in data
        startup = has_scan and data.get('startup')
        active = has_scan and not startup
        self.startup.set_visible(bool(startup))
        self.fov.set_visible(active)
        for patch in self.shadows:
            patch.set_visible(False)
        self.ab_label.set_visible(active)
        self.cd_label.set_visible(active)
        if not active:
            set_line(self.ab, [])
            set_line(self.cd, [])
            if startup:
                self.startup.set(center=data['scan_car'], radius=data['radius_m'])
            return
        origin = np.asarray(data['scan_origin'])
        axis = np.asarray(data['scan_axis'])
        radius, half = data['radius_m'], data['fov_angle_deg'] / 2
        self.fov.set_xy(sector(origin, axis, -half, half, radius))
        for patch, (low, high) in zip(self.shadows, shadow_bands(data)):
            patch.set(xy=sector(origin, axis, low, high, radius), visible=True,
                      linestyle='-' if data.get('shadow_selected') else '--')
        normal = np.array([-axis[1], axis[0]])
        a = origin + axis * radius * 0.35
        c = origin + normal * min(3.0, radius * 0.2)
        set_line(self.ab, [origin, a])
        set_line(self.cd, [2 * origin - c, c])
        self.ab_label.set_position(a)
        self.cd_label.set_position(c)

    def close(self):
        self.overview.close()
        plt.close(self.figure)
