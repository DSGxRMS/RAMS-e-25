"""Write the tracker capture and an offline planner report."""

import csv
from collections import Counter
from datetime import datetime
import html
import json
import math
from pathlib import Path
from tempfile import mkdtemp

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


COLOURS = ('blue', 'yellow', 'orange')
MOTION_FIELDS = (
    'odom_stamp_s', 'speed_mps', 'longitudinal_accel_mps2', 'wheel_stamp_s',
    'wheel_steering_rad', 'command_stamp_s', 'command_speed_mps',
    'command_steering_rad', 'command_accel_mps2',
)
FIELDS = (
    'sim_s', 'scan_s', 'x_m', 'y_m', 'travel_m', 'fov_origin_distance_m', 'mode', 'state', 'reason',
    'valid', 'provisional', 'startup_scan', 'ahead_m', 'curvature_1_per_m', 'shadow_outcome',
    'left_count', 'right_count', 'cached_points', 'retained_points', 'published_points',
    'shadow_candidate_points', 'shadow_added_points', 'shadow_published_points',
    'shadow_spline_published_points', 'tls_added_points', 'tls_published_points',
) + tuple(f'{sector}_{colour}_{kind}' for sector in ('fov', 'shadow')
          for colour in COLOURS for kind in ('available', 'current')) + MOTION_FIELDS


def published_curvature(path):
    """Estimate signed curvature at the start of the published centerline."""
    points = []
    for point in path:
        if not points or math.dist(points[-1], point) > 1e-6:
            points.append(point)
        if len(points) == 3:
            break
    if len(points) < 3:
        return None
    a, b, c = points
    span = math.dist(a, c)
    if span <= 1e-6:
        return None
    cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
    return 2 * cross / (math.dist(a, b) * math.dist(b, c) * span)


class PlannerReport:
    """Record every received snapshot independently of the plot history."""

    def __init__(self, root):
        root = Path(root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        prefix = 'planner-' + datetime.now().strftime('%Y%m%d-%H%M%S-')
        self.directory = Path(mkdtemp(prefix=prefix, dir=str(root)))
        self.frames = (self.directory / 'planner_frames.jsonl').open('w', encoding='utf-8')
        self.timeline = (self.directory / 'timeline.csv').open('w', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.timeline, fieldnames=FIELDS)
        self.writer.writeheader()
        self.previous_points = set()
        self.previous_car = None
        self.travel = 0.0

    def record(self, data, motion):
        snapshot = dict(data)
        snapshot['vehicle_motion'] = motion
        car = snapshot.get('car')
        if car is not None:
            if self.previous_car is not None:
                step = math.hypot(car[0] - self.previous_car[0], car[1] - self.previous_car[1])
                if step <= 2.0:
                    self.travel += step
            self.previous_car = car
        else:
            self.previous_car = None
        cache = snapshot.get('remembered_path', [])
        retained = [i for i, point in enumerate(cache) if tuple(point) in self.previous_points]
        self.previous_points = {tuple(point) for point in cache}
        scan_origin, scan_car = snapshot.get('scan_origin'), snapshot.get('scan_car')
        fov_distance = None
        if not snapshot.get('startup') and scan_origin is not None and scan_car is not None:
            fov_distance = math.dist(scan_origin, scan_car)
        curvature = published_curvature(snapshot.get('published_path', ()))
        snapshot.update(retained_cache_indices=retained, travel_m=self.travel,
                        fov_origin_distance_m=fov_distance, curvature_1_per_m=curvature)
        self.frames.write(json.dumps(snapshot, separators=(',', ':')) + '\n')

        published = Counter(snapshot.get('published_sources', []))
        x, y = car if car is not None else (None, None)
        left, right = snapshot.get('visible_counts', (None, None))
        row = dict(sim_s=snapshot['stamp'], scan_s=snapshot.get('scan_stamp'), x_m=x, y_m=y,
                   travel_m=self.travel, fov_origin_distance_m=fov_distance,
                   mode=snapshot.get('planning_mode', 'cone_memory'),
                   state=snapshot.get('state'), reason=snapshot.get('reason'), valid=int(bool(snapshot.get('valid'))),
                   provisional=int(bool(snapshot.get('provisional'))), startup_scan=int(bool(snapshot.get('startup'))),
                   ahead_m=snapshot.get('remaining_distance_m'), curvature_1_per_m=curvature,
                   shadow_outcome=snapshot.get('shadow_outcome'),
                   left_count=left, right_count=right, cached_points=len(cache), retained_points=len(retained),
                   published_points=len(snapshot.get('published_path', [])),
                   shadow_candidate_points=len(snapshot.get('shadow_candidate_path', [])),
                   shadow_added_points=len(snapshot.get('shadow_added', [])),
                   shadow_published_points=published['shadow'], shadow_spline_published_points=published['shadow_spline'],
                   tls_added_points=len(snapshot.get('tls_added', [])), tls_published_points=published['tls'])
        row.update(motion)
        for sector in ('fov', 'shadow'):
            cones = snapshot.get(f'{sector}_cones', [])
            available = Counter(cone['colour'] for cone in cones)
            current = Counter(cone['colour'] for cone in cones if cone['in_latest_map'])
            if snapshot.get('scan_stamp') is not None:
                for colour in COLOURS:
                    row[f'{sector}_{colour}_available'] = available[colour]
                    row[f'{sector}_{colour}_current'] = current[colour]
        self.writer.writerow(row)

    def finish(self, reason):
        self.frames.close()
        self.timeline.close()
        with (self.directory / 'timeline.csv').open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            return None
        write_charts(rows, self.directory / 'charts.png')
        report = self.directory / 'report.html'
        report.write_text(report_html(rows, reason), encoding='utf-8')
        return report


def write_charts(rows, target):
    figure = Figure(figsize=(12, 14), constrained_layout=True)
    FigureCanvasAgg(figure)
    time_s = [float(row['sim_s']) for row in rows]
    distance = [float(row['travel_m']) for row in rows]
    panels = (
        ('Path ahead (m)', time_s, (('Usable planner path', 'ahead_m'),)),
        ('Cones in latest map', time_s, (('FOV blue', 'fov_blue_current'), ('FOV yellow', 'fov_yellow_current'),
                                        ('Shadow blue', 'shadow_blue_current'), ('Shadow yellow', 'shadow_yellow_current'))),
        ('FOV origin distance (m)', time_s, (('Distance from car', 'fov_origin_distance_m'),)),
        ('Speed (m/s)', distance, (('Ground-truth odometry', 'speed_mps'), ('Command', 'command_speed_mps'))),
        ('Steering (rad)', distance, (('Applied wheel angle', 'wheel_steering_rad'),
                                      ('Command', 'command_steering_rad'))),
        ('Longitudinal acceleration (m/s²)', distance, (('Estimated from odometry', 'longitudinal_accel_mps2'),
                                                        ('Command', 'command_accel_mps2'))),
    )
    for index, (ylabel, domain, series) in enumerate(panels, 1):
        axis = figure.add_subplot(len(panels), 1, index)
        for label, key in series:
            values = [float(row[key]) if row[key] else math.nan for row in rows]
            if any(map(math.isfinite, values)):
                axis.plot(domain, values, label=label, linewidth=1.15)
        axis.set(xlabel='Simulation time (s)' if domain is time_s else 'Distance travelled (m)', ylabel=ylabel)
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend(loc='upper right')
        else:
            axis.text(0.5, 0.5, 'No matching telemetry', ha='center', transform=axis.transAxes)
    figure.savefig(target, dpi=115)


def report_html(rows, reason):
    scans = {row['scan_s']: row for row in rows if row['scan_s']}
    fov_distances = [float(row['fov_origin_distance_m']) for row in scans.values()
                     if row['fov_origin_distance_m']]
    fov_average = 'Unavailable'
    if fov_distances:
        fov_average = f'{math.fsum(fov_distances) / len(fov_distances):.2f} m across {len(fov_distances)} scans'
    outcomes = Counter(row['shadow_outcome'] for row in scans.values() if row['shadow_outcome'])
    shadow_frames = sum(int(row['shadow_published_points']) > 0 for row in rows)
    shadow_scans = sum(count for outcome, count in outcomes.items() if outcome != 'inactive')
    valid = sum(row['valid'] == '1' for row in rows)
    changes = []
    last = None
    for row in rows:
        state = (row['mode'], row['state'], row['reason'], row['provisional'], row['shadow_outcome'])
        if state != last:
            fields = [row[key] for key in ('sim_s', 'x_m', 'y_m', 'mode', 'state', 'reason', 'provisional',
                                            'ahead_m', 'curvature_1_per_m', 'shadow_outcome')]
            if fields[-2]:
                fields[-2] = f'{float(fields[-2]):.4f}'
            changes.append('<tr>' + ''.join(f'<td>{html.escape(value)}</td>' for value in fields) + '</tr>')
            last = state
    summary = ', '.join(f'{html.escape(label)} ({count})' for label, count in sorted(outcomes.items())
                        if label != 'inactive') or 'No active shadow scans recorded'
    return f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Cone-memory planner report</title>
<style>
body {{font: 16px system-ui; color: #182b3a; max-width: 1150px; margin: 32px auto; padding: 0 20px}}
img {{width: 100%; height: auto}} table {{border-collapse: collapse; width: 100%}}
td, th {{padding: 6px 8px; text-align: left; border-bottom: 1px solid #dce3eb}}
</style>
<h1>Cone-memory planner report</h1>
<p>{len(rows)} tracker frames from {float(rows[0]['sim_s']):.2f} to {float(rows[-1]['sim_s']):.2f} s simulation time.
{valid}/{len(rows)} frames published a valid path. Capture ended when {html.escape(reason)}.</p>
<p>{shadow_scans} distinct active shadow scans. {shadow_frames} frames published at least one
shadow point. Shadow outcomes by scan · {summary}.</p>
<p><strong>Average car-to-FOV-origin distance</strong> · {fov_average}.
Each non-startup FOV scan contributes one straight-line measurement in the map frame,
using the car position at the time of that scan.</p>
<p><a href="timeline.csv">Download timeline CSV</a> · <a href="planner_frames.jsonl">Download full planner frames</a></p>
<p>The full frames contain ordered cache and published coordinates, point sources, retained cache
indices from the previous captured frame, candidate and appended shadow and TLS paths,
and cone coordinates and colours by sector. FOV counts use the startup scan circle during startup.
The CSV distinguishes all planner-available cones from those seen in the latest map snapshot.</p>
<p>Vehicle plots use distance accumulated from sampled car positions. Applied steering comes from
wheel telemetry, speed comes from ground-truth odometry, and measured longitudinal acceleration is
estimated from successive body-frame forward velocities. Missing or stale samples leave gaps.</p>
<img src="charts.png" alt="Planner path, cone coverage, FOV origin distance, and vehicle motion">
<h2>Planner state and shadow transitions</h2>
<p>Curvature is estimated from the first three distinct published centerline points.
Positive values indicate a left turn. Negative values indicate a right turn.
The timeline includes curvature for every captured planner frame.</p>
<table><thead><tr><th>Sim s</th><th>X m</th><th>Y m</th><th>Mode</th><th>State</th>
<th>Reason</th><th>Provisional</th><th>Ahead m</th><th>Curvature (1/m)</th><th>Shadow</th></tr></thead>
<tbody>{''.join(changes)}</tbody></table>
</html>"""
