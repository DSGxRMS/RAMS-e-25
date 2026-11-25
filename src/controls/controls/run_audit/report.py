#!/usr/bin/env python3
"""Build a standalone run report and CSV evidence from capture.py output."""
import argparse
import bisect
import collections
import csv
import json
import math
import re
from pathlib import Path


def percentile(values, q):
    ordered = sorted(v for v in values if v is not None and math.isfinite(v))
    if not ordered:
        return None
    at = (len(ordered) - 1) * q
    lo = math.floor(at)
    return ordered[lo] + (ordered[math.ceil(at)] - ordered[lo]) * (at - lo)


def metrics(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return dict(n=len(values), median=percentile(values, .5), p95=percentile(values, .95),
                max=max(values, default=None),
                rms=math.sqrt(sum(v * v for v in values) / len(values)) if values else None)


def excursions(rows, field, threshold, absolute=False):
    episodes = []
    active = None
    for row in rows:
        value = row[field]
        magnitude = (abs(value) if absolute else value) if value is not None else None
        if magnitude is None or magnitude < threshold:
            active = None
            continue
        if active is None or row["t"] - active["end"] > .75:
            active = dict(start=row["t"], end=row["t"], peak=magnitude, peak_t=row["t"],
                          x=row["x"], y=row["y"], mode=row["mode"])
            episodes.append(active)
        active["end"] = row["t"]
        if magnitude > active["peak"]:
            active.update(peak=magnitude, peak_t=row["t"], x=row["x"], y=row["y"], mode=row["mode"])
    return episodes


def points(raw):
    return [p for p in raw if len(p) >= 2 and p[0] is not None and p[1] is not None]


def display_points(raw):
    return [[round(p[0], 4), round(p[1], 4)] for p in points(raw)]


def path_length(raw):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(raw, raw[1:]))


def distance_to_path(position, line, signed=False):
    if position is None or len(line) < 2:
        return None
    best_distance, side = math.inf, 1
    x, y = position
    for a, b in zip(line, line[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length_squared = dx * dx + dy * dy
        if length_squared == 0:
            continue
        fraction = min(1, max(0, ((x - a[0]) * dx + (y - a[1]) * dy) / length_squared))
        distance = math.hypot(x - a[0] - fraction * dx, y - a[1] - fraction * dy)
        if distance < best_distance:
            best_distance = distance
            side = -1 if dx * (y - a[1]) - dy * (x - a[0]) < 0 else 1
    return best_distance * (side if signed else 1) if math.isfinite(best_distance) else None


def load_archive(path):
    streams = collections.defaultdict(list)
    bad = 0
    with path.open(encoding="utf-8") as source:
        for line in source:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            streams[row["kind"]].append(row)
    return streams, bad


class Lookup:
    def __init__(self, rows):
        self.rows = rows
        self.times = [row["t"] for row in rows]

    def get(self, t, max_age=math.inf):
        i = bisect.bisect_right(self.times, t) - 1
        if i >= 0 and t - self.times[i] <= max_age:
            return self.rows[i]
        return {}

    def nearest(self, t, max_delta):
        i = bisect.bisect_left(self.times, t)
        candidates = self.rows[max(0, i - 1):i + 1]
        row = min(candidates, key=lambda item: abs(item["t"] - t), default={})
        return row if row and abs(row["t"] - t) <= max_delta else {}


def pp_dropouts(rows, start, end, age, minimum):
    """Intervals in receiver time without a fresh geometrically usable raw PP frame."""
    gaps = []
    covered_until = start
    for i, row in enumerate(rows):
        t = row["t"]
        if t >= end:
            break
        if len(points(row["data"]["points"])) < minimum:
            continue
        next_time = rows[i + 1]["t"] if i + 1 < len(rows) else end
        valid_end = min(t + age, next_time, end)
        if valid_end <= start:
            continue
        if t > covered_until:
            gaps.append((covered_until, t))
        covered_until = max(covered_until, valid_end)
    if end > covered_until:
        gaps.append((covered_until, end))
    return [dict(start=a, end=b, duration=b - a) for a, b in gaps if b > a]


def collision_entity(name):
    parts = name.split("::")
    if len(parts) >= 3:
        parts.pop()
    while len(parts) > 1 and re.fullmatch(r"(?:link|collision|collision_\d+)", parts[-1], re.I):
        parts.pop()
    return "::".join(parts)


def contact_episodes(streams, meta, separation):
    vehicle = re.compile(meta.get("vehicle_pattern", r"(^|::|/)eufs(::|/|$)"), re.I)
    cone = re.compile(meta.get("cone_pattern", "cone"), re.I)
    active = {}
    events = []
    rows = sorted(streams["contact_native"] + streams["contact_ros"], key=lambda x: x["t"])
    for row in rows:
        for pair in row["data"]["pairs"]:
            for car_name, cone_name in (pair, pair[::-1]):
                if not (vehicle.search(car_name) and cone.search(cone_name)):
                    continue
                entity = collision_entity(cone_name)
                t = row["t"]
                event = active.get(entity)
                if event is None or t - event["end"] > separation:
                    event = dict(t=t, end=t, cone=entity, vehicle_collision=car_name,
                                 cone_collision=cone_name, samples=0,
                                 sources=set(), sim_ns=row.get("sim_ns"))
                    active[entity] = event
                    events.append(event)
                event["end"] = t
                event["samples"] += 1
                event["sources"].add(row["kind"])
                break
    for event in events:
        event["sources"] = "+".join(sorted(event["sources"]))
    return events


def read_reference(path):
    if path is None:
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [[float(row["x"]), float(row["y"])] for row in csv.DictReader(handle)]


def read_mppi(path):
    if not path.exists():
        return Lookup([])
    with path.open(encoding="utf-8", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            try:
                data = {key: float(value) for key, value in row.items()}
                if not all(math.isfinite(value) for value in data.values()):
                    continue
            except (ValueError, TypeError):
                continue
            rows.append(dict(t=data["time"], data=data))
    rows.sort(key=lambda row: row["t"])
    return Lookup(rows)


def save_csv(path, rows, columns=None):
    with path.open("w", newline="", encoding="utf-8") as output:
        fields = columns or list(dict.fromkeys(key for row in rows for key in row))
        if fields:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def analyze(args):
    streams, invalid_rows = load_archive(args.run / "events.jsonl")
    meta = next((r["data"] for r in streams["meta"]), {})
    lookup = {name: Lookup(streams[name]) for name in
              ("gt", "pp", "buffer", "pseudo", "rollouts", "cmd", "map_cones", "odom_raw", "odom_slam")}
    legacy_csv = read_mppi(args.run / "mppi.csv")
    reference = read_reference(args.reference)
    contacts = contact_episodes(streams, meta, args.contact_gap)
    commands = streams["cmd"]
    basis = streams["cycle"] or commands or streams["gt"]
    if not basis:
        raise SystemExit("Capture has no controller cycles, /cmd messages or ground-truth odometry.")
    start, end = basis[0]["t"], basis[-1]["t"]
    intervals = pp_dropouts(streams["pp"], start, end, args.pp_age, args.min_pp_points)
    anchor = next((r["sim_ns"] for r in basis if r.get("sim_ns") is not None), None)
    frames = []
    timeline = []
    geometry_closest = None
    gt_frames = []
    distance = 0.0
    last = None
    last_stamp = None
    for record in streams["gt"]:
        data = record["data"]
        held = lookup["cmd"].get(record["t"])
        command = held.get("data", {})
        position = points([data["xy"]])
        if position:
            stamp = record.get("stamp_ns") or record.get("sim_ns")
            if last and (stamp is None or last_stamp is None or stamp >= last_stamp):
                distance += math.hypot(position[0][0] - last[0], position[0][1] - last[1])
            last, last_stamp = position[0], stamp
        gt_frames.append(dict(t=record["t"], sim_ns=record.get("sim_ns"), stamp_ns=record.get("stamp_ns"),
                              x=data["xy"][0], y=data["xy"][1], yaw=data["yaw"],
                              vx=data["vx"], vy=data["vy"], yaw_rate=data["yaw_rate"],
                              distance=distance, target_speed=command.get("speed"),
                              acceleration_command=command.get("acceleration"),
                              steering_command=command.get("steering_angle"),
                              command_age_s=record["t"] - held["t"] if held else None))
    for row in basis:
        t, cycle = row["t"], row["data"] if row["kind"] == "cycle" else {}
        legacy = (legacy_csv.get(row["wall_ns"] / 1e9 + .005, .25).get("data", {})
                  if row["kind"] == "cmd" else {})
        actual = lookup["gt"].get(t, .5)
        gt = actual.get("data", {})
        cmd = cycle.get("command") or (row["data"] if row["kind"] == "cmd" else {})
        pp = lookup["pp"].get(t)
        path_raw = points(pp.get("data", {}).get("points", []))
        pp_age = t - pp["t"] if pp else None
        pp_usable = pp_age is not None and pp_age <= args.pp_age and len(path_raw) >= args.min_pp_points
        buffered = cycle.get("buffer", lookup["buffer"].get(t, .5).get("data", {}).get("points", []))
        pseudo = lookup["pseudo"].get(t, 1.5).get("data", {}).get("points", [])
        if row["kind"] == "cycle":
            rollouts = cycle.get("rollouts", [])
        else:
            rollouts = lookup["rollouts"].get(t, .35).get("data", {}).get("paths", [])
        desired = cycle.get("reference", [])
        path_spline = cycle.get("spline", [])
        actual_xy = next(iter(points([gt.get("xy", [])])), None)
        vx, target = gt.get("vx"), cmd.get("speed")
        previous = lookup["gt"].get(t - .15, .3)
        stamps = (actual.get("stamp_ns"), previous.get("stamp_ns"))
        previous_vx = previous.get("data", {}).get("vx")
        actual_accel = None
        if vx is not None and previous_vx is not None and stamps[0] and stamps[1] and stamps[0] > stamps[1]:
            actual_accel = (vx - previous_vx) * 1e9 / (stamps[0] - stamps[1])
        if vx is None or target is None:
            speed_error = None
        else:
            speed_error = vx - target
        cone_row = lookup["map_cones"].get(t, 1.0)
        cone_colours = cone_row.get("data", {})
        cone_points = points([p for colour in cone_colours.values() for p in colour])
        closest = (min((math.hypot(actual_xy[0] - p[0], actual_xy[1] - p[1]) for p in cone_points),
                       default=None) if actual_xy and cone_row.get("frame") in ("map", "world") else None)
        if closest is not None:
            geometry_closest = closest if geometry_closest is None else min(geometry_closest, closest)
        raw = lookup["odom_raw"].get(t, .5).get("data", {}).get("xy")
        slam = lookup["odom_slam"].get(t, .5).get("data", {}).get("xy")
        split = (math.hypot(raw[0] - slam[0], raw[1] - slam[1])
                 if raw is not None and slam is not None and len(points([raw, slam])) == 2 else None)
        best = points(rollouts[0]) if rollouts else []
        ref = points(desired)
        best_tracking_rms = metrics([math.hypot(p[0] - q[0], p[1] - q[1])
                                     for p, q in zip(best, ref)])["rms"]
        sim_ns = cycle.get("sim_ns", row.get("sim_ns"))
        sample = dict(t=t, sim_s=(sim_ns - anchor) / 1e9 if sim_ns is not None and anchor is not None else None,
                      mode=cycle.get("mode"), model=cycle.get("model"),
                      x=actual_xy[0] if actual_xy else None, y=actual_xy[1] if actual_xy else None,
                      vx=vx, vy=gt.get("vy"), yaw=gt.get("yaw"), yaw_rate=gt.get("yaw_rate"),
                      ground_truth_age_s=t - actual["t"] if actual else None,
                      target_speed=target, speed_error=speed_error,
                      steering=cmd.get("steering_angle"), accel=cmd.get("acceleration"),
                      actual_accel_mps2=actual_accel,
                      pp_age_s=pp_age, pp_points=len(path_raw) if pp else None, pp_usable=int(pp_usable),
                      pp_length_m=path_length(path_raw) if pp else None, buffer_points=len(buffered),
                      buffer_length_m=path_length(points(buffered)),
                      actual_to_pp_m=distance_to_path(actual_xy, path_raw, True) if pp_usable else None,
                      actual_to_buffer_m=distance_to_path(actual_xy, points(buffered), True),
                      actual_to_course_m=distance_to_path(actual_xy, reference, True),
                      cte=cycle.get("cte", legacy.get("cte")), best_to_ref_rms_m=best_tracking_rms,
                      predicted_paths=len(rollouts), scored_reference_points=len(ref),
                      cone_center_distance_m=closest, odom_frame_delta_m=split)
        for key in ("speed_used", "raw_target_speed", "corner_speed", "speed_cap", "path_ahead",
                    "ahead_eff", "heading_err", "roll_ms", "cycle_ms", "horizon", "n_eff",
                    "cost_min", "cost_max", "temperature", "a0_std", "I_spd", "ff_part", "mppi_part",
                    "blue_in_fov", "yellow_in_fov", "path_exhausted", "path_bad", "bridge_fresh", "pp_silent",
                    "pseudo_reject", "pseudo_added", "cur", "N", "path_speed_cap", "braking_speed_cap",
                    "lookahead_m", "v_plan", "yaw_rate_seed", "frame_split_m", "path_source",
                    "planner_available"):
            sample[key] = cycle.get(key, legacy.get(key))
        timeline.append(sample)
        # Keep per-command geometry in the HTML. All original ROS frames are in the archive.
        frames.append(dict(pp=path_raw if pp_usable else [], buffer=buffered, pseudo=pseudo,
                           spline=path_spline, desired=ref, rollouts=rollouts,
                           cones=cone_colours if cone_row.get("frame") in ("map", "world") else {}))

    # Compare the best predicted rollout with future measured poses at its own simulator-time offsets.
    gt_sim = sorted((r["stamp_ns"] / 1e9, r["data"]["xy"]) for r in streams["gt"]
                    if r.get("stamp_ns") and points([r["data"]["xy"]]))
    sim_times = [t for t, _ in gt_sim]
    clock_resets = sum(r["data"].get("previous_sim_ns") is not None for r in streams["clock_epoch"])
    for row, sample, frame in zip(basis, timeline, frames):
        cycle = row["data"] if row["kind"] == "cycle" else {}
        best = frame["rollouts"][0] if frame["rollouts"] else []
        errors = []
        if gt_sim and cycle.get("sim_ns") and cycle.get("rollout_dt") and not clock_resets:
            for i, predicted in enumerate(best):
                if not points([predicted]):
                    continue
                target_t = cycle["sim_ns"] / 1e9 + cycle["rollout_first_dt"] + i * cycle["rollout_dt"]
                j = bisect.bisect_left(sim_times, target_t)
                candidate = min(gt_sim[max(0, j - 1):j + 1],
                                key=lambda item: abs(item[0] - target_t), default=None)
                if candidate and abs(candidate[0] - target_t) <= .04:
                    actual_pose = candidate[1]
                    errors.append(math.hypot(predicted[0] - actual_pose[0], predicted[1] - actual_pose[1]))
        sample["best_to_future_car_rms_m"] = metrics(errors)["rms"]
        sample["future_car_samples"] = len(errors)
        for key in ("pp", "buffer", "pseudo", "spline", "desired"):
            frame[key] = display_points(frame[key])
        frame["rollouts"] = [display_points(path) for path in frame["rollouts"]]
        frame["cones"] = {name: display_points(path) for name, path in frame["cones"].items()}

    # Downsample the background trail for the browser only.
    trail = [[r["t"], r["x"], r["y"]] for r in gt_frames if r["x"] is not None and r["y"] is not None]
    stride = max(1, math.ceil(len(trail) / 5500))
    gap_s = sum(entry["duration"] for entry in intervals)
    available_pp = [points(r["data"]["points"]) for r in streams["pp"]]
    pp_valid = [p for p in available_pp if len(p) >= args.min_pp_points]
    coverage = [r["data"] for r in streams["coverage"]]
    native_payload = any(item.get("native_contact_payload") for item in coverage)
    ros_payload = bool(streams["contact_ros"])
    observed_contact_stream = native_payload or ros_payload
    overshoots = excursions(timeline, "speed_error", args.overshoot_mps)
    deviations = excursions(timeline, "cte", args.cte_event_m, absolute=True)
    anomalies = ([dict(kind="Overspeed", unit="m/s", **item) for item in overshoots] +
                 [dict(kind="Cross-track", unit="m", **item) for item in deviations])
    anomalies.sort(key=lambda item: item["start"])
    paired_speeds = [r for r in timeline if r["speed_error"] is not None]
    steering = [r["steering"] for r in timeline if r["steering"] is not None]
    summary = dict(
        model=next((row["model"] for row in timeline if row["model"]), None),
        observed_seconds=end-start, clock_resets=clock_resets, invalid_json_rows=invalid_rows,
        clean_end=bool(streams["end"]), sample_basis=basis[0]["kind"],
        message_counts={name: len(rows) for name, rows in sorted(streams.items())},
        distance_m=distance, speed_error_mps=metrics([r["speed_error"] for r in timeline]),
        cross_track_m=metrics([abs(r["cte"]) for r in timeline if r["cte"] is not None]),
        peak_actual_speed=metrics([r["vx"] for r in timeline])["max"],
        rollout_ms=metrics([r["roll_ms"] for r in timeline]),
        cycle_ms=metrics([r["cycle_ms"] for r in timeline]),
        command_interval_ms=metrics([(b["t"] - a["t"]) * 1000 for a, b in zip(commands, commands[1:])]),
        overshoot=dict(threshold_mps=args.overshoot_mps, episodes=len(overshoots),
                       samples_braking=sum(r["speed_error"] >= args.overshoot_mps and
                                           r["accel"] is not None and r["accel"] < 0 for r in paired_speeds)),
        deviation=dict(threshold_m=args.cte_event_m, episodes=len(deviations)),
        steering_near_limit_fraction=(sum(abs(v) >= .30 for v in steering) / len(steering)
                                      if steering else None),
        modes=dict(collections.Counter(r["mode"] for r in timeline if r["mode"])),
        pp=dict(messages=len(available_pp), valid_messages=len(pp_valid),
                unique_valid_geometry=len({tuple((round(x, 2), round(y, 2)) for x, y in p) for p in pp_valid}),
                minimum_points=args.min_pp_points, freshness_s=args.pp_age,
                unavailable_seconds=gap_s, dropouts=len(intervals)),
        contact=dict(episodes=len(contacts) if observed_contact_stream else None,
                     unique_cones=len({e["cone"] for e in contacts}) if observed_contact_stream else None,
                     episode_gap_s=args.contact_gap, native_payload=native_payload, ros_payload=ros_payload,
                     partial_or_unknown=True, coverage=coverage),
        closest_mapped_cone_center_m=geometry_closest)

    for event in contacts:
        at = lookup["gt"].nearest(event["t"], .25).get("data", {})
        event.update(vx=at.get("vx"), x=(at.get("xy") or [None, None])[0], y=(at.get("xy") or [None, None])[1])
    save_csv(args.run / "timeline.csv", timeline)
    save_csv(args.run / "ground_truth.csv", gt_frames)
    save_csv(args.run / "cone_contacts.csv", contacts,
             ["t", "end", "cone", "vehicle_collision", "cone_collision", "samples", "sources", "sim_ns", "vx", "x", "y"])
    save_csv(args.run / "pp_dropouts.csv", intervals, ["start", "end", "duration"])
    save_csv(args.run / "excursions.csv", anomalies, ["kind", "unit", "start", "end", "peak", "peak_t", "x", "y", "mode"])
    (args.run / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    payload = dict(summary=summary, timeline=timeline, frames=frames, contacts=contacts, anomalies=anomalies,
                   gaps=intervals, reference=reference, trail=trail[::stride])
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")
    (args.run / "report.html").write_text(template.replace("/*__RUN_DATA__*/", encoded),
                                         encoding="utf-8")
    print(args.run / "report.html")
    print(f"{len(timeline)} timeline rows, {len(gt_frames)} ground-truth poses, "
          f"{len(pp_valid)} usable PP messages, "
          f"cone-contact episodes: {summary['contact']['episodes']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--pp-age", type=float, default=.5)
    parser.add_argument("--min-pp-points", type=int, default=5)
    parser.add_argument("--contact-gap", type=float, default=.4)
    parser.add_argument("--overshoot-mps", type=float, default=.25)
    parser.add_argument("--cte-event-m", type=float, default=1.)
    parser.add_argument("--reference", type=Path, help="Optional fixed map-frame centerline CSV with x,y columns")
    args = parser.parse_args()
    if min(args.pp_age, args.contact_gap, args.overshoot_mps, args.cte_event_m) <= 0 or args.min_pp_points < 2:
        parser.error("Use positive thresholds and at least two PP points.")
    analyze(args)


if __name__ == "__main__":
    main()
