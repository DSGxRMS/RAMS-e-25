"""Cone-memory straights with TLS wall-curvature proposals through turns."""

from dataclasses import dataclass, replace
import math

import numpy as np
from scipy.spatial import cKDTree

from .cone_memory import (EPS, Config, ConeMemoryPlanner, Corridor, arc_length, cone_links,
                          connector_curve, cross_sections, entry_available, join_corridors,
                          project, sample_at, segments, unit, validate_prefix)
from .tls_curvature import guard_predictions, predict_wall


@dataclass(frozen=True)
class HybridConfig(Config):
    """Turn detection, shadow search, and confirmed-corridor retention."""

    lookahead: float = 5.0
    turn_min_confirmed_path_m: float = 5.0
    turn_confirm_updates: int = 2
    turn_exit_angle_deg: float = 8.0
    shadow_angle_deg: float = 40.0

    def __post_init__(self):
        super().__post_init__()
        if (not 0 < self.turn_min_confirmed_path_m <= self.horizon or self.turn_confirm_updates < 1
                or not 0 < self.turn_exit_angle_deg < 90
                or not 0 < self.shadow_angle_deg < 180 - self.fov_angle / 2):
            raise ValueError('Invalid hybrid turn settings')


def wall_turn(wall):
    """Return the strongest signed heading change on the approach to the turn."""
    legs = np.diff(wall[:5], axis=0)
    if len(legs) < 2:
        return 0.0
    headings = np.arctan2(legs[:, 1], legs[:, 0])
    changes = (np.diff(headings) + np.pi) % (2 * np.pi) - np.pi
    return float(np.degrees(changes[np.argmax(np.abs(changes))]))


def append_centerline(base, candidate, cfg, *, provisional_limit=None, visible=(), walls=None, shadow=False,
                      target_indices=None):
    """Extend a fixed prefix, using a spline for shadow joins."""
    if len(base.center) < 2 or len(candidate.center) < 2:
        return base
    end = base.center[-1]
    heading = unit(end - base.center[-2])
    if target_indices is None:
        legs = np.diff(candidate.center, axis=0)
        lengths = np.diff(candidate.arc)
        relative = end - candidate.center[:-1]
        u = np.clip(np.sum(relative * legs, axis=1) / np.maximum(lengths**2, EPS), 0, 1)
        distance = np.linalg.norm(relative - u[:, None] * legs, axis=1)
        aligned = (legs @ heading >= lengths * math.cos(math.radians(40))) & (lengths > EPS)
        seam_limit = 0.35
        if provisional_limit is None and not base.confirmed[-1]:
            seam_limit = max(seam_limit, 3 * cfg.sample_step)
        nearby = np.flatnonzero(aligned & (distance <= seam_limit))
        if len(nearby):
            nearest = nearby[np.argmin(distance[nearby])]
            start = candidate.arc[nearest] + u[nearest] * lengths[nearest]
        elif shadow:
            start = candidate.arc[0]
        else:
            return base
        first = np.searchsorted(candidate.arc, start - EPS)
        # Keep the TLS overlap search local even with a longer FOV lookahead.
        lead = cfg.connector_max_lead if provisional_limit is None else min(cfg.lookahead, 2.5)
        final = np.searchsorted(candidate.arc, start + lead, side='right')
        target_indices = range(first, final)
    cone_tree = cKDTree(visible) if len(visible) else None
    for index in target_indices:
        target = candidate.center[index]
        span = np.linalg.norm(target - end)
        if span < 0.14 or (target - end) @ heading <= 0.12:
            continue
        if shadow and span > cfg.connector_max_lead:
            continue
        if not shadow:
            curve = np.empty((0, 2))
            if cone_tree is not None and span > cfg.sample_step + EPS:
                dense = np.linspace(end, target, math.ceil(span / 0.10) + 1)
                if np.any(cone_tree.query(dense)[0] < cfg.clearance + 0.05):
                    continue
        else:
            _, tangent = sample_at(candidate, candidate.arc[index])
            spline = connector_curve(end, heading, target, tangent, cfg)
            if spline is None:
                continue
            if cone_tree is not None:
                dense = spline(np.linspace(0, 1, max(5, math.ceil(span / 0.05) + 1)))
                if np.any(cone_tree.query(dense)[0] < cfg.clearance + 0.05):
                    continue
            curve = spline(np.linspace(0, 1, 5)[1:-1])
        portion = np.arange(1, len(curve) + 1)[:, None] / (len(curve) + 1)
        tail = candidate.part(start=index)
        center = np.vstack((curve, tail.center))
        left = np.vstack((base.left[-1] + portion * (tail.left[0] - base.left[-1]), tail.left))
        right = np.vstack((base.right[-1] + portion * (tail.right[0] - base.right[-1]), tail.right))
        bridge_stamp = min(base.seen[-1], tail.seen[0])
        confirmed = np.r_[np.zeros(len(curve), dtype=bool), tail.confirmed]
        count = len(center)
        if provisional_limit is not None:
            length = arc_length(np.vstack((end, center)))[1:]
            count = int(np.count_nonzero(length <= provisional_limit + EPS))
            confirmed[:] = False
        if cone_tree is not None and count:
            near_cone = cone_tree.query(center[:count])[0] < cfg.clearance + cfg.sample_step / 2
            if near_cone.any():
                count = int(np.argmax(near_cone))
        if not count:
            continue

        seen = np.r_[np.full(len(curve), bridge_stamp), tail.seen]
        sources = tail.sources
        if shadow:
            sources = np.r_[np.repeat('shadow_spline', len(curve)), np.repeat('shadow', len(tail.center))]
        elif provisional_limit is not None:
            sources = np.repeat('tls', len(tail.center))
        merged = Corridor(np.vstack((base.center, center[:count])), np.vstack((base.left, left[:count])),
                          np.vstack((base.right, right[:count])), np.r_[base.seen, seen[:count]],
                          np.r_[base.confirmed, confirmed[:count]], np.r_[base.sources, sources[:count]])
        starts, ends = segments(merged.left, merged.right)
        if walls is not None:
            observed_starts, observed_ends = segments(walls[0][0], walls[1][0])
            starts, ends = np.vstack((starts, observed_starts)), np.vstack((ends, observed_ends))
        check_from = len(base.center) - 2
        valid = check_from + validate_prefix(merged.center[check_from:], starts, ends, cfg)
        if valid > len(base.center) and (provisional_limit is not None
                                         or merged.confirmed[len(base.center):valid].any()):
            return merged.part(end=valid)
    return base


def refresh_centerline(base, measured):
    """Update the evidence of stored samples observed again in the same direction."""
    distance, match = cKDTree(measured.center).query(base.center)
    base_legs = np.diff(base.center, axis=0)
    measured_legs = np.diff(measured.center, axis=0)
    incoming = np.vstack((base_legs, base_legs[-1]))
    observed = measured_legs[np.minimum(match, len(measured_legs) - 1)]
    alignment = np.sum(incoming * observed, axis=1)
    alignment /= np.maximum(np.linalg.norm(incoming, axis=1) * np.linalg.norm(observed, axis=1), EPS)
    within = (match != 0) | ((base.center - measured.center[0]) @ measured_legs[0] >= -EPS)
    within &= (match != len(measured.center) - 1) | ((base.center - measured.center[-1]) @ measured_legs[-1] <= EPS)
    supported = (distance <= 0.18) & (alignment > 0.97) & within & measured.confirmed[match]
    if not supported.any():
        return base
    return replace(base, seen=np.where(supported, np.maximum(base.seen, measured.seen[match]), base.seen),
                   confirmed=base.confirmed | supported)


class HybridTurnPlanner(ConeMemoryPlanner):
    """Keep measured path points and use TLS only to propose new turn geometry."""

    nearest_fallback = True
    turn_planning_mode = 'tls_turn'

    def __init__(self, config=None):
        super().__init__(config or HybridConfig())

    def reset(self, *, keep_path_memory=False):
        super().reset(keep_path_memory=keep_path_memory)
        self.saved_walls = [None, None]
        self.turn_active = False
        self.exit_revision = -1
        self.exit_votes = 0
        self.prediction = None
        self.shadow_candidate = None

    def _probe(self, car, heading):
        return self._path_probe(car, heading, self.cfg.lookahead)

    def _search_offsets(self, recovering, startup):
        return ()

    def _shadow_bands(self, startup):
        if startup:
            return ()
        left, right = self.counts
        half = self.cfg.fov_angle / 2
        outer = half + self.cfg.shadow_angle_deg
        if left == right == 0:
            return ((-outer, -half), (half, outer))
        if left <= 1 and left < right:
            return ((half, outer),)
        if right <= 1 and right < left:
            return ((-outer, -half),)
        return ()

    def _defer_shadow(self, corridor):
        if self.cache is None:
            return False
        self.shadow_candidate = corridor
        return True

    def _join_candidate(self, candidate):
        first = max(0, np.searchsorted(self.cache.arc, self.progress, side='right') - 1)
        if self.turn_active and not self.cache.confirmed[first:].all():
            base = refresh_centerline(self.cache, candidate)
            return append_centerline(base, candidate, self.cfg)
        return join_corridors(self.cache, candidate, self.progress, self.cfg, match_walls=False)

    def _confirmed_ahead(self, candidate, car, heading):
        """Measure the continuous confirmed path after joining current and saved geometry."""
        corridor, station = self.cache, self.progress
        if candidate is not None and len(candidate.center) >= 2:
            start = project(candidate, car, heading)
            if start is None and entry_available(candidate, car, heading, self.cfg):
                start = 0.0
            if start is not None:
                corridor = self._join_candidate(candidate) if self.cache is not None else candidate
                if corridor is candidate:
                    station = start
        if corridor is None:
            return 0.0
        at = max(0, np.searchsorted(corridor.arc, station, side='right') - 1)
        if not corridor.confirmed[at]:
            return 0.0
        first = np.searchsorted(corridor.arc, station)
        invalid = np.flatnonzero(~corridor.confirmed[first:])
        stop = first + invalid[0] if len(invalid) else len(corridor.arc)
        return max(0.0, float(corridor.arc[stop - 1] - station)) if stop > first else 0.0

    def _update_turn_mode(self, walls, confirmed_ahead, now):
        if min(self.counts) <= 1 or confirmed_ahead < self.cfg.turn_min_confirmed_path_m:
            self.turn_active = True
            self.exit_votes = 0
            self.exit_revision = self.revision
            return
        if not self.turn_active or walls is None or self.revision == self.exit_revision:
            return
        self.exit_revision = self.revision
        left, right = (wall_turn(points) for points, _ in walls)
        fresh_walls = all(np.count_nonzero(now - seen <= 0.4) >= 4 for _, seen in walls)
        straight = max(abs(left), abs(right)) <= self.cfg.turn_exit_angle_deg
        if fresh_walls and straight:
            self.exit_votes += 1
        else:
            self.exit_votes = 0
        if self.exit_votes >= self.cfg.turn_confirm_updates:
            self.turn_active = False
            self.exit_votes = 0

    def _remember_walls(self, walls, now):
        if walls is None:
            return
        for side, (points, seen) in enumerate(walls):
            if len(points) < 4:
                continue
            tip, stamps = points[-4:], seen[-4:]
            old = self.saved_walls[side]
            if now - min(stamps) <= self.cfg.max_age and (old is None or max(stamps) > max(old[1])):
                self.saved_walls[side] = (tip.copy(), stamps.copy())

    def _pair_walls(self, left, right, left_stamps, right_stamps):
        points = np.vstack((left, right))
        links, _ = cone_links(points, self.cfg.boundary_max_edge)
        return cross_sections(points, np.r_[left_stamps, right_stamps], range(len(left)),
                              range(len(left), len(points)), links, self.cfg)

    def _tail_span(self, corridor, car, heading):
        if corridor is None or len(corridor.center) < 2:
            return 0.0
        station = project(corridor, car, heading)
        if station is None:
            return 0.0
        if self.cache is not None:
            near, tangent = sample_at(self.cache, self.progress)
            overlap = project(corridor, near, tangent)
            if overlap is None or np.linalg.norm(sample_at(corridor, overlap)[0] - near) > 0.5:
                return 0.0
        return corridor.arc[-1] - station

    def _assemble(self, car, heading, probe, now):
        self.shadow_candidate = None
        candidate, walls, observed = super()._assemble(car, heading, probe, now)
        self.prediction = None
        if self.startup_scan_complete:
            self._update_turn_mode(walls, self._confirmed_ahead(candidate, car, heading), now)
        candidate = self._candidate_from_saved_walls(candidate, walls, car, heading, now)
        return candidate, walls, observed

    def _candidate_from_saved_walls(self, candidate, walls, car, heading, now):
        # Use the previously saved walls for this proposal, then store this cycle's evidence.
        if self.turn_active and all(wall is not None for wall in self.saved_walls):
            (left, lt), (right, rt) = self.saved_walls
            if now - min(min(lt), min(rt)) <= self.cfg.max_age:
                saved = self._pair_walls(left, right, lt, rt)
                if self._tail_span(saved, car, heading) > self._tail_span(candidate, car, heading) + 0.15:
                    candidate = replace(saved, point_sources=np.repeat('saved_boundaries', len(saved.center)))
                    self.scan.candidate = saved.center
                lpred, lk = predict_wall(left)
                rpred, rk = predict_wall(right)
                lpred, rpred = guard_predictions(left, lpred, lk, right, rpred, rk, self.width)
                if len(lpred) or len(rpred):
                    full_left = np.vstack((left, lpred))
                    full_right = np.vstack((right, rpred))
                    self.prediction = self._pair_walls(full_left, full_right,
                                                       np.r_[lt, np.repeat(lt[-1], len(lpred))],
                                                       np.r_[rt, np.repeat(rt[-1], len(rpred))])
                    if self.prediction is not None:
                        self.scan.tls_candidate = self.prediction.center
        self._remember_walls(walls, now)
        return candidate

    def _validate_cache(self, visible, walls, now):
        super()._validate_cache(visible, walls, now)
        if self.last_build != now or self.cache is None:
            return
        if self.prediction is not None:
            self._append_prediction(visible, walls)
        if self.shadow_candidate is not None:
            self._merge_shadow(visible, walls)

    def _merge_shadow(self, visible, walls):
        candidate = self.shadow_candidate
        base = refresh_centerline(self.cache, candidate)
        merged, prefix_size = self._splice_shadow(base, candidate, visible, walls)
        self.cache = merged
        if merged is not base:
            self.scan.candidate = candidate.center
            self.scan.selected_yaw = 0.0
            self.scan.shadow_selected = True
            self.scan.shadow_outcome = 'merged'
            self.scan.shadow_added = merged.center[prefix_size:]

    def _splice_shadow(self, base, candidate, visible, walls):
        prefix_size = len(base.center)
        merged = append_centerline(base, candidate, self.cfg, visible=visible, walls=walls, shadow=True)
        if merged is base and not base.confirmed[-1]:
            # Try an earlier overlap on the predicted tail if it diverges from observed geometry.
            measured = np.flatnonzero(base.confirmed)
            earliest = int(measured[-1]) if len(measured) else 0
            earliest = max(earliest, int(np.searchsorted(base.arc, self.progress)))
            for index in range(len(base.center) - 2, max(earliest, 1) - 1, -1):
                prefix = base.part(end=index + 1)
                trial = append_centerline(prefix, candidate, self.cfg, visible=visible, walls=walls, shadow=True)
                if trial is not prefix and trial.arc[-1] > base.arc[-1] + 0.08:
                    merged = trial
                    prefix_size = len(prefix.center)
                    break
        return merged, prefix_size

    def _append_prediction(self, visible, walls):
        confirmed = np.flatnonzero(self.cache.confirmed)
        if len(confirmed):
            available = self.cfg.provisional_length - (self.cache.arc[-1] - self.cache.arc[confirmed[-1]])
            if available > EPS:
                prefix_size = len(self.cache.center)
                self.cache = append_centerline(self.cache, self.prediction, self.cfg,
                                               provisional_limit=available, visible=visible, walls=walls)
                if len(self.cache.center) > prefix_size:
                    self.scan.tls_added = self.cache.center[prefix_size:]

    def plan(self, car, yaw, now):
        result = super().plan(car, yaw, now)
        result.planning_mode = self.turn_planning_mode if self.turn_active else 'straight'
        return result


def main(args=None):
    """Run the hybrid using the cone-memory topics and visualizer."""
    from .pp_cone_memory import ConeMemoryNode, run_planner

    run_planner(lambda: ConeMemoryNode(HybridTurnPlanner, HybridConfig, 'pp_hybrid_turn', timer_period=0.1), args)
