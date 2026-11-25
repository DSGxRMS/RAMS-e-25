"""Cone-memory straights with TLS wall-curvature proposals through turns."""

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import cKDTree

from .cone_memory import (EPS, Config, ConeMemoryPlanner, Corridor, arc_length, cone_links,
                          cross_sections, project, sample_at, segments, unit, validate_prefix)
from .tls_curvature import guard_predictions, predict_wall


@dataclass(frozen=True)
class HybridConfig(Config):
    """Turn detection, directional search, and confirmed-corridor retention."""

    turn_count_difference: int = 3
    turn_confirm_updates: int = 2
    turn_angle_deg: float = 18.0
    turn_exit_angle_deg: float = 8.0
    turn_max_yaw_deg: float = 60.0
    turn_memory_s: float = 30.0

    def __post_init__(self):
        super().__post_init__()
        if (self.turn_count_difference < 2 or self.turn_confirm_updates < 1
                or not 0 < self.turn_exit_angle_deg < self.turn_angle_deg < 90
                or not 0 < self.turn_max_yaw_deg < 90 or self.search_yaw > self.turn_max_yaw_deg
                or not math.isfinite(self.turn_memory_s) or self.turn_memory_s < self.max_age):
            raise ValueError('Invalid hybrid turn settings')


def wall_turn(wall):
    """Return the strongest signed heading change on the approach to the turn."""
    legs = np.diff(wall[:5], axis=0)
    if len(legs) < 2:
        return 0.0
    headings = np.arctan2(legs[:, 1], legs[:, 0])
    changes = (np.diff(headings) + np.pi) % (2 * np.pi) - np.pi
    return float(np.degrees(changes[np.argmax(np.abs(changes))]))


class HybridTurnPlanner(ConeMemoryPlanner):
    """Keep measured path points and use TLS only to propose new turn geometry."""

    def __init__(self, config=None):
        super().__init__(config or HybridConfig())

    def reset(self):
        super().reset()
        self.saved_walls = [None, None]
        self.turn_active = False
        self.sweep_offset = 0.0
        self.imbalance_revision = -1
        self.imbalance_direction = 0
        self.imbalance_votes = 0
        self.exit_revision = -1
        self.exit_votes = 0
        self.prediction = None

    def _search_offsets(self, recovering, startup):
        cfg = self.cfg
        if startup:
            return ()
        left, right = self.counts
        delta = right - left
        direction = int(np.sign(delta)) if abs(delta) >= cfg.turn_count_difference else 0
        if self.revision != self.imbalance_revision:
            self.imbalance_revision = self.revision
            self.imbalance_votes = self.imbalance_votes + 1 if direction and direction == self.imbalance_direction else int(bool(direction))
            self.imbalance_direction = direction
        if direction and self.imbalance_votes >= cfg.turn_confirm_updates:
            self.turn_active = True
            self.sweep_offset = float(np.clip(self.sweep_offset + direction * cfg.search_yaw,
                                              -cfg.turn_max_yaw_deg, cfg.turn_max_yaw_deg))
        elif not self.turn_active:
            self.sweep_offset = math.copysign(max(0.0, abs(self.sweep_offset) - cfg.search_yaw), self.sweep_offset)
        if self.sweep_offset:
            return (self.sweep_offset,)
        if not self.turn_active and not direction:
            return super()._search_offsets(recovering, startup)
        return ()

    def _check_turn_geometry(self, walls, now):
        if walls is None or self.revision == self.exit_revision:
            return
        self.exit_revision = self.revision
        left, right = (wall_turn(points) for points, _ in walls)
        if left * right > 0 and min(abs(left), abs(right)) >= self.cfg.turn_angle_deg:
            self.turn_active = True
        fresh_walls = all(np.count_nonzero(now - seen <= 0.4) >= 4 for _, seen in walls)
        straight = max(abs(left), abs(right)) <= self.cfg.turn_exit_angle_deg
        if fresh_walls and straight and min(self.counts) >= 2 and abs(self.counts[0] - self.counts[1]) <= 1:
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
        candidate, walls, observed = super()._assemble(car, heading, probe, now)
        self.prediction = None
        if self.startup_scan_complete:
            self._check_turn_geometry(walls, now)
        # Use the previously saved walls for this proposal, then store this cycle's evidence.
        if self.turn_active and all(wall is not None for wall in self.saved_walls):
            (left, lt), (right, rt) = self.saved_walls
            if now - min(min(lt), min(rt)) <= self.cfg.max_age:
                saved = self._pair_walls(left, right, lt, rt)
                if self._tail_span(saved, car, heading) > self._tail_span(candidate, car, heading) + 0.15:
                    candidate = saved
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
        self._remember_walls(walls, now)
        return candidate, walls, observed

    def _expired_samples(self, cache, now):
        limit = np.where(cache.confirmed & self.turn_active, self.cfg.turn_memory_s, self.cfg.max_age)
        return now - cache.seen > limit

    def _validate_cache(self, visible, walls, now):
        super()._validate_cache(visible, walls, now)
        if self.last_build != now or self.cache is None or self.prediction is None:
            return
        self._append_prediction(visible, walls)

    def _append_prediction(self, visible, walls):
        prediction = self.prediction
        confirmed = np.flatnonzero(self.cache.confirmed)
        if not len(confirmed) or len(prediction.center) < 2:
            return
        end = int(confirmed[-1])
        base = self.cache.part(end=end + 1)
        if len(base.center) < 2 or self.cfg.provisional_length <= 0:
            return
        distance, seam = cKDTree(prediction.center).query(base.center[-1])
        if distance > 0.35 or seam >= len(prediction.center) - 1:
            return
        tangent = unit(base.center[-1] - base.center[-2])
        outgoing = unit(prediction.center[seam + 1] - prediction.center[seam])
        if tangent @ outgoing < math.cos(math.radians(18)):
            return
        if max(np.linalg.norm(base.left[-1] - prediction.left[seam]),
               np.linalg.norm(base.right[-1] - prediction.right[seam])) > 0.45:
            return
        start = seam + 1
        while start < len(prediction.center) and (prediction.center[start] - base.center[-1]) @ tangent < 0.12:
            start += 1
        tail = prediction.part(start=start)
        if not len(tail.center):
            return
        length = arc_length(np.vstack((base.center[-1], tail.center)))[1:]
        count = int(np.count_nonzero(length <= self.cfg.provisional_length + EPS))
        if len(visible) and count:
            distance = cKDTree(visible).query(tail.center[:count])[0]
            hits = np.flatnonzero(distance < self.cfg.clearance + self.cfg.sample_step / 2)
            if len(hits):
                count = int(hits[0])
        if not count:
            return
        tail = tail.part(end=count)
        merged = Corridor(np.vstack((base.center, tail.center)), np.vstack((base.left, tail.left)),
                          np.vstack((base.right, tail.right)), np.r_[base.seen, tail.seen],
                          np.r_[base.confirmed, np.zeros(count, dtype=bool)])
        starts, ends = segments(merged.left, merged.right)
        if walls is not None:
            actual_start, actual_end = segments(walls[0][0], walls[1][0])
            starts, ends = np.vstack((starts, actual_start)), np.vstack((ends, actual_end))
        check_from = max(0, end - 1)
        size = validate_prefix(merged.center[check_from:], starts, ends, self.cfg)
        merged = merged.part(end=check_from + size)
        if len(merged.center) > end + 1 and merged.arc[-1] > self.cache.arc[-1] + 0.08:
            self.cache = merged

    def plan(self, car, yaw, now):
        result = super().plan(car, yaw, now)
        result.planning_mode = 'tls_turn' if self.turn_active else 'straight'
        return result


def main(args=None):
    """Run the hybrid using the cone-memory topics and visualizer."""
    from .pp_cone_memory import ConeMemoryNode, run_planner

    run_planner(lambda: ConeMemoryNode(HybridTurnPlanner, HybridConfig, 'pp_hybrid_turn'), args)
