"""Plan from Delaunay midpoints, remembering the path through shadow turns."""

from dataclasses import dataclass
import math

import numpy as np

from .cone_memory import (EPS, MAX_PROVISIONAL_TAIL, Corridor, cone_links, cross, segments,
                          trace_wall, validate_prefix)
from .pp_hybrid_turn import HybridConfig, HybridTurnPlanner, append_centerline


@dataclass(frozen=True)
class ShadowOnlyConfig(HybridConfig):
    """Settings for the shadow planner, including the one-sided tail limit."""

    provisional_length: float = MAX_PROVISIONAL_TAIL


class ShadowOnlyPlanner(HybridTurnPlanner):
    """Use the shadow search and build a centerline between the current lane's walls."""

    turn_planning_mode = 'shadow_turn'

    def __init__(self, config=None):
        super().__init__(config or ShadowOnlyConfig())

    def _trace_boundary(self, points, indices, links, triangulated, car, facing, side, *, orange_entry=False):
        return trace_wall(points, indices, links, triangulated, car, facing, side, self.cfg, delaunay_only=True)

    def _build_corridor(self, points, ages, colours, left, right, links, *, entry_pairs=None):
        if len(left) < 2 or len(right) < 2:
            return None
        # Triangulate the selected lane alone so neighbouring lanes cannot supply cross edges.
        left, right = np.asarray(left), np.asarray(right)
        lwall, rwall = points[left], points[right]
        mesh, _ = cone_links(np.vstack((lwall, rwall)), self.cfg.width_max)
        pairs = [(i, j - len(left)) for i, j in mesh if i < len(left) <= j]
        pairs.sort(key=lambda pair: (sum(pair), pair))
        left_tangent = np.gradient(lwall, axis=0)
        right_tangent = np.gradient(rwall, axis=0)
        entry_pairs = entry_pairs or {}
        gates = []
        last_left, last_right = -1, -1
        for i, j in pairs:
            if i < last_left or j < last_right:
                continue
            a, b = left[i], right[j]
            entry_gate = entry_pairs.get(a) == b
            if colours[a] == colours[b] and not entry_gate:
                continue
            span = lwall[i] - rwall[j]
            if (np.linalg.norm(span) < self.cfg.width_min - EPS or
                    min(cross(left_tangent[i], span), cross(right_tangent[j], span)) <= EPS):
                continue
            midpoint = (lwall[i] + rwall[j]) / 2
            if gates and np.linalg.norm(midpoint - gates[-1][0]) <= 0.08:
                continue
            gates.append((midpoint, lwall[i], rwall[j], min(ages[a], ages[b]), entry_gate))
            last_left, last_right = i, j
        if len(gates) < 2:
            return None
        center, a, b, seen, entry = (np.array(column) for column in zip(*gates))
        size = validate_prefix(center, *segments(lwall, rwall), self.cfg)
        if size < 2 or (entry_pairs and not entry[:size].any()):
            return None
        return Corridor(center[:size], a[:size], b[:size], seen[:size], np.ones(size, dtype=bool))

    def _candidate_from_saved_walls(self, candidate, walls, car, heading, now):
        """Keep the observed candidate and skip the TLS wall proposal."""
        return candidate

    def _splice_shadow(self, base, candidate, visible, walls):
        # The first published segment gives the two remembered points at the car.
        first = max(0, np.searchsorted(base.arc, self.progress, side='right') - 1)
        prefix = base.part(end=first + 2)
        merged = append_centerline(prefix, candidate, self.cfg, visible=visible, walls=walls,
                                   shadow=True, target_indices=range(len(candidate.center)))
        if merged is prefix:
            return base, len(base.center)
        return merged, len(prefix.center)

    def plan(self, car, yaw, now):
        result = super().plan(car, yaw, now)
        if not len(result.points):
            return result
        labels = self.cache.sources[-len(result.points):]
        samples, sources = [result.points[:1]], [labels[0]]
        for start, end, source, next_source in zip(result.points[:-1], result.points[1:], labels[:-1], labels[1:]):
            # Keep the five spline samples and fill only the straight segments between path vertices.
            steps = 1 if 'shadow_spline' in (source, next_source) else max(
                1, math.ceil(np.linalg.norm(end - start) / self.cfg.sample_step - EPS))
            samples.append(np.linspace(start, end, steps + 1)[1:])
            sources.extend([source] * (steps - 1) + [next_source])
        result.points = np.concatenate(samples)
        result.point_sources = np.array(sources)
        return result


def main(args=None):
    """Run the shadow-only planner using the hybrid's topics and rate."""
    from .pp_cone_memory import ConeMemoryNode, run_planner

    run_planner(lambda: ConeMemoryNode(ShadowOnlyPlanner, ShadowOnlyConfig, 'pp_shadow_only', timer_period=0.1), args)
