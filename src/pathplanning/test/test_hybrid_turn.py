"""Regression checks for the hybrid planner's turn handoff."""

from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import numpy as np

from pathplanning.cone_memory import Corridor, Scan, cone_links, connector_curve, provisional_tail, sample_at
from pathplanning.pp_hybrid_turn import HybridConfig, HybridTurnPlanner, append_centerline
from pathplanning.pp_shadow_only import ShadowOnlyConfig, ShadowOnlyPlanner


def straight_corridor(start, end, half_width, offset=0.0):
    x = np.arange(start, end + 0.125, 0.25)
    center = np.column_stack((x, np.full(len(x), offset)))
    wall_offset = np.array([0.0, half_width])
    return Corridor(center, center + wall_offset, center - wall_offset,
                    np.ones(len(x)), np.ones(len(x), dtype=bool))


class HybridTurnTest(unittest.TestCase):
    def test_orange_midpoint_bridges_startup_boundary_gaps(self):
        config = HybridConfig(boundary_max_edge=4.0, anchor_ahead=7.0, entry_gap=6.5,
                              max_age=1.0, width_age=0.5)
        planner = HybridTurnPlanner(config)
        blue = [(x, 1.9) for x in (2.5, 7.5, 10.5, 13.5)]
        yellow = [(x, -1.9) for x in (2.6, 6.8, 9.8, 12.8)]
        orange = [(4.8, 1.8), (4.7, -1.9)]
        car = (0.0, 0.0)

        planner.observe(blue, yellow, 1.0, True, orange=orange[:1])
        # Both coloured walls bridge the long gaps even with a lone orange cone.
        self.assertEqual(planner.plan(car, 0.0, 1.0).state, 'track')
        planner = HybridTurnPlanner(config)

        planner.observe(blue, yellow, 1.2, True, orange=orange)
        result = planner.plan(car, 0.0, 1.2)
        self.assertTrue(planner.startup_scan_complete)
        self.assertGreater(result.points[-1, 0], 10.0)
        midpoint = np.mean(orange, axis=0)
        self.assertTrue(np.isclose(planner.cache.center, midpoint).all(axis=1).any())

        for stamp in (1.7, 2.2, 2.7, 3.2):
            planner.observe(blue, yellow, stamp, True, orange=orange)
            result = planner.plan(car, 0.0, stamp)
        self.assertFalse(planner.scan.startup)
        self.assertGreater(result.points[-1, 0], 10.0)

    def test_shadow_appends_after_tls_without_matching_wall_positions(self):
        planner = HybridTurnPlanner()
        planner.cache = remembered = straight_corridor(0.0, 2.0, 1.5)
        planner.prediction = straight_corridor(1.75, 5.0, 2.1)
        planner.shadow_candidate = straight_corridor(3.75, 7.0, 1.5)
        planner.startup_scan_complete = True
        planner.turn_active = True
        planner.last_build = 1.0
        car = np.zeros(2)
        heading = np.array([1.0, 0.0])
        planner.scan = Scan(car, (car, heading), 1.0, False)

        with patch('pathplanning.pp_hybrid_turn.connector_curve', wraps=connector_curve) as spline:
            planner._validate_cache(np.empty((0, 2)), None, 1.0)
        self.assertEqual(spline.call_count, 1)
        cache = planner.cache
        np.testing.assert_allclose(cache.center[:len(remembered.center)], remembered.center)
        np.testing.assert_allclose(cache.center[:, 0], np.r_[np.arange(0, 4, 0.25), np.linspace(4, 4.25, 5),
                                                           np.arange(4.5, 7.125, 0.25)])
        np.testing.assert_allclose(cache.left[(cache.center[:, 0] > 2.0) & (cache.center[:, 0] <= 4.0), 1], 2.1)
        np.testing.assert_allclose(cache.left[cache.center[:, 0] >= 4.25, 1], 1.5)
        np.testing.assert_array_equal(cache.confirmed, (cache.center[:, 0] <= 2.0) | ((cache.center[:, 0] >= 3.75) & (cache.center[:, 0] <= 4)) | (cache.center[:, 0] >= 4.25))
        self.assertTrue(planner.scan.shadow_selected)
        self.assertAlmostEqual(planner._confirmed_ahead(None, car, heading), 2.0)
        self.assertTrue(planner.plan(car, 0.0, 1.05).provisional)

        planner.progress = 3.625
        self.assertEqual(planner._confirmed_ahead(None, car, heading), 0.0)
        planner.progress = 3.75
        self.assertAlmostEqual(planner._confirmed_ahead(None, car, heading), 0.25)

    def test_shadow_append_requires_nearby_centerlines(self):
        base = straight_corridor(0.0, 2.0, 1.5)
        disconnected = straight_corridor(1.75, 7.0, 1.5, offset=1.0)
        self.assertIs(append_centerline(base, disconnected, HybridConfig()), base)

    def test_five_point_shadow_spline_across_gap(self):
        base = straight_corridor(0, 2, 1.5)
        shadow = straight_corridor(3.5, 7, 1.5, offset=0.06)
        merged = append_centerline(base, shadow, HybridConfig(), shadow=True)
        seam = merged.center[len(base.center)-1:len(base.center)+4]
        self.assertEqual(len(seam), 5)
        np.testing.assert_allclose(seam[[0, -1]], [base.center[-1], shadow.center[0]])
        self.assertTrue(np.all(np.diff(merged.center[:, 0]) > 0))
        np.testing.assert_allclose(merged.center[-len(shadow.center):], shadow.center)
        planner = HybridTurnPlanner()
        planner.cache = merged
        result = planner.plan((0, 0), 0, 50.0)
        np.testing.assert_array_equal(result.points, merged.center)
        self.assertEqual(planner.remembered_path_points, set(map(tuple, merged.center)))

    def test_shadow_width_is_fixed_and_points_toward_the_sparser_side(self):
        planner = HybridTurnPlanner()
        cases = [((5, 2), ()), ((2, 1), ((-70, -30),)),
                 ((1, 2), ((30, 70),)), ((0, 1), ((30, 70),)), ((1, 0), ((-70, -30),)),
                 ((0, 0), ((-70, -30), (30, 70))), ((1, 1), ())]
        for counts, bands in cases:
            with self.subTest(counts=counts):
                planner.counts = counts
                self.assertEqual(planner._shadow_bands(False), bands)
                self.assertEqual(planner._shadow_bands(True), ())
                self.assertEqual(planner._search_offsets(True, False), ())

    def test_shadow_search_adds_only_the_selected_band_to_the_nominal_scan(self):
        planner = HybridTurnPlanner(HybridConfig(horizon=12, colour_mode='blue_left'))
        planner.startup_scan_complete = True
        origin, tangent = np.array([2.5, 0.0]), np.array([1.0, 0.0])

        def at_angle(angle, radius=7.0):
            angle = math.radians(angle)
            return origin + radius * np.array([math.cos(angle), math.sin(angle)])

        blue = [at_angle(15), at_angle(50), at_angle(85), at_angle(50, 13)]
        yellow = [at_angle(-15), at_angle(-20, 9), at_angle(-50)]
        planner.observe(blue, yellow, 1.0, True)
        traced = []

        def record_chain(points, indices, *_):
            traced.append({tuple(point) for point in points[indices]})
            return indices.tolist()

        def rounded(point):
            return tuple(np.round(point, 4))

        with patch('pathplanning.cone_memory.trace_wall', side_effect=record_chain):
            planner._assemble(np.zeros(2), tangent, (origin, tangent), 1.0)
        self.assertEqual(planner.counts, (1, 2))
        self.assertEqual(planner.scan.tested_yaws, (0.0,))
        self.assertEqual(planner.scan.shadow_bands_deg, ((30, 70),))
        self.assertEqual(traced[0], {rounded(blue[0])})
        self.assertEqual(traced[2], {rounded(blue[0]), rounded(blue[1])})
        self.assertEqual(traced[3], traced[1])

        planner.observe(blue[1:], yellow[-1:], 9.1, True)
        traced.clear()
        with patch('pathplanning.cone_memory.trace_wall', side_effect=record_chain):
            planner._assemble(np.zeros(2), tangent, (origin, tangent), 9.1)
        self.assertEqual(planner.counts, (0, 0))
        self.assertEqual(planner.scan.shadow_bands_deg, ((-70, -30), (30, 70)))
        self.assertEqual(traced[2], {rounded(blue[1])})
        self.assertEqual(traced[3], {rounded(yellow[-1])})
        self.assertEqual({rounded(cone.point) for cone, _ in planner.scan.shadow_cones},
                         {rounded(blue[1]), rounded(yellow[-1])})

    def test_shadow_cones_can_extend_the_confirmed_path(self):
        blue = [(x, 2.4) for x in (-1, 1, 3, 5, 7)]
        yellow = [(x, -1.2) for x in (-2, 0, 2, 4, 6, 8)]
        car = (0, 0.6)
        for planner_type, ends in ((HybridTurnPlanner, (3.0, 7.0)), (ShadowOnlyPlanner, (3.5, 7.5))):
            for width, confirmed_end in zip((0.01, 40.0), ends):
                with self.subTest(planner=planner_type.__name__, shadow_width=width):
                    config = HybridConfig(width_max=4.0, nominal_width=3.6, lookahead=2.5,
                                          colour_mode='blue_left', shadow_angle_deg=width)
                    planner = planner_type(config)
                    # Isolate shadow recovery from the independent long-edge fallback.
                    planner.nearest_fallback = False
                    planner.observe(blue[:3], yellow[:4], 1.0, True)
                    planner.plan(car, 0, 1.0)
                    planner.observe(blue, yellow, 1.2, True)
                    planner.plan(car, 0, 1.2)
                    self.assertEqual(planner.scan.shadow_selected, width == 40.0)
                    self.assertEqual(planner.scan.shadow_candidate is not None, width == 40.0)
                    self.assertEqual(planner.counts, (0, 2))
                    self.assertAlmostEqual(planner.cache.center[planner.cache.confirmed][-1, 0], confirmed_end)

    def test_shadow_only_builds_measured_delaunay_midpoints_in_the_near_lane(self):
        planner = ShadowOnlyPlanner(ShadowOnlyConfig(colour_mode='blue_left'))
        stations = np.arange(-2, 17, 2)
        blue = [(x, 1.5) for x in stations] + [(x + 0.5, 2.0) for x in stations]
        yellow = [(x + 1, -1.5) for x in stations[:-1]] + [(x + 0.5, 5.0) for x in stations]
        planner.observe(blue, yellow, 1.0, True)
        result = planner.plan((0.0, 0.0), 0.0, 1.0)
        self.assertGreater(result.remaining, 10.0)
        lwall, rwall = planner.scan.walls
        mesh, _ = cone_links(np.vstack((lwall, rwall)), planner.cfg.width_max)
        self.assertEqual(len(result.points), len(result.point_sources))
        self.assertGreater(len(result.points), len(planner.cache.center))
        for center, left, right in zip(planner.cache.center, planner.cache.left, planner.cache.right):
            i = np.flatnonzero((lwall == left).all(axis=1))[0]
            j = np.flatnonzero((rwall == right).all(axis=1))[0] + len(lwall)
            self.assertIn((i, j), mesh)
            np.testing.assert_array_equal(center, (left + right) / 2)
        np.testing.assert_allclose(result.points[:, 1], 0.0)

    def test_shadow_only_keeps_the_same_colour_distance_limit(self):
        planner = ShadowOnlyPlanner(ShadowOnlyConfig(boundary_max_edge=3.0))
        planner.observe([(x, 1.5) for x in (0, 2, 6, 8, 10)],
                        [(x, -1.5) for x in range(0, 11, 2)], 1.0, True)
        planner.plan((0.0, 0.0), 0.0, 1.0)
        np.testing.assert_array_equal(planner.scan.walls[0], [(0.0, 1.5), (2.0, 1.5)])

    def test_shadow_only_provisional_tail_stops_at_three_metres(self):
        self.assertEqual(ShadowOnlyPlanner().cfg.provisional_length, 3.0)
        cfg = ShadowOnlyConfig(provisional_length=6.0)
        base = straight_corridor(0.0, 3.0, 1.5)
        wall = np.array([(x, -1.5) for x in range(0, 13)])
        stamps = np.ones(len(wall))
        merged = provisional_tail(base, wall, stamps, -1, 3.0, cfg)
        self.assertAlmostEqual(merged.arc[-1] - base.arc[-1], 3.0)
        self.assertIs(provisional_tail(merged, wall, stamps, -1, 3.0, cfg), merged)

    def test_shadow_only_tries_shadow_points_from_the_first_active_segment(self):
        planner = ShadowOnlyPlanner()
        planner.cache = base = straight_corridor(-2.0, 4.0, 1.5)
        planner.progress = 2.125
        planner.shadow_candidate = straight_corridor(-1.0, 7.0, 1.5)
        car, heading = np.array([0.125, 0.0]), np.array([1.0, 0.0])
        planner.scan = Scan(car, (car, heading), 1.0, False)
        attempts = []

        def reject_first_targets(start, direction, target, tangent, cfg):
            np.testing.assert_array_equal(start, [0.25, 0.0])
            np.testing.assert_array_equal(direction, heading)
            attempts.append(target[0])
            if target[0] >= 1.25:
                return connector_curve(start, direction, target, tangent, cfg)
            return None

        with patch('pathplanning.pp_hybrid_turn.connector_curve', side_effect=reject_first_targets):
            planner._merge_shadow(np.empty((0, 2)), None)
        np.testing.assert_array_equal(attempts, [0.5, 0.75, 1.0, 1.25])
        np.testing.assert_array_equal(planner.cache.center[:10], base.center[:10])
        np.testing.assert_array_equal(planner.cache.center[13], [1.25, 0.0])
        self.assertEqual(np.count_nonzero(planner.cache.sources == 'shadow_spline'), 3)
        self.assertEqual(planner.scan.shadow_outcome, 'merged')

    def test_probe_uses_path_tangent_at_the_origin_in_track_and_search(self):
        planner = HybridTurnPlanner()
        center = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 2.0], [7.0, 3.0]])
        planner.cache = Corridor(center, center, center, np.ones(4), np.ones(4, dtype=bool))
        car, heading = np.zeros(2), np.array([0.0, -1.0])
        origin, tangent = planner._probe(car, heading)
        np.testing.assert_allclose((origin, tangent), sample_at(planner.cache, 5.0))

        planner.state = 'search'
        planner.held_probe = (origin, np.array([1.0, 0.0]))
        np.testing.assert_allclose(planner._probe(car, -heading), (origin, tangent))

        planner.progress = 1.0
        np.testing.assert_allclose(planner._probe(car, -heading), sample_at(planner.cache, 6.0))
        planner.progress = 4.0
        np.testing.assert_allclose(planner._probe(car, -heading), sample_at(planner.cache, planner.cache.arc[-1]))

    def test_short_confirmed_path_triggers_with_two_cones_on_each_side(self):
        config = HybridConfig(turn_min_confirmed_path_m=11.0)
        short = [[(x, y) for x in range(-2, 11, 2)] for y in (1.5, -1.5)]
        long = [[(x, y) for x in range(-2, 15, 2)] for y in (1.5, -1.5)]
        for observed, expected in ((short, 'tls_turn'), (long, 'straight')):
            planner = HybridTurnPlanner(config)
            planner.observe(*short, 1.0, True)
            planner.plan((0, 0), 0, 1.0)
            planner.observe(*observed, 1.2, True)
            result = planner.plan((0, 0), 0, 1.2)
            self.assertGreaterEqual(min(result.cone_counts), 2)
            self.assertEqual(result.planning_mode, expected)
            self.assertEqual(planner.scan.tested_yaws, (0.0,))
            self.assertEqual(planner.scan.shadow_bands_deg, ())

    def test_recovers_on_two_healthy_updates_even_if_counts_differ(self):
        planner = HybridTurnPlanner()
        planner.counts = (3, 1)
        planner._update_turn_mode(None, 10.0, 1.0)
        planner.counts = (5, 2)
        walls = [(np.array([(x, y) for x in range(4)]), np.full(4, 1.0)) for y in (1.5, -1.5)]
        planner.revision = 1
        planner._update_turn_mode(walls, 5.0, 1.2)
        planner._update_turn_mode(walls, 5.0, 1.2)
        self.assertTrue(planner.turn_active)
        planner.revision = 2
        planner._update_turn_mode(walls, 5.0, 1.3)
        self.assertFalse(planner.turn_active)

    def test_provisional_suffix_keeps_the_measured_startup_join(self):
        planner = HybridTurnPlanner(HybridConfig(anchor_ahead=7.0, entry_gap=6.5))
        walls = [[(x, y) for x in range(6, 15, 2)] for y in (1.5, -1.5)]
        planner.observe(*walls, 1.0, True)
        startup = planner.plan((0, 0), 0, 1.0)
        self.assertGreater(startup.remaining, 3.0)
        self.assertIs(startup.provisional, True)
        confirmed = planner.cache.confirmed.copy()
        confirmed[-3:] = False
        planner.cache = replace(planner.cache, confirmed=confirmed)
        result = planner.plan((0.2, 0), 0, 1.05)
        self.assertGreater(result.remaining, 3.0)
        self.assertIs(result.provisional, True)

    def test_measured_turn_survives_dropout_and_tls_tail_expires(self):
        planner = HybridTurnPlanner()
        radius = 7.0
        angles = [i * 0.32 for i in range(-2, 8)]
        walls = [[(r * math.sin(a), radius - r * math.cos(a)) for a in angles]
                 for r in (radius - 1.5, radius + 1.5)]
        planner.observe(*walls, 1.0, True)
        straight = planner.plan((0, 0), 0, 1.0)
        self.assertEqual(straight.planning_mode, 'straight')
        planner.observe(walls[0], [], 1.2, True)
        turn = planner.plan((0, 0), 0, 1.2)
        self.assertEqual(turn.planning_mode, 'tls_turn')
        self.assertTrue(turn.provisional)
        self.assertGreater(turn.remaining, straight.remaining)
        self.assertLessEqual(turn.remaining - straight.remaining, planner.cfg.provisional_length)
        self.assertLess(planner._confirmed_ahead(None, np.zeros(2), np.array([1.0, 0.0])), turn.remaining)
        measured = planner.cache.center[planner.cache.confirmed].copy()
        self.assertTrue(planner.remembered_path_points.isdisjoint(map(tuple, planner.cache.center[~planner.cache.confirmed])))
        prior_right = planner.saved_walls[1][0].copy()
        planner.observe(walls[0], [], 1.4, True)
        planner.plan((0, 0), 0, 1.4)
        np.testing.assert_allclose(planner.saved_walls[1][0], prior_right)
        planner.observe([], [], 10.0, True)
        continued = planner.plan((0, 0), 0, 10.0)
        self.assertEqual(planner.scan.shadow_bands_deg, ((-70, -30), (30, 70)))
        self.assertEqual(planner.scan.shadow_outcome, 'no_shadow_cones')
        self.assertEqual(continued.planning_mode, 'tls_turn')
        self.assertGreater(continued.remaining, 10.0)
        self.assertFalse(continued.provisional)
        np.testing.assert_allclose(planner.cache.center, measured)
        yaw = 0.18
        car = (radius * math.sin(yaw), radius * (1 - math.cos(yaw)))
        advanced = planner.plan(car, yaw, 10.2)
        self.assertGreater(advanced.remaining, 3.0)
        self.assertLess(advanced.remaining, continued.remaining)
        np.testing.assert_allclose(advanced.points[-1], continued.points[-1])
        later = planner.plan(car, yaw, 60.0)
        np.testing.assert_array_equal(later.points, advanced.points)

    def test_shadow_only_skips_tls_with_sparse_cones_on_a_measured_turn(self):
        radius = 7.0
        angles = [i * 0.32 for i in range(-2, 8)]
        walls = [[(r * math.sin(a), radius - r * math.cos(a)) for a in angles]
                 for r in (radius - 1.5, radius + 1.5)]
        planner = ShadowOnlyPlanner()
        planner.observe(*walls, 1.0, True)
        self.assertEqual(planner.plan((0, 0), 0, 1.0).planning_mode, 'straight')
        planner.observe(walls[0], [], 1.2, True)
        with patch('pathplanning.pp_hybrid_turn.predict_wall') as predict_wall:
            result = planner.plan((0, 0), 0, 1.2)
        predict_wall.assert_not_called()
        self.assertEqual(result.planning_mode, 'shadow_turn')
        self.assertEqual(result.cone_counts, (0, 0))
        self.assertEqual(planner.scan.shadow_bands_deg, ((-70, -30), (30, 70)))
        self.assertIsNone(planner.scan.tls_candidate)
        self.assertIsNone(planner.scan.tls_added)
        self.assertNotIn('saved_boundaries', planner.cache.sources)
