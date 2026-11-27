"""Geometry and dropout regressions for the cone-memory planner."""

import math
import unittest

import numpy as np

from pathplanning.cone_memory import Config, ConeMemoryPlanner, cone_links, validate_prefix


def straight(end, stagger=0):
    blue = [(x, 1.5) for x in range(-2, end + 1, 2)]
    yellow = [(x, -1.5) for x in range(-2 + stagger, end + 1, 2)]
    return blue, yellow


class ConeMemoryTest(unittest.TestCase):
    def test_cone_links_include_3m_and_exclude_longer_edges(self):
        edges, triangulated = cone_links(np.array([(0, 0), (3, 0), (0, 3.001)]))
        self.assertTrue(triangulated)
        self.assertEqual(edges, {(0, 1)})
        planner = ConeMemoryPlanner(Config(provisional_length=0.0))
        blue, yellow = straight(16)
        blue = [point for point in blue if point[0] not in (4, 6)]
        planner.observe(blue, yellow, 1.0, True)
        result = planner.plan((0, 0), 0, 1.0)
        self.assertAlmostEqual(result.points[-1, 0], 2.0)

    def test_staggered_cross_sections_and_cd(self):
        planner = ConeMemoryPlanner()
        planner.observe(*straight(16, stagger=1), 1.0, True)
        result = planner.plan((0, 0), 0, 1.0)
        self.assertEqual(result.state, 'track')
        self.assertGreater(result.remaining, 10.0)
        np.testing.assert_allclose(result.points[:, 1], 0.0, atol=1e-6)
        np.testing.assert_allclose(result.probe[0], [2.5, 0])
        np.testing.assert_allclose(result.probe[1], [1, 0])
        self.assertEqual(result.anchors, ((2, 1.5), (3, -1.5)))

    def test_small_track_profile_joins_a_wide_corridor_ahead_of_spawn(self):
        config = Config(boundary_max_edge=4.0, anchor_ahead=7.0, entry_gap=6.5,
                        nominal_width=4.8, width_max=5.5, connector_max_lead=5.0)
        planner = ConeMemoryPlanner(config)
        blue = [(6.1, 2.776), (8.934, 3.064), (12.869, 3.013)]
        yellow = [(6.09, -1.855), (8.983, -1.797), (12.94, -1.924)]
        # This cone is on the old approach segment but outside the published centerline.
        planner.observe(blue, yellow, 1.0, True, orange=[(2.0, 0.0)])
        result = planner.plan((0, 0), 0, 1.0)
        self.assertTrue(result.provisional)
        self.assertGreater(result.points[-1, 0], 12.0)
        np.testing.assert_array_equal(result.points, planner.cache.center)
        self.assertGreater(result.points[0, 0], 6.0)
        planner.reset()
        planner.observe(blue, yellow, 1.0, True)
        self.assertEqual(planner.plan((-1, 0), 0, 1.0).state, 'no_valid_path')

    def test_startup_sees_mapped_cones_outside_virtual_fov(self):
        planner = ConeMemoryPlanner(Config(fov_angle=10.0, search_yaw=0.0))
        self.assertEqual(planner.plan((0, 0), 0, 0.5).state, 'no_valid_path')
        self.assertFalse(planner.startup_scan_complete)
        planner.observe(*straight(16), 1.0, True)
        self.assertGreater(planner.plan((0, 0), 0, 1.0).remaining, 12.0)
        self.assertTrue(planner.startup_scan_complete)
        normal = ConeMemoryPlanner(planner.cfg)
        normal.startup_scan_complete = True
        normal.observe(*straight(16), 1.0, True)
        self.assertLess(normal.plan((0, 0), 0, 1.0).remaining, 6.0)

    def test_one_cone_total_dropout_and_recovery(self):
        planner = ConeMemoryPlanner()
        blue, yellow = straight(12)
        planner.observe(blue, yellow, 1.0, True)
        initial = planner.plan((0, 0), 0, 1.0)
        planner.observe([(8, 1.5)], yellow, 1.2, True)
        sparse = planner.plan((0.5, 0), 0, 1.2)
        self.assertEqual(sparse.state, 'search')
        self.assertEqual(sparse.cone_counts[0], 1)
        np.testing.assert_allclose(sparse.points[-1], initial.points[-1])
        planner.observe([], [], 1.4, True)
        dropout = planner.plan((1, 0), 0, 1.4)
        np.testing.assert_allclose(dropout.probe[0], sparse.probe[0])
        np.testing.assert_allclose(dropout.points[-1], sparse.points[-1])
        self.assertLess(dropout.remaining, sparse.remaining)
        for i, now in enumerate((1.6, 1.8, 2.0)):
            planner.observe(*straight(18), now, True)
            recovered = planner.plan((1.5 + i * 0.5, 0), 0, now)
        self.assertEqual(recovered.state, 'track')
        self.assertGreater(recovered.points[-1, 0], initial.points[-1, 0])

    def test_provisional_tail_uses_one_cumulative_cap(self):
        planner = ConeMemoryPlanner()
        planner.observe(*straight(8), 1.0, True)
        initial = planner.plan((0, 0), 0, 1.0)
        for now in (1.2, 1.4, 1.6):
            planner.observe([], straight(16)[1], now, True)
            result = planner.plan((0.5, 0), 0, now)
            self.assertTrue(result.provisional)
            self.assertLessEqual(result.points[-1, 0], initial.points[-1, 0] + 2.0 + 1e-8)
        self.assertGreater(result.points[-1, 0], initial.points[-1, 0])

    def test_identical_map_messages_do_not_reinstate_expired_cones(self):
        planner = ConeMemoryPlanner(Config(max_age=1.0, width_age=0.5))
        planner.observe(*straight(16), 1.0)
        initial = planner.plan((0, 0), 0, 1.0)
        self.assertEqual(initial.state, 'track')
        for now in (2.2, 2.4, 2.6):
            planner.observe(*straight(16), now)
            result = planner.plan((0, 0), 0, now)
            self.assertEqual(result.state, 'search')
            self.assertEqual(result.cone_counts, (0, 0))
            np.testing.assert_array_equal(result.points, initial.points)

    def test_full_run_memory_keeps_trimmed_and_blocked_centerline_points(self):
        planner = ConeMemoryPlanner(Config(max_age=1.0, width_age=0.5))
        planner.observe(*straight(16), 1.0, True)
        initial = planner.plan((0, 0), 0, 1.0)
        saved = set(map(tuple, planner.cache.center))
        planner.observe([], [], 3.0, True)
        for x, now in ((1.0, 3.0), (2.0, 4.0), (3.0, 10.0)):
            result = planner.plan((x, 0), 0, now)
            self.assertEqual(result.cone_counts, (0, 0))
            np.testing.assert_array_equal(result.points[-1], initial.points[-1])
        self.assertFalse(planner.cones)
        self.assertGreater(planner.cache.center[0, 0], 0)
        self.assertEqual(planner.remembered_path_points, saved)

        planner.observe([(6, 0)], [], 11.0, True)
        blocked = planner.plan((3, 0), 0, 11.0)
        self.assertLess(blocked.points[-1, 0], 6.0)
        self.assertIn((6.0, 0.0), planner.remembered_path_points)

    def test_bend_uses_finite_wall_and_publishes_the_centerline(self):
        radius = 5.0
        angles = [i * 0.24 for i in range(-2, 11)]
        walls = [[(r * math.sin(a), radius - r * math.cos(a)) for a in angles]
                 for r in (radius - 1.5, radius + 1.5)]
        planner = ConeMemoryPlanner()
        planner.observe(*walls, 1.0, True)
        result = planner.plan((0, 0), 0, 1.0)
        self.assertGreater(result.remaining, 4.0)
        self.assertGreater(result.points[-1, 1], 1.0)
        starts = np.array([[0, -1], [0, 1]])
        ends = np.array([[6, -1], [6, 1]])
        points = np.array([[0, 0], [2, 0], [4, 2]])
        self.assertEqual(validate_prefix(points, starts, ends, Config()), 2)
        planner = ConeMemoryPlanner()
        planner.observe(*straight(16), 1.0, True)
        result = planner.plan((0, 0.2), 0, 1.0)
        np.testing.assert_allclose(result.points[:, 1], 0.0, atol=1e-6)
        np.testing.assert_array_equal(result.points, planner.cache.center[-len(result.points):])

    def test_pose_jump_invalidates_prior_branch(self):
        planner = ConeMemoryPlanner()
        planner.observe(*straight(16), 1.0, True)
        self.assertTrue(len(planner.plan((0, 0), 0, 1.0).points))
        remembered = planner.remembered_path_points.copy()
        self.assertEqual(planner.plan((0, 5), 0, 1.2).state, 'no_valid_path')
        self.assertEqual(planner.plan((0, 5), 0, 1.4).state, 'no_valid_path')
        self.assertEqual(planner.remembered_path_points, remembered)
        planner.plan((0, 5), 0, 0.1)
        self.assertFalse(planner.remembered_path_points)

    def test_nearby_return_leg_stays_out_of_the_active_corridor(self):
        planner = ConeMemoryPlanner()
        blue, yellow = straight(16)
        blue.extend((x + 0.5, 2.0) for x in range(-2, 17, 2))
        yellow.extend((x + 0.5, 5.0) for x in range(-2, 17, 2))
        planner.observe(blue, yellow, 1.0, True)
        result = planner.plan((0, 0), 0, 1.0)
        self.assertGreater(result.remaining, 8)
        np.testing.assert_allclose(result.points[:, 1], 0.0, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
