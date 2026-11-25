import math

import numpy as np

from controls.path_reference import OrderedReference, project_reference


def test_ordered_intake_expires_and_keeps_recovery_separate():
    pose = (0, 0, 0)
    path = ((0, 0), (2, 0), (4, 1), (6, 1), (8, 0))
    recovery = ((0, 1), (2.5, 1), (5, 1))
    ref = OrderedReference()
    ref.accept_planner(path, "map", 1.0, 5)
    ref.accept_planner(path[:2], "map", 1.2, 5)
    ref.accept_planner((*path[:-1], (math.nan, 1)), "map", 1.2, 5)
    ref.accept_planner(path[::-1], "odom", 1.2, 5)
    ref.accept_planner(path[::-1], "map", 0.9, 5)
    assert ref.accept_recovery(recovery, 1.25)
    assert ref.planner_path(1.4, pose) == path
    assert ref.recovery_path(1.4, pose) == recovery
    assert not ref.planner_path(1.6, pose)
    assert not ref.recovery_path(2.8, pose)
    ref.clear()
    assert not ref.planner_path(1.4, pose)
    assert not ref.recovery_path(1.4, pose)


def test_recovery_needs_remaining_forward_distance():
    ref = OrderedReference()
    assert ref.accept_recovery(((0, 0), (5, 0)), 1.0)
    assert ref.recovery_path(1.2, (0, 0, 0))
    assert not ref.recovery_path(1.3, (3, 0, 0))


def test_projection_uses_heading_progress_and_terminal_tangent():
    path = np.array([[-3., 0.], [3., 0.], [3., -3.], [0., -3.], [0., 3.]])
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    assert project_reference(0., 0., 0., path, arc)[0] == 0
    assert project_reference(0., 0., math.pi/2, path, arc)[0] == 3
    assert project_reference(0., 0., math.pi/2, path, arc, 3., 2.)[0] == 0
    idx, progress, cte, heading = project_reference(0., 4., math.pi/2, path, arc)
    assert idx == 3
    assert progress == arc[-1]
    assert abs(cte) < 1e-9
    assert heading == math.pi/2
