import math

import torch

from controls.cone_clearance import swept_clearance


def test_projected_footprint_covers_wheels_and_respects_heading():
    poses = torch.tensor([[[0., 0., 0.], [0., 0., -math.pi / 4]]])
    result = swept_clearance(poses, torch.tensor([[.79, .90, .15]]))
    assert result[0] < 0
    assert result[1] > 0


def test_clearance_covers_motion_between_sampled_poses():
    poses = torch.tensor([[[x, 0., 0.]] for x in (-4., -2., 0., 2., 4., 6.)])
    assert swept_clearance(poses, torch.tensor([[0., 0., .15]]))[0] < 0


def test_clearance_accepts_centered_gate_pass_with_short_tail():
    poses = torch.tensor([[[x, .5, 0.]] for x in (1., 1.2, 1.4, 1.6, 1.8, 2., 2.2)])
    cones = torch.tensor([[3.2, 2.69, .16], [3.2, -1.90, .16]])
    assert swept_clearance(poses, cones)[0] > .5
