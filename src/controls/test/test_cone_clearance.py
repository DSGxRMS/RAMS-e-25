import math
from unittest.mock import patch

import numpy as np
import torch

from controls.cone_clearance import (
    ClearanceSpeedGovernor, clearance_log_penalty, measured_footprint_gap, swept_clearance,
)


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


def test_measured_gap_uses_the_same_footprint_without_rollout_padding():
    poses = torch.zeros(1, 1, 3)
    for cones in ([[.79, .90, .15]], [[3.2, 2.69, .16], [3.2, -1.90, .16]]):
        measured = measured_footprint_gap(cones)
        predicted = swept_clearance(poses, torch.tensor(cones))[0].item()
        assert math.isclose(measured - .10, predicted, abs_tol=1e-6)
    assert measured_footprint_gap([]) is None


def test_clearance_penalty_strongly_discounts_small_positive_margins():
    penalty = clearance_log_penalty(torch.tensor([.01, .10, .25, .50]))
    weights = torch.exp(-penalty)
    assert 0 < weights[0] < .01 < weights[1] < weights[2] == weights[3] == 1


def test_clearance_speed_tightens_quadratically_without_compounding():
    governor = ClearanceSpeedGovernor(6.0)
    governor.observe(80.0, .05)
    assert governor.cap_mps == 6.0
    governor.observe(32.5, .05)
    assert governor.cap_mps == 2.625
    governor.observe(32.5, .05)
    assert governor.cap_mps == 2.625
    governor.observe(0.0, .05)
    assert governor.cap_mps == 1.5


def test_clearance_speed_hysteresis_holds_and_recovers_at_a_bounded_rate():
    governor = ClearanceSpeedGovernor(6.0)
    governor.observe(50.0, .05)
    limited = governor.cap_mps
    governor.observe(59.0, .1)
    governor.observe(None, .1)
    assert governor.cap_mps == limited
    governor.observe(100.0, .05)
    assert math.isclose(governor.cap_mps, limited + .075)
    for _ in range(50):
        governor.observe(100.0, .10)
    assert governor.cap_mps == 6.0


def test_clearance_retry_retimes_steering_and_stop_uses_the_pi_controller():
    from controls.mppi_controller import MPPIController

    class SmallBatchController(MPPIController):
        K = 8

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with patch.dict("os.environ", {"MPPI_MODEL": "exp13", "MPPI_SEED": "42",
                                       "MPPI_AUDIT": "1", "MPPI_CSV": "",
                                       "MPPI_CONE_CLEARANCE": "1", "MPPI_CLEARANCE_SPEED": "1",
                                       "MPPI_STEERING_ACTUATOR": "1", "MPPI_FORCE_STEER0": "0"}):
            controller = SmallBatchController(force_cpu=True)
            controller.update_path([(float(x), 0.0) for x in range(25)])
            controller._vr_lp = 4.0
            prediction = (0.0, np.zeros(48, dtype=np.float32), .04, True)
            _, _, info = controller.compute(
                0.0, 0.0, 0.0, 4.0, .05, one_sided=False,
                steering_prediction=prediction, steering_state_s=1.0,
                clearance_cones=[(4.7, .15, .15)])
            audit = info["audit"]
            assert audit["clearance_initial_feasible_pct"] == 0.0
            assert audit["clearance_feasible_pct"] > 0.0
            assert audit["clearance_selected_margin_m"] > 0.0
            assert audit["clearance_rollout_target_mps"] == info["target_speed"] == 1.5
            assert max(audit["rollout_speed_targets"]) <= 1.5
            assert math.isclose(audit["reference"][1][0] - audit["reference"][0][0],
                                1.5 * .025, abs_tol=.002)
            assert math.isclose(controller._vr_lp, 5.2)

            _, acceleration, stopped = controller.compute(
                0.0, 0.0, 0.0, .8, .05, one_sided=False, clearance_cones=None)
            assert stopped["clearance_blocked"] and stopped["target_speed"] == 0.0
            assert math.isclose(acceleration, -1.2)
    finally:
        torch.set_num_threads(threads)
