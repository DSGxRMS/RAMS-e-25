import math
from unittest.mock import patch

import numpy as np
import torch

from controls.cone_clearance import (
    ClearanceSpeedGovernor, clearance_log_penalty, cone_clearance_enabled,
    measured_footprint_gap, swept_clearance, swept_clearance_with_escape,
)


def test_projected_footprint_covers_wheels_and_respects_heading():
    poses = torch.tensor([[[0., 0., 0.], [0., 0., -math.pi / 4]]])
    result = swept_clearance(poses, torch.tensor([[.79, .90, .15]]))
    assert result[0] < 0
    assert result[1] > 0


def test_tapered_nose_clears_the_previous_false_overlap():
    stopped_pose_cone = (1.36023, -.68876, .15)
    assert measured_footprint_gap([stopped_pose_cone]) > .10
    assert measured_footprint_gap([(1.82, 0., .15)]) < 0


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


def test_escape_checks_progress_and_reentry_for_the_same_cone():
    # Both starts are in the buffer. The lower y case also begins in contact
    # with the planar footprint.
    paths = [
        [0., -.01, -.02, -.04, -.07, -.15],    # moves away and clears
        [0., -.005, -.01, -.02, -.025, -.03],  # measurable progress inside the buffer
        [0., .01, .02, -.04, -.07, -.15],      # deepens overlap before moving away
        [0., 0., 0., 0., 0., 0.],             # no progress
        [0., -.09, -.15, -.04, -.01, -.15],    # leaves the buffer and re-enters
    ]
    poses = torch.zeros(6, len(paths), 3)
    poses[..., 1] = torch.tensor(paths).T
    for cone_y in (.90, .82):
        margins, escaping = swept_clearance_with_escape(poses, torch.tensor([[0., cone_y, .15]]))
        assert (margins <= 0).all()
        assert escaping.tolist() == [True, True, False, False, False]


def test_escape_requires_clearance_from_each_other_cone():
    poses = torch.zeros(6, 1, 3)
    poses[:, 0, 1] = torch.linspace(0, -.25, 6)
    first = torch.tensor([[0., .90, .15]])
    second = torch.tensor([[0., -1.03, .15]])
    _, allowed = swept_clearance_with_escape(poses, first)
    assert allowed.item()
    _, allowed = swept_clearance_with_escape(poses, torch.cat((first, second)))
    assert not allowed.item()
    # A second cone already inside the buffer also has to improve on its own.
    _, allowed = swept_clearance_with_escape(poses, torch.tensor([[0., .90, .15], [0., -.90, .15]]))
    assert not allowed.item()


def test_clearance_penalty_strongly_discounts_small_positive_margins():
    penalty = clearance_log_penalty(torch.tensor([.01, .10, .25, .50]))
    weights = torch.exp(-penalty)
    assert 0 < weights[0] < .01 < weights[1] < weights[2] == weights[3] == 1
    overlaps = clearance_log_penalty(torch.tensor([-.15, -.05, 0.0]))
    assert overlaps[0] > overlaps[1] > overlaps[2]


def test_clearance_speed_tightens_quadratically_without_compounding():
    governor = ClearanceSpeedGovernor(4.0)
    governor.observe(80.0, .05)
    assert governor.cap_mps == 4.0
    governor.observe(32.5, .05)
    assert governor.cap_mps == 2.125
    governor.observe(32.5, .05)
    assert governor.cap_mps == 2.125
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


def test_clearance_default_is_specific_to_exp13(monkeypatch):
    monkeypatch.delenv("MPPI_CONE_CLEARANCE", raising=False)
    assert cone_clearance_enabled("exp13")
    assert not cone_clearance_enabled("exp10")
    monkeypatch.setenv("MPPI_CONE_CLEARANCE", "0")
    assert not cone_clearance_enabled("exp13")


def test_clearance_crawl_retimes_rollouts_and_preserves_path_stops(monkeypatch):
    from controls.mppi_controller import MPPIController

    class SmallBatchController(MPPIController):
        K = 8

    monkeypatch.delenv("MPPI_CONE_CLEARANCE", raising=False)
    monkeypatch.delenv("MPPI_CLEARANCE_SPEED", raising=False)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with patch.dict("os.environ", {"MPPI_MODEL": "exp13", "MPPI_SEED": "42",
                                       "MPPI_AUDIT": "1", "MPPI_CSV": "",
                                       "MPPI_STEERING_ACTUATOR": "1", "MPPI_FORCE_STEER0": "0"}):
            controller = SmallBatchController(force_cpu=True)
            controller.update_path([(float(x), 0.0) for x in range(25)])
            controller._vr_lp = controller.V_MAX
            prediction = (0.0, np.zeros(48, dtype=np.float32), .04, True)

            def solve(t, speed, cones):
                return controller.compute(
                    0.0, 0.0, 0.0, speed, .05, one_sided=False,
                    steering_prediction=prediction, steering_state_s=t, clearance_cones=cones)

            _, _, info = solve(1.0, 4.0, [(4.7, .15, .15)])
            audit = info["audit"]
            assert audit["clearance_initial_feasible_pct"] == 0.0
            assert audit["clearance_feasible_pct"] > 0.0
            assert audit["clearance_selected_margin_m"] > 0.0
            assert audit["clearance_rollout_target_mps"] == info["target_speed"] == .5
            assert max(audit["rollout_speed_targets"]) <= .5
            assert math.isclose(audit["reference"][1][0] - audit["reference"][0][0],
                                .5 * .025, abs_tol=.002)
            assert controller._vr_lp == controller.V_MAX
            assert audit["clearance_speed_next_cap_mps"] == .5

            obstacle = [(1.65, .15, .15)]
            for t in (1.05, 1.10):
                steer, acceleration, crawl = solve(t, 0.0, obstacle)
                audit = crawl["audit"]
                assert crawl["clearance_crawl"] and not crawl["clearance_blocked"]
                assert crawl["target_speed"] == audit["clearance_rollout_target_mps"] == .5
                assert audit["clearance_speed_next_cap_mps"] == .5
                assert audit["clearance_feasible_count"] == 0
                assert audit["clearance_infeasible_pct"] == 100.0
                assert audit["clearance_decision"] == "crawl"
                assert audit["clearance_selected_margin_m"] < 0.0
                assert steer == audit["selected_steer_sequence"][0]
                assert max(audit["rollout_speed_targets"]) <= .5
                assert "clearance_retry_target_mps" not in audit
                assert acceleration > 0.0

            _, _, away = solve(1.15, .5, [(-1.20, 0., .15)])
            assert away["audit"]["clearance_decision"] == "escape"
            assert away["audit"]["clearance_escape_count"] > 0
            assert away["audit"]["clearance_feasible_pct"] == 0.0
            assert away["target_speed"] == away["audit"]["clearance_speed_next_cap_mps"] == .5

            _, _, next_step = solve(1.20, .5, [])
            assert not next_step["clearance_crawl"]
            assert next_step["target_speed"] == .5
            assert math.isclose(next_step["audit"]["clearance_speed_next_cap_mps"], .575)

            _, acceleration, stopped = solve(1.25, .8, None)
            assert stopped["clearance_blocked"] and stopped["target_speed"] == 0.0
            assert math.isclose(acceleration, -1.2)

            controller.update_path([(0., 0.), (.7, 0.)])
            _, _, end_of_path = solve(1.30, 0.0, obstacle)
            assert end_of_path["clearance_blocked"] and not end_of_path["clearance_crawl"]
            assert end_of_path["target_speed"] == 0.0
    finally:
        torch.set_num_threads(threads)
