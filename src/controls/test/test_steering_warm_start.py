import pytest
import torch

from controls.mppi_controller import MPPIController


def fixture_plan():
    controller = MPPIController.__new__(MPPIController)
    controller.dev = torch.device("cpu")
    controller.T_MAX = 8
    # Unscored tail deliberately contains unrelated values.
    controller.nom = torch.tensor([[0., 1., 2., 3., 4., -9., -9., -9.]])
    controller._previous_steering_plan = 1., 3, 7, 5
    return controller


def test_warm_start_advances_by_elapsed_time_with_fractional_steps_and_holds():
    controller = fixture_plan()
    shifted, elapsed = controller._warm_start(1.0225, hold_steps=2)
    assert elapsed == pytest.approx(4.5)
    assert shifted.tolist()[0] == pytest.approx([2.5, 4., 4., 4., 4., 4., 4., 4.])


def test_warm_start_aligns_new_hold_and_uses_only_previously_scored_actions():
    controller = fixture_plan()
    shifted, elapsed = controller._warm_start(1.0, hold_steps=5)
    assert elapsed == 0
    assert shifted.tolist()[0] == pytest.approx([0., 3., 4., 4., 4., 4., 4., 4.])
    shifted, _ = controller._warm_start(1.038, hold_steps=3)
    assert not torch.count_nonzero(shifted)
    controller.reset_steering_plan()
    shifted, elapsed = controller._warm_start(1.06, hold_steps=3)
    assert elapsed is None and not torch.count_nonzero(shifted)
