import numpy as np
import pytest

from controls.steering_actuator import (
    SteeringActuator, steering_actuator_enabled, steering_state_time, wheel_at_or_before,
)


def test_pending_commands_are_replayed_before_a_new_candidate():
    actuator = SteeringActuator()
    actuator.observe_command(0.04, 0.2, 0, solved=True)
    actuator.observe_command(0.08, -0.2, 0.04, solved=True)
    initial, queued, solve, measured = actuator.prediction(0.2, (0.19, 0), .005, 200)
    assert initial == 0
    assert measured
    assert solve == pytest.approx(.04)
    assert len(queued) == 48
    assert np.all(queued[:8] == 0)
    assert queued[8] == .2
    assert queued[-1] == -.2
    initial, _, _, _ = actuator.prediction(.27, (.25, .01), .005, 200)
    assert initial == pytest.approx(.01 + .02 * 1.04)


def test_stale_feedback_uses_history_and_reset_drops_old_commands():
    actuator = SteeringActuator()
    actuator.observe_command(0, .3, 0, solved=False)
    initial, _, solve, measured = actuator.prediction(.21, (0, .5), .005, 200)
    assert initial == pytest.approx(.01 * 1.04)
    initial, _, _, _ = actuator.prediction(.26, (0, .5), .005, 200)
    assert initial == pytest.approx(.06 * 1.04)
    assert solve == pytest.approx(.04)
    assert not measured
    actuator.reset()
    initial, queued, _, measured = actuator.prediction(.10, (.20, .5), .005, 200)
    assert initial == 0
    assert not measured
    assert not queued.any()


def test_fresh_applied_angle_seeds_first_solve_with_no_command_history():
    actuator = SteeringActuator()
    initial, queued, _, measured = actuator.prediction(1.0, (.98, -.1), .005, 200)
    assert initial == pytest.approx(-.1)
    assert (queued == -.1).all()
    assert measured


def test_exp13_default_and_baseline_override(monkeypatch):
    monkeypatch.delenv("MPPI_STEERING_ACTUATOR", raising=False)
    assert steering_actuator_enabled("exp13")
    assert not steering_actuator_enabled("exp10")
    monkeypatch.setenv("MPPI_STEERING_ACTUATOR", "0")
    assert not steering_actuator_enabled("exp13")
    monkeypatch.setenv("MPPI_STEERING_ACTUATOR", "other")
    with pytest.raises(ValueError, match="0 or 1"):
        steering_actuator_enabled("exp13")


def test_pose_stamped_ahead_of_clock_aligns_feedback_without_postponing_new_command():
    actuator = SteeringActuator()
    origin = steering_state_time(10.0, 10.025)
    feedback = wheel_at_or_before(((10.01, .1), (10.03, .3), (10.02, .2)), origin)
    assert feedback == (10.02, .2)
    angle, queued, _, measured = actuator.prediction(origin, feedback, .005, 200, decision_s=10.0)
    assert measured and angle == pytest.approx(.2)
    assert len(queued) == 43


def test_pose_and_wheel_from_another_clock_epoch_do_not_seed_rollout():
    assert steering_state_time(1.0, 20.0) == 1.0
    assert steering_state_time(20.0, 1.0) == 20.0
    assert steering_state_time(20.0, None) == 20.0
    assert wheel_at_or_before(((20.001, .2),), 20.0) is None
