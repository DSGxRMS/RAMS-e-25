import pytest
import numpy as np

from controls.mppi_controller import MPPIController


def test_preview_includes_interior_apex_and_keeps_lower_current_limit():
    mppi = MPPIController.__new__(MPPIController)
    distances = np.arange(5, dtype=float)
    speeds = np.array([6., 5., 3., 4., 6.])
    mppi.corner_preview = False
    assert mppi._corner_speed(0., 6., distances, speeds) == 6.
    mppi.corner_preview = True
    assert mppi._corner_speed(0., 6., distances, speeds) == 3.
    assert mppi._corner_speed(2., 6., distances, speeds) == 3.
    assert mppi._corner_speed(0., .75, distances, speeds) == 5.5
    assert mppi._corner_speed(0., 0., distances, speeds) == 6.


@pytest.mark.parametrize("sign", [-1., 1.])
def test_tight_arc_speed_uses_supported_curvature_and_keeps_steering_clip(sign):
    mppi = MPPIController.__new__(MPPIController)
    mppi.dev = "cpu"
    mppi.V_MAX = 6.
    angles = np.linspace(0, np.pi / 2, 40)
    path = list(zip(2.8 * np.sin(angles), sign * 2.8 * (1 - np.cos(angles))))
    for preview in (False, True):
        mppi.corner_preview = preview
        mppi._path_key = None
        mppi.update_path(path)
        profile = mppi._path_data["vt_np"]
        if preview:
            assert np.median(profile) < 2.8
        else:
            assert np.min(profile) == pytest.approx(np.sqrt(2.5 / .3))
        assert mppi._path_data["ff"].abs().max() <= np.arctan(mppi.L_WB_KIN * .3) + 1e-6


def controller(enabled=True):
    mppi = MPPIController.__new__(MPPIController)
    mppi.corner_braking = enabled
    mppi._vr_lp = 5.99
    mppi.I_spd = 0.0
    return mppi


def test_boa_entry_brakes_without_waiting_for_the_speed_filter():
    mppi = controller()
    target = mppi._speed_target(2.88, None, 0.04)
    accel = mppi._longitudinal_effort(target - 4.74, 0.04) * mppi.ACCEL_MAX
    assert target == 2.88
    assert accel == pytest.approx(-2.79)


def test_positive_integral_cannot_weaken_braking():
    mppi = controller()
    mppi.I_spd = 1.5
    assert mppi._longitudinal_effort(-0.1, 0.05) == pytest.approx(-0.05)


def test_speed_rebound_uses_elapsed_simulation_time():
    mppi = controller()
    mppi._speed_target(2.88, None, 0.04)
    assert mppi._speed_target(6.0, None, 0.1) == pytest.approx(3.18)
    assert mppi._speed_target(3.2, None, 0.1) == pytest.approx(3.2)


@pytest.mark.parametrize("enabled", [False, True])
def test_recovery_cap_is_still_hard(enabled):
    mppi = controller(enabled)
    assert mppi._speed_target(1.5, 1.5, 0.04) == 1.5


def test_disabled_switch_reproduces_boa_target_and_throttle():
    mppi = controller(False)
    target = mppi._speed_target(2.88, None, 0.04)
    accel = mppi._longitudinal_effort(target - 4.66, 0.04) * mppi.ACCEL_MAX
    assert target == pytest.approx(5.5235)
    assert accel == pytest.approx(1.29525)
