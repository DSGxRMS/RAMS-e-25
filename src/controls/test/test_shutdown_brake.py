"""Regression tests for shutdown using a simulated clock and odometry."""
from controls import shutdown_brake


class Sim:
    def __init__(self, monkeypatch, sample):
        self.wall = 0.0
        self.commands = []
        self.sample = sample
        monkeypatch.setattr(shutdown_brake.time, "monotonic", lambda: self.wall)
        monkeypatch.setattr(shutdown_brake.time, "sleep", self.advance)
        monkeypatch.setattr(shutdown_brake, "TIMEOUT_S", 1.0)

    def advance(self, seconds):
        self.wall += seconds

    def clock(self):
        return round(self.wall * 1e9)

    def publish(self):
        stamp = self.clock()
        self.commands.append(stamp)
        return stamp

    def brake(self):
        return shutdown_brake.brake_to_stop(
            self.clock, lambda: self.sample(self.clock()), self.publish, lambda: True, .2)


def test_first_brake_is_immediate_and_stop_needs_fresh_dwell(monkeypatch):
    sim = Sim(monkeypatch, lambda stamp: (stamp, 2.0 if stamp < 300_000_000 else 0.0))
    result = sim.brake()
    assert sim.commands[0] == result["first_brake_sim_ns"] == 0
    assert result["status"] == "confirmed"
    assert result["standstill_confirmed_sim_ns"] == 500_000_000
    assert len(sim.commands) == 11


def test_repeated_zero_stamp_does_not_confirm_standstill(monkeypatch):
    sim = Sim(monkeypatch, lambda stamp: (300_000_000, 0.0) if stamp >= 300_000_000 else None)
    assert sim.brake()["status"] == "timeout"


def test_stale_or_sparse_samples_do_not_confirm_standstill(monkeypatch):
    sim = Sim(monkeypatch, lambda stamp: (stamp // 200_000_000 * 200_000_000, 0.0))
    assert sim.brake()["status"] == "timeout"


def test_confirmation_waits_through_command_delay_at_rest(monkeypatch):
    sim = Sim(monkeypatch, lambda stamp: (stamp, 0.0))
    assert sim.brake()["standstill_confirmed_sim_ns"] == 400_000_000


def test_paused_clock_sends_one_request_then_times_out(monkeypatch):
    sim = Sim(monkeypatch, lambda stamp: (stamp, 0.0))
    sim.clock = lambda: 5
    assert sim.brake()["status"] == "timeout"
    assert sim.commands == [5]


def test_clock_reset_prevents_confirmation(monkeypatch):
    sim = Sim(monkeypatch, lambda stamp: (stamp, 0.0))
    original_clock = sim.clock
    sim.clock = lambda: original_clock() if sim.wall < .35 else 0
    assert sim.brake()["status"] == "clock_reset"
