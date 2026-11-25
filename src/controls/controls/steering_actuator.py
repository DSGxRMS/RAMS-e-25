"""Predict the EUFS steering queue and wheel travel in simulator time."""
from collections import deque
import math
import os

import numpy as np


def steering_actuator_enabled(model):
    default = "1" if model == "exp13" else "0"
    value = os.environ.get("MPPI_STEERING_ACTUATOR", default)
    if value not in ("0", "1"):
        raise ValueError("MPPI_STEERING_ACTUATOR must be 0 or 1")
    return value == "1"


class SteeringActuator:
    """Track requested steering and seed the rate limiter from wheel feedback."""

    DELAY_S = 0.20
    RATE_RAD_S = 1.04
    FEEDBACK_MAX_AGE_S = 0.10

    def __init__(self):
        self.reset()

    def reset(self):
        self.commands = deque()
        self.publish_delays = deque(maxlen=5)
        self.estimate = None

    def observe_command(self, sent_s, angle, decision_s, *, solved):
        self.commands.append((sent_s + self.DELAY_S, angle))
        if solved and sent_s >= decision_s:
            self.publish_delays.append(sent_s - decision_s)

    def prediction(self, now, feedback, dt, horizon):
        """Return the current angle and the fixed requests before MPPI can act.

        Each feedback sample is (simulator timestamp, applied angle).
        Recent commands represent the queue using host publication timestamps.
        """
        feedback_fresh = (feedback is not None
                          and 0 <= now - feedback[0] <= self.FEEDBACK_MAX_AGE_S)
        if feedback_fresh:
            start, angle = feedback
        elif self.estimate is not None and self.estimate[0] <= now:
            start, angle = self.estimate
        else:
            start = min(now, self.commands[0][0]) if self.commands else now
            angle = 0.0

        retained_from = min(start, now - self.FEEDBACK_MAX_AGE_S)
        while len(self.commands) >= 2 and self.commands[1][0] <= retained_from:
            self.commands.popleft()

        initial_request = angle if feedback_fresh or self.estimate is not None else 0.0
        requested = initial_request
        for activation, command in self.commands:
            if activation > now:
                break
            duration = max(activation - start, 0.0)
            angle = self._advance(angle, requested, duration)
            start = max(start, activation)
            requested = command
        angle = self._advance(angle, requested, now - start)
        self.estimate = now, angle

        solve_s = float(np.median(self.publish_delays)) if self.publish_delays else 0.04
        prefix_steps = min(horizon, math.ceil((self.DELAY_S + solve_s - 1e-9) / dt))
        query = now + np.arange(prefix_steps) * dt
        # Right-continuous zero-order hold for already-issued commands.
        if self.commands:
            times, commands = zip(*self.commands)
            indices = np.searchsorted(times, query + 1e-9, side="right")
            prefix = np.r_[initial_request, commands][indices]
        else:
            prefix = np.full(prefix_steps, initial_request)
        return angle, prefix, solve_s, feedback_fresh

    @classmethod
    def _advance(cls, angle, target, duration):
        travel = duration * cls.RATE_RAD_S
        return angle + min(max(target - angle, -travel), travel)
