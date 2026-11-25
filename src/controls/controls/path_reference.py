"""Ordered reference intake and progress for the Exp13 controller."""
import math

import numpy as np


PLANNER_MAX_AGE = 0.5
RECOVERY_MAX_AGE = 1.5


def ordered_points(points, minimum):
    """Check all coordinates and remove only consecutive coincident points."""
    ordered = []
    for x, y in points:
        if not (math.isfinite(x) and math.isfinite(y)):
            return ()
        if not ordered or math.hypot(x - ordered[-1][0], y - ordered[-1][1]) > 1e-3:
            ordered.append((x, y))
    return tuple(ordered) if len(ordered) >= minimum else ()


def project_reference(x, y, yaw, points, arc, previous_s=None, forward_window=None):
    """Project onto an ordered line while preferring the forward branch."""
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    step = np.diff(arc)
    rel = np.array([x, y]) - points[:-1]
    fraction = np.clip(np.sum(rel * segments, axis=1) / np.maximum(lengths**2, 1e-12), 0, 1)
    heading = np.arctan2(segments[:, 1], segments[:, 0])
    angle = np.arctan2(np.sin(yaw - heading), np.cos(yaw - heading))
    # Break near ties by heading and then by earlier progress along the line.
    eligible = lengths > 1e-6
    if previous_s is not None:
        eligible &= (arc[1:] >= previous_s - 0.75) & (arc[:-1] <= previous_s + forward_window)
        denom = np.maximum(step, 1e-12)
        lower = np.clip((previous_s - 0.75 - arc[:-1]) / denom, 0, 1)
        upper = np.clip((previous_s + forward_window - arc[:-1]) / denom, 0, 1)
        fraction = np.clip(fraction, lower, upper)
    progress = arc[:-1] + fraction * step
    error = rel - fraction[:, None] * segments
    score = np.linalg.norm(error, axis=1) + 0.5 * abs(angle) + 0.02 * progress
    idx = int(np.argmin(np.where(eligible, score, np.inf)))
    cte = -math.sin(heading[idx]) * error[idx, 0] + math.cos(heading[idx]) * error[idx, 1]
    return idx, float(progress[idx]), float(cte), float(heading[idx])


class OrderedReference:
    """Keep one planner message and one independently generated recovery path."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.planner = None
        self.recovery = None

    def accept_planner(self, points, frame, stamp, minimum):
        if frame != "map":
            return
        path = ordered_points(points, minimum)
        if path and (self.planner is None or stamp >= self.planner[0]):
            self.planner = (stamp, path)

    def accept_recovery(self, points, stamp):
        path = ordered_points(points, 2)
        if not path:
            return False
        self.recovery = (stamp, path)
        return True

    @staticmethod
    def _select(reference, now, max_age, pose, min_ahead):
        if reference is None or not -0.1 <= now - reference[0] <= max_age:
            return ()
        path = reference[1]
        points = np.asarray(path)
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        _, progress, _, _ = project_reference(*pose, points, arc)
        return path if arc[-1] - progress > min_ahead else ()

    def planner_path(self, now, pose):
        return self._select(self.planner, now, PLANNER_MAX_AGE, pose, min_ahead=3.0)

    def recovery_path(self, now, pose):
        return self._select(self.recovery, now, RECOVERY_MAX_AGE, pose, min_ahead=3.0)
