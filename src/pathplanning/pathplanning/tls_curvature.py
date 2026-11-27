"""Shared TLS wall-curvature extrapolation for the ROS and hybrid planners."""

import math

import numpy as np
from scipy.spatial import cKDTree


def predict_wall(chain, *, count=1, min_chain=4, min_radius=3.5, max_radius=40.0,
                 max_turn=45.0, decay=0.6):
    """Extrapolate curvature at the end of an ordered wall, as in TLS PCD."""
    empty = np.empty((0, 2))
    if len(chain) < max(3, min_chain):
        return empty, None
    legs = np.diff(chain, axis=0)
    lengths = np.linalg.norm(legs, axis=1)
    headings = np.arctan2(legs[:, 1], legs[:, 0])
    turn = (np.diff(headings) + np.pi) % (2 * np.pi) - np.pi
    curvature = turn / np.maximum(0.5 * (lengths[:-1] + lengths[1:]), 1e-6)
    last = float(curvature[-1])
    if abs(last) < 1.0 / max_radius:
        return empty, None
    trend = float((curvature[-1] - curvature[-2]) / max(lengths[-2], 1e-6)) if len(curvature) >= 2 else None
    step = float(np.median(lengths))
    cap = math.radians(max_turn)
    initial_heading = heading = float(headings[-1])
    point = chain[-1].copy()
    prediction = []
    kappa = 0.0
    for index in range(count):
        if trend is None:
            kappa = last * decay ** (index + 1)
        else:
            kappa = last + trend * (lengths[-1] + index * step)
            if kappa * last <= 0:
                kappa = 0.0
            limit = min(1 / min_radius, 1.5 * abs(last))
            kappa = float(np.clip(kappa, -limit, limit))
        if abs(heading + kappa * step - initial_heading) > cap:
            break
        middle = heading + 0.5 * kappa * step
        point = point + step * np.array([math.cos(middle), math.sin(middle)])
        heading += kappa * step
        prediction.append(point.copy())
    return np.asarray(prediction).reshape(-1, 2), kappa if prediction else None


def guard_predictions(left, left_prediction, left_curvature, right, right_prediction, right_curvature, width):
    """Apply TLS checks for opposing curvature and predicted wall crossings."""
    if (len(left_prediction) and len(right_prediction) and left_curvature is not None
            and right_curvature is not None and left_curvature * right_curvature < 0
            and min(abs(left_curvature), abs(right_curvature)) > 1 / 40):
        return np.empty((0, 2)), np.empty((0, 2))

    def keep(prediction, other):
        if not len(prediction) or not len(other):
            return prediction
        within = cKDTree(other).query(prediction)[0] > 0.6 * width
        return prediction[np.cumprod(within).astype(bool)]

    return (keep(left_prediction, np.vstack((right, right_prediction))),
            keep(right_prediction, np.vstack((left, left_prediction))))
