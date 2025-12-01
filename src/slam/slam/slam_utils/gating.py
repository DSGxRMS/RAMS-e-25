#!/usr/bin/env python3
# slam_utils/mahalanobis_gate.py
#
# Simple Mahalanobis gating with yaw-rate–dependent ellipse scaling.
#
# - Landmarks live in MAP frame: mean (2,), cov (2x2), class_id.
# - Measurements z live in MAP frame: (2,) each.
# - We inflate the "measurement covariance" based on |yawrate|:
#       scale = clamp(1 + k_yawrate * |yawrate|, scale_min, scale_max)
#   and build an anisotropic ellipse aligned with the car's heading.

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class GateConfig:
    # 2D chi^2 gate: 9.21 ≈ 99% level
    chi2_thr: float = 9.21

    # Base 1-σ footprint in lane frame (before chi^2)
    # These are tuned for your use-case:
    #   ~0.25 m along track, ~0.15 m across track on straights,
    #   then scaled on curves.
    sigma_along_base: float = 0.25
    sigma_across_base: float = 0.15

    # Yaw-rate scaling: scale = 1 + k_yawrate * |yawrate|
    # (yawrate in rad/s)
    k_yawrate: float = 2.5
    scale_min: float = 1.0
    scale_max: float = 2.5


@dataclass
class Landmark:
    mean: np.ndarray      # shape (2,)
    cov: np.ndarray       # 2x2
    class_id: int         # 0=blue,1=yellow,2=orange,3=big, etc.


def _build_meas_cov(yaw: float, yawrate: float, cfg: GateConfig) -> np.ndarray:
    """
    Build a measurement covariance S_meas in MAP frame, anisotropic and
    scaled with |yawrate|.

    - Along-track axis ~ car heading (cos(yaw), sin(yaw)).
    - Across-track axis is the normal.
    """
    # Dynamic scale based on curvature (yaw-rate)
    scale = 1.0 + cfg.k_yawrate * abs(yawrate)
    scale = max(cfg.scale_min, min(cfg.scale_max, scale))

    sigma_along = cfg.sigma_along_base * scale
    sigma_across = cfg.sigma_across_base * scale

    # Rotation matrix from lane frame (along/across) to world/MAP
    c = math.cos(yaw)
    s = math.sin(yaw)
    R_lane = np.array([[c, -s],
                       [s,  c]], dtype=float)

    S_lane = np.diag([sigma_along ** 2, sigma_across ** 2])
    S_world = R_lane @ S_lane @ R_lane.T
    return S_world


def gate_measurements(
    meas_xy: List[np.ndarray],
    landmarks: List[Landmark],
    yaw: float,
    yawrate: float,
    cfg: GateConfig,
) -> Tuple[List[int], List[int]]:
    """
    Mahalanobis gating:

    Inputs:
      - meas_xy: list of measurement positions (2D, in MAP frame)
      - landmarks: existing landmarks
      - yaw, yawrate: car state at this frame (MAP frame yaw, rad; yawrate, rad/s)
      - cfg: GateConfig

    Returns:
      - associated_idxs: list of indices into meas_xy that were considered
                         matches to some landmark (within chi^2 gate)
      - new_idxs: list of measurement indices that had NO landmark within gate
                  => treat these as *new* landmarks

    NOTE:
    - This is deliberately simple: we only decide "any landmark within gate or not".
      Higher-level association (which landmark exactly) can be done separately if needed.
    """
    if not meas_xy:
        return [], []

    S_meas = _build_meas_cov(yaw, yawrate, cfg)

    # Precompute inverse for speed when adding to each landmark
    associated_idxs = []
    new_idxs = []

    for i, z in enumerate(meas_xy):
        z = np.asarray(z, dtype=float).reshape(2,)
        best_m2 = None

        for lm in landmarks:
            # Effective covariance for gating: landmark cov + measurement cov
            S = lm.cov + S_meas
            try:
                Sinv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                Sinv = np.linalg.pinv(S)

            innov = z - lm.mean
            m2 = float(innov.T @ Sinv @ innov)

            if best_m2 is None or m2 < best_m2:
                best_m2 = m2

        if (best_m2 is None) or (best_m2 > cfg.chi2_thr):
            # too far from all landmarks -> new landmark
            new_idxs.append(i)
        else:
            associated_idxs.append(i)

    return associated_idxs, new_idxs
