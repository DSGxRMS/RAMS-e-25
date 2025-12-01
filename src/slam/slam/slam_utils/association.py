# slam_utils/association.py
#
# Mahalanobis gating + Hungarian association utilities.

import math
from typing import List, Tuple

import numpy as np

# Optional: use SciPy if available; otherwise fallback to greedy
try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def chi2_quantile_2d(p: float) -> float:
    """
    Approx chi-square quantile for 2 DOF (2D gating).
    Uses Wilson–Hilferty approximation via normal quantile.
    """
    # quick normal inverse CDF
    def _norm_ppf(p_):
        if p_ <= 0.0:
            return -float("inf")
        if p_ >= 1.0:
            return float("inf")

        a1 = -3.969683028665376e+01
        a2 = 2.209460984245205e+02
        a3 = -2.759285104469687e+02
        a4 = 1.383577518672690e+02
        a5 = -3.066479806614716e+01
        a6 = 2.506628277459239e+00

        b1 = -5.447609879822406e+01
        b2 = 1.615858368580409e+02
        b3 = -1.556989798598866e+02
        b4 = 6.680131188771972e+01
        b5 = -1.328068155288572e+01

        c1 = -7.784894002430293e-03
        c2 = -3.223964580411365e-01
        c3 = -2.400758277161838e+00
        c4 = -2.549732539343734e+00
        c5 = 4.374664141464968e+00
        c6 = 2.938163982698783e+00

        d1 = 7.784695709041462e-03
        d2 = 3.224671290700398e-01
        d3 = 2.445134137142996e+00
        d4 = 3.754408661907416e+00

        plow = 0.02425
        phigh = 1.0 - plow

        if p_ < plow:
            q = math.sqrt(-2.0 * math.log(p_))
            return (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / \
                   (((((d1*q + d2)*q + d3)*q + d4)*q) + 1.0)
        if p_ > phigh:
            q = math.sqrt(-2.0 * math.log(1.0 - p_))
            return -(((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / \
                    (((((d1*q + d2)*q + d3)*q + d4)*q) + 1.0)

        q = p_ - 0.5
        r = q * q
        return (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6)*q / \
               (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0)

    k = 2  # DOF
    z = _norm_ppf(p)
    t = 1.0 - 2.0 / (9.0 * k) + z * math.sqrt(2.0 / (9.0 * k))
    return k * (t ** 3)


def build_mahalanobis_cost_matrix(
    lm_means: np.ndarray,
    lm_covs: np.ndarray,
    meas_points: np.ndarray,
    meas_sigma_xy: float,
    gate_chi2: float,
    large_cost: float = 1e6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For a SINGLE colour-class:

      lm_means: (M,2)
      lm_covs:  (M,2,2)
      meas_points: (N,2)  (in SAME frame as landmarks)

    Returns:
      cost: (N,M) matrix of m^2 or large_cost
      valid_mask: (N,M) True if inside gate
    """
    if lm_means.size == 0 or meas_points.size == 0:
        # no landmarks or no measurements
        N = meas_points.shape[0]
        M = lm_means.shape[0]
        return np.full((N, M), large_cost, float), np.zeros((N, M), bool)

    lm_means = np.asarray(lm_means, float)
    lm_covs = np.asarray(lm_covs, float)
    meas_points = np.asarray(meas_points, float)

    N = meas_points.shape[0]
    M = lm_means.shape[0]

    cost = np.full((N, M), large_cost, float)
    valid = np.zeros((N, M), bool)

    R = np.diag([meas_sigma_xy**2, meas_sigma_xy**2])

    for i in range(N):
        z = meas_points[i]
        for j in range(M):
            mu = lm_means[j]
            P = lm_covs[j]
            S = P + R
            try:
                Sinv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                Sinv = np.linalg.pinv(S)
            dz = z - mu
            m2 = float(dz.T @ Sinv @ dz)
            if m2 <= gate_chi2:
                cost[i, j] = m2
                valid[i, j] = True

    return cost, valid


def hungarian_assign(
    cost: np.ndarray,
    valid_mask: np.ndarray,
    large_cost: float = 1e6,
) -> List[Tuple[int, int, float]]:
    """
    Hungarian assignment on (N,M) cost matrix.
    Returns list of (meas_idx, lm_idx, m2) for VALID pairs only.

    If SciPy isn't available, falls back to a greedy NN per measurement.
    """
    cost = np.asarray(cost, float)
    valid_mask = np.asarray(valid_mask, bool)

    if cost.size == 0 or cost.shape[0] == 0 or cost.shape[1] == 0:
        return []

    N, M = cost.shape
    pairs: List[Tuple[int, int, float]] = []

    if _HAS_SCIPY:
        row_ind, col_ind = linear_sum_assignment(cost)
        for i, j in zip(row_ind, col_ind):
            if i < N and j < M and valid_mask[i, j] and cost[i, j] < 0.5 * large_cost:
                pairs.append((int(i), int(j), float(cost[i, j])))
        return pairs

    # Greedy fallback (no global optimality, but fine for small N,M)
    used_lm = set()
    for i in range(N):
        best_j = -1
        best_c = large_cost
        for j in range(M):
            if j in used_lm:
                continue
            if not valid_mask[i, j]:
                continue
            c = cost[i, j]
            if c < best_c:
                best_c = c
                best_j = j
        if best_j >= 0 and best_c < 0.5 * large_cost:
            used_lm.add(best_j)
            pairs.append((int(i), int(best_j), float(best_c)))

    return pairs
