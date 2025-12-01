# slam_utils/association.py
#
# Utilities for association between:
#   - landmarks: existing map points (shape: [L, 2])
#   - measurements: new observations (shape: [M, 2])
#
# We use:
#   - Mahalanobis distance with isotropic covariance (sigma_xy)
#   - Chi-square gating in 2D (chi2_gate)
#   - Hungarian assignment (via SciPy) if available; otherwise
#     greedy 1-to-1 nearest neighbour inside the gate.

from typing import List, Tuple
import numpy as np

LARGE_COST_DEFAULT = 1e6


def build_mahalanobis_cost_matrix(
    landmarks_xy: np.ndarray,
    meas_xy: np.ndarray,
    sigma_xy: float,
    chi2_gate: float,
    large_cost: float = LARGE_COST_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    landmarks_xy: (L, 2) array of existing landmark positions in world.
    meas_xy:      (M, 2) array of new measurements in world.
    sigma_xy:     isotropic std dev [m] for x & y.
    chi2_gate:    chi-square threshold in 2D (e.g. ~9.21 for 99%).
    large_cost:   sentinel cost for "invalid/outside gate".

    Returns:
      cost:  (M, L) matrix of Mahalanobis distance^2 (or large_cost).
      valid: (M, L) boolean mask, True where inside gate.
    """
    if meas_xy.size == 0 or landmarks_xy.size == 0:
        # shape must still be (M, L)
        M = meas_xy.shape[0]
        L = landmarks_xy.shape[0]
        return np.full((M, L), large_cost, dtype=float), np.zeros((M, L), dtype=bool)

    # Ensure correct shapes
    meas_xy = np.asarray(meas_xy, dtype=float).reshape(-1, 2)
    landmarks_xy = np.asarray(landmarks_xy, dtype=float).reshape(-1, 2)

    M = meas_xy.shape[0]
    L = landmarks_xy.shape[0]
    cost = np.full((M, L), large_cost, dtype=float)
    valid = np.zeros((M, L), dtype=bool)

    if sigma_xy <= 0.0:
        sigma_xy = 1e-3
    inv_sigma2 = 1.0 / (sigma_xy * sigma_xy)

    # For each measurement row, compute all landmark distances in vectorised form
    for i in range(M):
        dx = landmarks_xy[:, 0] - meas_xy[i, 0]
        dy = landmarks_xy[:, 1] - meas_xy[i, 1]
        d2 = (dx * dx + dy * dy) * inv_sigma2  # Mahalanobis^2 with isotropic S

        inside = d2 <= chi2_gate
        cost[i, inside] = d2[inside]
        valid[i, inside] = True

    return cost, valid


def hungarian_assign(
    cost: np.ndarray,
    valid_mask: np.ndarray,
    large_cost: float = LARGE_COST_DEFAULT,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """
    Apply Hungarian assignment (if SciPy available) on the cost matrix,
    then enforce gating via valid_mask & large_cost.

    cost:       (M, L) cost matrix (distance^2 or large_cost).
    valid_mask: (M, L) boolean; True if this (meas, landmark) was inside the gate.

    Returns:
      matches:           list[(m_idx, l_idx, cost_val)]
      unmatched_meas:    list of measurement indices (0..M-1) with no match
      unmatched_landmarks: list of landmark indices (0..L-1) with no match
    """
    cost = np.asarray(cost, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    M, L = cost.shape
    if M == 0 or L == 0:
        return [], list(range(M)), list(range(L))

    # Try SciPy Hungarian; fall back to greedy if not present.
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore
        row_ind, col_ind = linear_sum_assignment(cost)
        use_hungarian = True
    except Exception:
        use_hungarian = False

    matches: List[Tuple[int, int, float]] = []

    if use_hungarian:
        # Hungarian returns a 1-to-1 assignment but may include
        # (meas,landmark) pairs we marked as invalid (large_cost).
        # We filter them out using valid_mask & large_cost.
        threshold = 0.5 * large_cost
        for r, c in zip(row_ind, col_ind):
            if valid_mask[r, c] and cost[r, c] < threshold:
                matches.append((int(r), int(c), float(cost[r, c])))
    else:
        # Greedy 1-to-1 nearest neighbour on valid entries
        idxs = np.argwhere(valid_mask)
        if idxs.size > 0:
            pairs = [(float(cost[i, j]), int(i), int(j)) for (i, j) in idxs]
            pairs.sort(key=lambda t: t[0])  # small cost first

            used_m = set()
            used_l = set()
            for d, i, j in pairs:
                if i in used_m or j in used_l:
                    continue
                if d >= large_cost:
                    continue
                matches.append((i, j, d))
                used_m.add(i)
                used_l.add(j)

    matched_m = {m for (m, _, _) in matches}
    matched_l = {l for (_, l, _) in matches}

    unmatched_meas = [i for i in range(M) if i not in matched_m]
    unmatched_landmarks = [j for j in range(L) if j not in matched_l]

    return matches, unmatched_meas, unmatched_landmarks
