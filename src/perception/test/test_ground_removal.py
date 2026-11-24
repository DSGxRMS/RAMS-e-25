"""Tests for selectable LiDAR ground removal."""

import numpy as np
import pytest

from perception.ground_removal import (
    GroundRemovalMethod,
    default_cluster_eps,
    vectorized_ransac,
)


def _loop_ransac(points, iterations, threshold, minimum_inliers):
    """Previous implementation retained here as an equivalence oracle."""
    count = points.shape[0]
    cloud = points.astype(np.float32)
    best_inliers = None
    best_count = 0
    for _ in range(iterations):
        indices = np.random.choice(count, 3, replace=False)
        first, second, third = cloud[indices]
        normal = np.cross(second - first, third - first)
        length = np.linalg.norm(normal)
        if length < 1e-3:
            continue
        normal /= length
        if abs(normal[2]) < 0.7:
            continue
        offset = -np.dot(normal, first)
        inliers = np.where(
            np.abs(cloud @ normal + offset) < threshold
        )[0]
        if inliers.size > best_count:
            best_count = inliers.size
            best_inliers = inliers
    if best_inliers is None or best_count < minimum_inliers:
        return None
    mask = np.ones(count, dtype=bool)
    mask[best_inliers] = False
    return cloud[mask]


def test_method_parser_and_defaults():
    assert (
        GroundRemovalMethod.parse("PATCHWORK")
        is GroundRemovalMethod.PATCHWORK
    )
    assert GroundRemovalMethod.parse("ransac") is GroundRemovalMethod.RANSAC
    assert default_cluster_eps("patchwork") == 0.50
    assert default_cluster_eps("ransac") == 1.00
    assert default_cluster_eps(GroundRemovalMethod.RANSAC) == 1.00
    with pytest.raises(ValueError, match="ground_removal_method"):
        GroundRemovalMethod.parse("unknown")


def test_vectorized_ransac_matches_previous_implementation():
    generator = np.random.default_rng(7)
    ground_xy = generator.uniform([-8.0, -5.0], [15.0, 5.0], (500, 2))
    ground_z = generator.normal(-1.0, 0.004, (500, 1))
    objects = generator.uniform([2.0, -4.0, -0.8], [14.0, 4.0, 0.5], (80, 3))
    cloud = np.vstack(
        (np.hstack((ground_xy, ground_z)), objects)
    ).astype(np.float32)

    np.random.seed(42)
    expected = _loop_ransac(cloud, 50, 0.05, 30)
    expected_state = np.random.get_state()
    np.random.seed(42)
    actual = vectorized_ransac(cloud, 50, 0.05, 30)
    actual_state = np.random.get_state()

    assert np.array_equal(actual, expected)
    assert expected_state[0] == actual_state[0]
    assert np.array_equal(expected_state[1], actual_state[1])
    assert expected_state[2:] == actual_state[2:]


def test_vectorized_ransac_handles_insufficient_cloud():
    cloud = np.zeros((2, 3), dtype=np.float32)
    assert vectorized_ransac(cloud) is None
