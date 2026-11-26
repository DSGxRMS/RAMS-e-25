"""Ground-removal backends shared by the LiDAR perception node."""

from enum import Enum

import numpy as np


class GroundRemovalMethod(str, Enum):
    """Supported ground-removal implementations."""

    PATCHWORK = "patchwork"
    RANSAC = "ransac"

    @classmethod
    def parse(cls, value):
        """Parse a ROS string parameter and provide a useful error."""
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError as error:
            choices = ", ".join(method.value for method in cls)
            raise ValueError(
                "ground_removal_method must be one of: {}".format(choices)
            ) from error


def default_cluster_eps(method):
    """Return the clustering radius validated for a ground-removal method."""
    method = GroundRemovalMethod.parse(method)
    if method is GroundRemovalMethod.PATCHWORK:
        return 0.50
    return 1.00


def vectorized_ransac(points, iterations=50, threshold=0.05,
                      minimum_inliers=30):
    """Remove the best horizontal plane using batched RANSAC hypotheses.

    Sampling intentionally uses NumPy's global RNG in the same order as the
    previous loop implementation. Given the same seed, this produces the same
    retained points while evaluating plane distances in one matrix operation.
    """
    count = points.shape[0]
    if count < 3:
        return None

    iterations = max(1, int(iterations))
    threshold = float(threshold)
    minimum_inliers = max(3, int(minimum_inliers))
    cloud = points.astype(np.float32)

    samples = np.vstack([
        np.random.choice(count, 3, replace=False)
        for _ in range(iterations)
    ])
    first = cloud[samples[:, 0]]
    normals = np.cross(
        cloud[samples[:, 1]] - first,
        cloud[samples[:, 2]] - first,
    )
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths >= 1e-3
    normals[valid] /= lengths[valid, None]
    valid &= np.abs(normals[:, 2]) >= 0.7

    offsets = -np.einsum("ij,ij->i", normals, first)
    distances = np.abs(cloud @ normals.T + offsets)
    inliers = distances < threshold
    counts = inliers.sum(axis=0)
    counts[~valid] = -1
    best = int(np.argmax(counts))

    if counts[best] < minimum_inliers:
        return None
    return cloud[~inliers[:, best]]


class PatchworkGroundRemover:
    """Small adapter around the Patchwork++ Python binding."""

    def __init__(self, sensor_height=1.08, minimum_range=0.2,
                 maximum_range=80.0, distance_threshold=0.04,
                 enable_rnr=False):
        try:
            import pypatchworkpp as patchworkpp
        except ImportError as error:
            raise RuntimeError(
                "Patchwork++ is selected but pypatchworkpp is unavailable. "
                "Install dependencies from src/perception/requirements.txt."
            ) from error

        parameters = patchworkpp.Parameters()
        parameters.verbose = False
        parameters.sensor_height = float(sensor_height)
        parameters.min_range = float(minimum_range)
        parameters.max_range = float(maximum_range)
        parameters.th_dist = float(distance_threshold)
        parameters.enable_RNR = bool(enable_rnr)
        self._engine = patchworkpp.patchworkpp(parameters)

    def remove(self, points):
        """Return Patchwork++ non-ground points as a contiguous Nx3 array."""
        cloud = np.ascontiguousarray(points, dtype=np.float32)
        self._engine.estimateGround(cloud)
        retained = np.asarray(
            self._engine.getNonground(), dtype=np.float32
        )
        return retained.reshape(-1, 3)
