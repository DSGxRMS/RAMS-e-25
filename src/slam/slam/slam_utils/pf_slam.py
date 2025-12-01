# slam_utils/pf_slam.py
#
# Minimal 2D particle-filter SLAM core for cone maps.

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import numpy as np

from .association import (
    build_mahalanobis_cost_matrix,
    hungarian_assign,
    chi2_quantile_2d,
)


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class Landmark:
    mean: np.ndarray          # (2,) in SLAM/map frame
    cov: np.ndarray           # (2,2)
    cls: int                  # colour / class_id
    hits: int = 1             # how many updates


@dataclass
class Particle:
    x: float
    y: float
    yaw: float
    weight: float
    landmarks: List[Landmark] = field(default_factory=list)


class ParticleFilterSLAM:
    """
    Very simple PF-SLAM engine (SE(2) + per-particle 2D landmarks).

    Interface you care about from the visualiser:
      - init_pose(x0,y0,yaw0)
      - predict(delta=(dx,dy,dyaw))
      - update(meas_body)
      - get_best_particle()
    """

    def __init__(
        self,
        num_particles: int = 80,
        process_std_xy: float = 0.03,
        process_std_yaw: float = 0.01,
        meas_sigma_xy: float = 0.20,
        birth_sigma_xy: float = 0.40,
        gate_prob: float = 0.997,          # ~3-sigma in 2D
        resample_neff_ratio: float = 0.5,
    ):
        self.N = int(num_particles)
        self.proc_xy = float(process_std_xy)
        self.proc_yaw = float(process_std_yaw)
        self.meas_sigma = float(meas_sigma_xy)
        self.birth_sigma = float(birth_sigma_xy)
        self.gate_chi2 = chi2_quantile_2d(gate_prob)
        self.resample_neff_ratio = float(resample_neff_ratio)

        self.particles: List[Particle] = []
        self.initialised = False

    # ------------- Helpers -------------

    def _ensure_init(self, x0: float, y0: float, yaw0: float):
        if self.initialised:
            return
        self.particles = [
            Particle(
                x=x0,
                y=y0,
                yaw=yaw0,
                weight=1.0 / self.N,
                landmarks=[],
            )
            for _ in range(self.N)
        ]
        self.initialised = True

    # ------------- API -------------

    def init_pose(self, x0: float, y0: float, yaw0: float):
        """
        Hard reset to a given pose, clears landmarks.
        """
        self.initialised = False
        self._ensure_init(x0, y0, yaw0)

    def predict(self, delta: Tuple[float, float, float]):
        """
        Apply odom increment (dx,dy,dyaw) in world frame + process noise.
        """
        if not self.initialised:
            return

        dx, dy, dyaw = delta

        if self.proc_xy > 0.0:
            noise_xy = np.random.normal(0.0, self.proc_xy, size=(self.N, 2))
        else:
            noise_xy = np.zeros((self.N, 2))

        if self.proc_yaw > 0.0:
            noise_y = np.random.normal(0.0, self.proc_yaw, size=(self.N,))
        else:
            noise_y = np.zeros((self.N,))

        for i, p in enumerate(self.particles):
            p.x += dx + noise_xy[i, 0]
            p.y += dy + noise_xy[i, 1]
            p.yaw = wrap(p.yaw + dyaw + noise_y[i])

    def update(self, meas_body: List[Tuple[float, float, int]]):
        """
        Perform measurement update with cones in BODY frame:

          meas_body: list of (bx, by, class_id)

        For each particle:
          - convert to world -> z_w
          - per-class Mahalanobis gating + Hungarian association
          - EKF landmark update
          - births for unmatched
          - weight update from sum of m^2
        """
        if not self.initialised:
            return
        if not meas_body:
            return

        # pre-group measurements by class
        meas_by_cls: Dict[int, List[Tuple[float, float]]] = {}
        for bx, by, cls_id in meas_body:
            cls = int(cls_id)
            meas_by_cls.setdefault(cls, []).append((float(bx), float(by)))

        # log-weight update for stability
        log_w = np.log(np.array([max(p.weight, 1e-12) for p in self.particles], float))

        for idx, p in enumerate(self.particles):
            # car pose
            xw, yw, yaw = p.x, p.y, p.yaw
            c = math.cos(yaw)
            s = math.sin(yaw)
            R = np.array([[c, -s], [s, c]], float)

            m2_sum = 0.0

            # Landmarks per class
            lms_by_cls: Dict[int, List[int]] = {}
            for j, lm in enumerate(p.landmarks):
                lms_by_cls.setdefault(lm.cls, []).append(j)

            # ------ per class association ------
            for cls, obs_b in meas_by_cls.items():
                # world coords of measurements under this particle
                if not obs_b:
                    continue
                obs_w = []
                for bx, by in obs_b:
                    pb = np.array([bx, by], float)
                    pw = np.array([xw, yw], float) + R @ pb
                    obs_w.append(pw)
                obs_w = np.asarray(obs_w, float)  # (N_c,2)

                lm_indices = lms_by_cls.get(cls, [])
                if not lm_indices:
                    # no landmarks of this class yet -> births later
                    for z in obs_w:
                        self._birth_landmark(p, z, cls)
                    continue

                lm_means = np.stack([p.landmarks[j].mean for j in lm_indices], axis=0)
                lm_covs = np.stack([p.landmarks[j].cov for j in lm_indices], axis=0)

                cost, valid = build_mahalanobis_cost_matrix(
                    lm_means,
                    lm_covs,
                    obs_w,
                    meas_sigma_xy=self.meas_sigma,
                    gate_chi2=self.gate_chi2,
                )

                pairs = hungarian_assign(cost, valid)

                # mark all measurements as unmatched initially
                matched_meas = set()
                matched_lms = set()

                # EKF updates
                R_meas = np.diag([self.meas_sigma**2, self.meas_sigma**2])

                for (mi, li_local, m2) in pairs:
                    matched_meas.add(mi)
                    matched_lms.add(li_local)
                    lm_idx = lm_indices[li_local]
                    lm = p.landmarks[lm_idx]

                    mu = lm.mean
                    P = lm.cov
                    z = obs_w[mi]

                    S = P + R_meas
                    try:
                        Sinv = np.linalg.inv(S)
                    except np.linalg.LinAlgError:
                        Sinv = np.linalg.pinv(S)
                    K = P @ Sinv
                    innov = z - mu

                    lm.mean = mu + K @ innov
                    lm.cov = (np.eye(2) - K) @ P
                    lm.hits += 1

                    m2_sum += m2

                # births for measurements with no assignment
                for mi, z in enumerate(obs_w):
                    if mi not in matched_meas:
                        self._birth_landmark(p, z, cls)

            # update log weight from total m^2 (likelihood ∝ exp(-0.5 * m2_sum))
            log_w[idx] += -0.5 * m2_sum

        # normalise weights
        max_log = float(np.max(log_w))
        w = np.exp(log_w - max_log)
        sw = float(np.sum(w))
        if sw <= 0.0 or not math.isfinite(sw):
            # reset to uniform
            for p in self.particles:
                p.weight = 1.0 / self.N
        else:
            w /= sw
            for i, p in enumerate(self.particles):
                p.weight = float(w[i])

        # resample if degeneracy
        neff = 1.0 / float(np.sum(w * w))
        if neff < self.resample_neff_ratio * self.N:
            self._systematic_resample()

    def _birth_landmark(self, p: Particle, z_w: np.ndarray, cls: int):
        P0 = np.diag([self.birth_sigma**2, self.birth_sigma**2])
        lm = Landmark(mean=z_w.copy(), cov=P0, cls=int(cls), hits=1)
        p.landmarks.append(lm)

    def _systematic_resample(self):
        N = self.N
        weights = np.array([p.weight for p in self.particles], float)
        if np.any(weights < 0.0):
            weights = np.maximum(weights, 0.0)
        sw = float(np.sum(weights))
        if sw <= 0.0 or not math.isfinite(sw):
            weights[:] = 1.0 / N
        else:
            weights /= sw

        positions = (np.arange(N) + np.random.uniform()) / N
        cumulative_sum = np.cumsum(weights)
        indexes = np.zeros(N, dtype=int)

        i = 0
        j = 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1

        new_particles: List[Particle] = []
        for idx in indexes:
            src = self.particles[idx]
            # deep copy of landmarks
            new_lms = [
                Landmark(
                    mean=lm.mean.copy(),
                    cov=lm.cov.copy(),
                    cls=lm.cls,
                    hits=lm.hits,
                )
                for lm in src.landmarks
            ]
            new_particles.append(
                Particle(
                    x=src.x,
                    y=src.y,
                    yaw=src.yaw,
                    weight=1.0 / N,
                    landmarks=new_lms,
                )
            )
        self.particles = new_particles

    # ------------- Query -------------

    def get_best_particle(self) -> Optional[Particle]:
        if not self.initialised or not self.particles:
            return None
        best_idx = int(np.argmax([p.weight for p in self.particles]))
        return self.particles[best_idx]
