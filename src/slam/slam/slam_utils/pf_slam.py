# slam_utils/pf_slam.py

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set

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
    hits: int = 1             # how many successful updates (matched)
    misses: int = 0           # how many consecutive frames not matched


@dataclass
class CandidateLandmark:
    """
    Proto-landmark that must survive multiple observations before being
    promoted to a real Landmark. Helps avoid spawning ghosts from one-off
    spurious measurements.
    """
    mean: np.ndarray          # (2,) in SLAM/map frame
    cov: np.ndarray           # (2,2)
    cls: int
    hits: int = 1             # how many times this candidate was re-observed
    last_seen_frame: int = 0  # frame_counter of last observation


@dataclass
class Particle:
    x: float
    y: float
    yaw: float
    weight: float
    landmarks: List[Landmark] = field(default_factory=list)
    candidates: List[CandidateLandmark] = field(default_factory=list)


class ParticleFilterSLAM:
    """
    Very simple PF-SLAM engine (SE(2) + per-particle 2D landmarks).

    Interface:
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
        # New robustness / persistence tuning:
        birth_hits: int = 5,               # candidate must be seen this many times before promotion
        candidate_assoc_dist: float = 0.7, # [m] radius to associate unmatched meas to candidate
        candidate_max_age: int = 15,       # [frames] drop candidates not seen this long
        merge_dist: float = 0.7,           # [m] merge landmarks closer than this (same class)
        min_hits_keep: int = 3,            # once a landmark has this many hits, it is considered "mature"
        max_misses_drop: int = 5,          # immature landmarks dropped after this many misses
    ):
        self.N = int(num_particles)
        self.proc_xy = float(process_std_xy)
        self.proc_yaw = float(process_std_yaw)
        self.meas_sigma = float(meas_sigma_xy)
        self.birth_sigma = float(birth_sigma_xy)
        self.gate_chi2 = chi2_quantile_2d(gate_prob)
        self.resample_neff_ratio = float(resample_neff_ratio)

        # Persistence / robustness parameters
        self.birth_hits = int(birth_hits)
        self.candidate_assoc_dist = float(candidate_assoc_dist)
        self.candidate_max_age = int(candidate_max_age)
        self.merge_dist = float(merge_dist)
        self.min_hits_keep = int(min_hits_keep)
        self.max_misses_drop = int(max_misses_drop)

        # Corridor NN birth-limiting (heuristic)
        self.corridor_length = 10.0  # [m] ahead of car
        self.corridor_width = 4.0    # [m] total width (±2 m lateral)
        self.nn_max_dist = 1.0       # [m] max distance for NN reuse in corridor

        self.particles: List[Particle] = []
        self.initialised = False

        # Frame counter for candidate ageing
        self.frame_counter: int = 0

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
                candidates=[],
            )
            for _ in range(self.N)
        ]
        self.initialised = True

    def _in_corridor(self, car_x: float, car_y: float, car_yaw: float,
                     px: float, py: float) -> bool:
        """
        Check if world point (px,py) is in a rectangular corridor
        [0, corridor_length] x [-corridor_width/2, +corridor_width/2]
        in the car frame.
        """
        dx = px - car_x
        dy = py - car_y

        c = math.cos(-car_yaw)
        s = math.sin(-car_yaw)
        x_local = c * dx - s * dy
        y_local = s * dx + c * dy

        if x_local < 0.0 or x_local > self.corridor_length:
            return False
        if abs(y_local) > 0.5 * self.corridor_width:
            return False
        return True

    # ------------- API -------------

    def init_pose(self, x0: float, y0: float, yaw0: float):
        """
        Hard reset to a given pose, clears landmarks and candidates.
        """
        self.initialised = False
        self._ensure_init(x0, y0, yaw0)
        self.frame_counter = 0

    def predict(self, delta: Tuple[float, float, float]):
        """
        Apply odom increment (dx,dy,dyaw) in the CAR/BODY frame + process noise.
        Converts (dx,dy) into world frame using each particle yaw.
        """
        if not self.initialised:
            return

        dx_b, dy_b, dyaw = delta

        if self.proc_xy > 0.0:
            noise_xy = np.random.normal(0.0, self.proc_xy, size=(self.N, 2))
        else:
            noise_xy = np.zeros((self.N, 2))

        if self.proc_yaw > 0.0:
            noise_y = np.random.normal(0.0, self.proc_yaw, size=(self.N,))
        else:
            noise_y = np.zeros((self.N,))

        for i, p in enumerate(self.particles):
            # body -> world using particle yaw
            c = math.cos(p.yaw)
            s = math.sin(p.yaw)
            dx_w = c * dx_b - s * dy_b
            dy_w = s * dx_b + c * dy_b

            p.x += dx_w + noise_xy[i, 0]
            p.y += dy_w + noise_xy[i, 1]
            p.yaw = wrap(p.yaw + dyaw + noise_y[i])



    def update(self, meas_body: List[Tuple[float, float, int]]):
        """
        Perform measurement update with cones in BODY frame:

          meas_body: list of (bx, by, class_id)

        For each particle:
          - convert to world -> z_w
          - per-class Mahalanobis gating + Hungarian association
          - EKF landmark update
          - corridor-based NN reuse + capped births
          - births via candidate mechanism outside corridor
          - landmark pruning / merging
          - weight update from Gaussian log-likelihood
        """
        if not self.initialised:
            return
        if not meas_body:
            return

        self.frame_counter += 1

        # pre-group measurements by class
        meas_by_cls: Dict[int, List[Tuple[float, float]]] = {}
        for bx, by, cls_id in meas_body:
            cls = int(cls_id)
            meas_by_cls.setdefault(cls, []).append((float(bx), float(by)))

        # log-weight update for stability
        log_w = np.log(np.array([max(p.weight, 1e-12) for p in self.particles], float))

        for idx, p in enumerate(self.particles):
            # car pose
            car_x, car_y, car_yaw = p.x, p.y, p.yaw
            c = math.cos(car_yaw)
            s = math.sin(car_yaw)
            R = np.array([[c, -s], [s, c]], float)

            # Sum of Gaussian terms (m^2 + log|S|) for all associated measurements
            ll_sum = 0.0

            # Landmarks per class
            lms_by_cls: Dict[int, List[int]] = {}
            for j, lm in enumerate(p.landmarks):
                lms_by_cls.setdefault(lm.cls, []).append(j)

            # Global set of landmarks matched this frame (indices in p.landmarks)
            matched_lm_global: Set[int] = set()

            # ------ per class association + corridor NN reuse ------
            for cls, obs_b in meas_by_cls.items():
                if not obs_b:
                    continue

                # world coords of measurements under this particle
                obs_w = []
                for bx, by in obs_b:
                    pb = np.array([bx, by], float)
                    pw = np.array([car_x, car_y], float) + R @ pb
                    obs_w.append(pw)
                obs_w = np.asarray(obs_w, float)  # (N_c,2)
                Nc = obs_w.shape[0]

                lm_indices = lms_by_cls.get(cls, [])

                # measurement noise
                R_meas = np.diag([self.meas_sigma**2, self.meas_sigma**2])

                # no landmarks of this class yet -> normal births (no corridor magic)
                if not lm_indices:
                    for z in obs_w:
                        self._handle_unmatched_measurement(p, z, cls)
                    continue

                # normal Mahalanobis-gated + Hungarian association
                cost, valid = build_mahalanobis_cost_matrix(
                    np.stack([p.landmarks[j].mean for j in lm_indices], axis=0),
                    np.stack([p.landmarks[j].cov for j in lm_indices], axis=0),
                    obs_w,
                    meas_sigma_xy=self.meas_sigma,
                    gate_chi2=self.gate_chi2,
                )

                pairs = hungarian_assign(cost, valid)

                matched_meas: Set[int] = set()

                # primary EKF updates for gated pairs
                for (mi, li_local, m2) in pairs:
                    matched_meas.add(mi)
                    lm_idx = lm_indices[li_local]
                    matched_lm_global.add(lm_idx)

                    lm = p.landmarks[lm_idx]
                    mu = lm.mean
                    P = lm.cov
                    z = obs_w[mi]

                    S = P + R_meas
                    try:
                        Sinv = np.linalg.inv(S)
                    except np.linalg.LinAlgError:
                        Sinv = np.linalg.pinv(S)

                    sign, logdet = np.linalg.slogdet(S)
                    if sign <= 0.0 or not math.isfinite(logdet):
                        logdet = 0.0

                    K = P @ Sinv
                    innov = z - mu

                    lm.mean = mu + K @ innov
                    lm.cov = (np.eye(2) - K) @ P
                    lm.hits += 1
                    lm.misses = 0

                    ll_sum += m2 + logdet

                # ---- Corridor-based NN reuse + capped births ----

                # 1) Unmatched measurements (global indices) and landmarks of this class
                unmatched_meas_all: List[Tuple[int, np.ndarray]] = [
                    (mi, obs_w[mi]) for mi in range(Nc) if mi not in matched_meas
                ]
                unmatched_lm_all: List[int] = [
                    lm_idx for lm_idx in lm_indices if lm_idx not in matched_lm_global
                ]

                # Split into corridor vs outside (for both meas and landmarks)
                unmatched_meas_corr: List[Tuple[int, np.ndarray]] = []
                unmatched_meas_outside: List[Tuple[int, np.ndarray]] = []

                for mi, z in unmatched_meas_all:
                    if self._in_corridor(car_x, car_y, car_yaw, z[0], z[1]):
                        unmatched_meas_corr.append((mi, z))
                    else:
                        unmatched_meas_outside.append((mi, z))

                unmatched_lm_corr: List[int] = []
                unmatched_lm_outside: List[int] = []
                for lm_idx in unmatched_lm_all:
                    lm = p.landmarks[lm_idx]
                    if self._in_corridor(car_x, car_y, car_yaw,
                                         float(lm.mean[0]), float(lm.mean[1])):
                        unmatched_lm_corr.append(lm_idx)
                    else:
                        unmatched_lm_outside.append(lm_idx)

                # 2) NN within corridor (ignoring gates), updating landmarks instead of birthing
                unmatched_meas_corr_left: List[Tuple[int, np.ndarray]] = unmatched_meas_corr
                unmatched_lm_corr_left: List[int] = unmatched_lm_corr

                if unmatched_meas_corr and unmatched_lm_corr:
                    used_lm: Set[int] = set()
                    max_d2 = self.nn_max_dist ** 2

                    new_unmatched_meas_corr: List[Tuple[int, np.ndarray]] = []

                    for mi, z in unmatched_meas_corr:
                        best_lm = -1
                        best_d2 = max_d2
                        for lm_idx in unmatched_lm_corr:
                            if lm_idx in used_lm:
                                continue
                            lm = p.landmarks[lm_idx]
                            d2 = float(np.sum((z - lm.mean) ** 2))
                            if d2 < best_d2:
                                best_d2 = d2
                                best_lm = lm_idx

                        if best_lm >= 0:
                            # Treat as an extra match: EKF update without gating
                            lm = p.landmarks[best_lm]
                            mu = lm.mean
                            P = lm.cov

                            S = P + R_meas
                            try:
                                Sinv = np.linalg.inv(S)
                            except np.linalg.LinAlgError:
                                Sinv = np.linalg.pinv(S)

                            innov = z - mu
                            m2_nn = float(innov.T @ Sinv @ innov)
                            sign, logdet = np.linalg.slogdet(S)
                            if sign <= 0.0 or not math.isfinite(logdet):
                                logdet = 0.0

                            K = P @ Sinv
                            lm.mean = mu + K @ innov
                            lm.cov = (np.eye(2) - K) @ P
                            lm.hits += 1
                            lm.misses = 0

                            matched_meas.add(mi)
                            matched_lm_global.add(best_lm)
                            used_lm.add(best_lm)

                            ll_sum += m2_nn + logdet
                        else:
                            # still unmatched in corridor after NN
                            new_unmatched_meas_corr.append((mi, z))

                    unmatched_meas_corr_left = new_unmatched_meas_corr
                    unmatched_lm_corr_left = [
                        lm_idx for lm_idx in unmatched_lm_corr
                        if lm_idx not in used_lm
                    ]

                # 3) Births:
                #    - In corridor: cap births at max(0, m_corr - n_corr)
                #    - Outside: behave as before

                n_corr = len(unmatched_lm_corr_left)
                m_corr = len(unmatched_meas_corr_left)
                births_allowed_corr = max(0, m_corr - n_corr)

                # corridor measurements to birth (at most births_allowed_corr)
                meas_for_birth_corr = [
                    z for (_mi, z) in unmatched_meas_corr_left[:births_allowed_corr]
                ]

                # outside corridor: all unmatched measurements are eligible for birth
                meas_for_birth_outside = [z for (_mi, z) in unmatched_meas_outside]

                for z in meas_for_birth_corr:
                    self._handle_unmatched_measurement(p, z, cls)
                for z in meas_for_birth_outside:
                    self._handle_unmatched_measurement(p, z, cls)

            # landmark miss counters + pruning + merging
            self._update_landmark_miss_counters(p, matched_lm_global)
            self._merge_close_landmarks(p)
            self._prune_candidates(p)

            # update log weight from total Gaussian term
            # log p ∝ -0.5 * Σ (m^2 + log|S|)
            log_w[idx] += -0.5 * ll_sum

        # normalise weights (fixing stale w issue)
        max_log = float(np.max(log_w))
        w = np.exp(log_w - max_log)
        sw = float(np.sum(w))
        if sw <= 0.0 or not math.isfinite(sw):
            # reset to uniform
            w[:] = 1.0 / self.N
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

    # --------- Landmark / candidate helpers ---------

    def _birth_landmark(self, p: Particle, z_w: np.ndarray, cls: int):
        P0 = np.diag([self.birth_sigma**2, self.birth_sigma**2])
        lm = Landmark(
            mean=z_w.copy(),
            cov=P0,
            cls=int(cls),
            hits=1,
            misses=0,
        )
        p.landmarks.append(lm)

    def _handle_unmatched_measurement(self, p: Particle, z_w: np.ndarray, cls: int):
        """
        Handle an unmatched measurement via candidate proto-landmarks.

        - If a candidate of same class exists within candidate_assoc_dist -> update it.
        - If candidate hits >= birth_hits -> promote to real Landmark.
        - Else create a new candidate.
        """
        assoc_dist2 = self.candidate_assoc_dist ** 2

        best_idx = -1
        best_d2 = assoc_dist2

        for i, c in enumerate(p.candidates):
            if c.cls != cls:
                continue
            d2 = float(np.sum((c.mean - z_w) ** 2))
            if d2 < best_d2:
                best_d2 = d2
                best_idx = i

        R_meas = np.diag([self.meas_sigma**2, self.meas_sigma**2])

        if best_idx >= 0:
            # Update existing candidate
            c = p.candidates[best_idx]
            P = c.cov
            S = P + R_meas
            try:
                Sinv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                Sinv = np.linalg.pinv(S)
            K = P @ Sinv
            innov = z_w - c.mean
            c.mean = c.mean + K @ innov
            c.cov = (np.eye(2) - K) @ P
            c.hits += 1
            c.last_seen_frame = self.frame_counter

            # Promote if persistent enough
            if c.hits >= self.birth_hits:
                self._birth_landmark(p, c.mean, cls)
                # Remove candidate
                del p.candidates[best_idx]
        else:
            # New candidate
            P0 = np.diag([self.birth_sigma**2, self.birth_sigma**2])
            p.candidates.append(
                CandidateLandmark(
                    mean=z_w.copy(),
                    cov=P0,
                    cls=int(cls),
                    hits=1,
                    last_seen_frame=self.frame_counter,
                )
            )

    def _prune_candidates(self, p: Particle):
        """
        Drop candidates that have not been seen recently.
        """
        cutoff = self.frame_counter - self.candidate_max_age
        p.candidates = [
            c for c in p.candidates if c.last_seen_frame >= cutoff
        ]

    def _update_landmark_miss_counters(self, p: Particle, matched_lm_global: Set[int]):
        for idx, lm in enumerate(p.landmarks):
            if idx in matched_lm_global:
                # matched this frame -> reset miss count
                lm.misses = 0
            else:
                # only penalise immature landmarks
                if lm.hits < self.min_hits_keep:
                    lm.misses += 1

        # prune only immature landmarks that haven't been seen for too long
        p.landmarks = [
            lm
            for lm in p.landmarks
            if not (lm.hits < self.min_hits_keep and lm.misses > self.max_misses_drop)
        ]

    def _merge_close_landmarks(self, p: Particle):
        """
        Merge landmarks of the same class that are within merge_dist of each other.
        Helps collapse duplicate cones.
        """
        if len(p.landmarks) < 2:
            return

        merge_d2 = self.merge_dist ** 2
        used = [False] * len(p.landmarks)
        merged: List[Landmark] = []

        for i, lm_i in enumerate(p.landmarks):
            if used[i]:
                continue

            cluster_indices = [i]
            used[i] = True

            for j in range(i + 1, len(p.landmarks)):
                if used[j]:
                    continue
                lm_j = p.landmarks[j]
                if lm_j.cls != lm_i.cls:
                    continue
                d2 = float(np.sum((lm_j.mean - lm_i.mean) ** 2))
                if d2 <= merge_d2:
                    cluster_indices.append(j)
                    used[j] = True

            if len(cluster_indices) == 1:
                merged.append(p.landmarks[cluster_indices[0]])
                continue

            # Merge cluster
            cls = p.landmarks[cluster_indices[0]].cls
            means = []
            covs = []
            hits = []
            misses = []

            for k in cluster_indices:
                lm_k = p.landmarks[k]
                means.append(lm_k.mean)
                covs.append(lm_k.cov)
                hits.append(lm_k.hits)
                misses.append(lm_k.misses)

            means = np.stack(means, axis=0)
            covs = np.stack(covs, axis=0)
            hits_arr = np.asarray(hits, float)
            w = hits_arr / np.sum(hits_arr)

            mean_merged = np.sum(w[:, None] * means, axis=0)
            cov_merged = np.sum(w[:, None, None] * covs, axis=0)
            hits_merged = int(np.sum(hits_arr))
            misses_merged = int(min(misses))

            merged.append(
                Landmark(
                    mean=mean_merged,
                    cov=cov_merged,
                    cls=cls,
                    hits=hits_merged,
                    misses=misses_merged,
                )
            )

        p.landmarks = merged

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
            # deep copy of landmarks and candidates
            new_lms = [
                Landmark(
                    mean=lm.mean.copy(),
                    cov=lm.cov.copy(),
                    cls=lm.cls,
                    hits=lm.hits,
                    misses=lm.misses,
                )
                for lm in src.landmarks
            ]
            new_candidates = [
                CandidateLandmark(
                    mean=c.mean.copy(),
                    cov=c.cov.copy(),
                    cls=c.cls,
                    hits=c.hits,
                    last_seen_frame=c.last_seen_frame,
                )
                for c in src.candidates
            ]
            new_particles.append(
                Particle(
                    x=src.x,
                    y=src.y,
                    yaw=src.yaw,
                    weight=1.0 / N,
                    landmarks=new_lms,
                    candidates=new_candidates,
                )
            )
        self.particles = new_particles

    # ------------- Query -------------

    def get_best_particle(self) -> Optional[Particle]:
        if not self.initialised or not self.particles:
            return None
        best_idx = int(np.argmax([p.weight for p in self.particles]))
        return self.particles[best_idx]
