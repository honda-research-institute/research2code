"""Seeded synthetic fixtures for behavioral probes (recentering slice 1.2).

Design heritage: v3's orphaned harness layer (the v3 harness salvage note (internal, not shipped))
— tiny prototype-separable datasets with guaranteed signal so above-chance /
loss-decreases probes can't false-positive, hand-authored planning scenarios,
everything seeded and CPU-cheap (whole-fixture construction in milliseconds).

numpy-only by design: probes that need torch convert at their own boundary and
go `unprobeable` when torch is absent. The classification fixture's `scale`
knob doubles as the measurement reference for the US-4 scale-mismatch probe
and as UB-2's scale-spanning input (the R_0=2000-vs-[0,1] class).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SCALES = ("standardized", "zero_one", "raw255")


# ---------------------------------------------------------------------------
# Classification (active_learning / supervised paradigms)
# ---------------------------------------------------------------------------


@dataclass
class ClassificationFixture:
    x_pool: np.ndarray   # (N_pool, n_features) float32
    y_pool: np.ndarray   # (N_pool,) int64
    x_test: np.ndarray
    y_test: np.ndarray
    n_classes: int
    scale: str           # member of SCALES
    seed: int

    @property
    def chance(self) -> float:
        return 1.0 / self.n_classes

    @property
    def value_range(self) -> tuple[float, float]:
        return float(self.x_pool.min()), float(self.x_pool.max())


def _apply_scale(x: np.ndarray, scale: str) -> np.ndarray:
    """Deterministic affine maps from the native ~[-1.6, 1.6] prototype space."""
    if scale == "standardized":
        return x
    x01 = np.clip((x + 2.0) / 4.0, 0.0, 1.0)
    if scale == "zero_one":
        return x01
    if scale == "raw255":
        return x01 * 255.0
    raise ValueError(f"unknown scale {scale!r}; expected one of {SCALES}")


def make_classification_fixture(
    n_classes: int = 3,
    n_features: int = 16,
    n_per_class_pool: int = 18,
    n_per_class_test: int = 6,
    scale: str = "zero_one",
    seed: int = 0,
    noise: float = 0.18,
) -> ClassificationFixture:
    """Trivially separable blobs: class c sits at prototype linspace(-1,1)[c].

    Separable by construction (nearest-prototype accuracy >= 0.9 is asserted
    by the harness self-calibration tests), so a method that fails to beat
    chance on this fixture is broken, not under-resourced.
    """
    rng = np.random.default_rng(seed)
    protos = np.linspace(-1.0, 1.0, n_classes)

    def _split(n_per_class: int) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for c in range(n_classes):
            base = np.full((n_per_class, n_features), protos[c], dtype=np.float64)
            xs.append(base + rng.normal(0.0, noise, size=base.shape))
            ys.append(np.full(n_per_class, c, dtype=np.int64))
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        order = rng.permutation(len(y))
        return x[order], y[order]

    x_pool, y_pool = _split(n_per_class_pool)
    x_test, y_test = _split(n_per_class_test)
    return ClassificationFixture(
        x_pool=_apply_scale(x_pool, scale).astype(np.float32),
        y_pool=y_pool,
        x_test=_apply_scale(x_test, scale).astype(np.float32),
        y_test=y_test,
        n_classes=n_classes,
        scale=scale,
        seed=seed,
    )


def nearest_prototype_accuracy(fixture: ClassificationFixture) -> float:
    """Self-calibration: a trivial classifier must ace the fixture."""
    protos = np.stack([
        fixture.x_pool[fixture.y_pool == c].mean(axis=0)
        for c in range(fixture.n_classes)
    ])
    dists = ((fixture.x_test[:, None, :] - protos[None, :, :]) ** 2).sum(axis=2)
    pred = dists.argmin(axis=1)
    return float((pred == fixture.y_test).mean())


# ---------------------------------------------------------------------------
# Planning (motion_planning)
# ---------------------------------------------------------------------------


@dataclass
class PlanningFixture:
    start: tuple[float, float, float]      # x, y, theta
    goal: tuple[float, float]
    obstacle_tracks: np.ndarray            # (n_obstacles, n_steps, 2)
    dt: float
    v_candidates: np.ndarray
    omega_candidates: np.ndarray
    dynamic: bool
    seed: int

    @property
    def n_steps(self) -> int:
        return self.obstacle_tracks.shape[1]

    def obstacle_states_at(self, t: int) -> list[dict]:
        """Convenience in the common {'x','y','x_prev','y_prev'} dict shape
        (the convention the existing motion_planning notebooks use)."""
        t = min(t, self.n_steps - 1)
        prev = max(t - 1, 0)
        return [
            {
                "x": float(self.obstacle_tracks[i, t, 0]),
                "y": float(self.obstacle_tracks[i, t, 1]),
                "x_prev": float(self.obstacle_tracks[i, prev, 0]),
                "y_prev": float(self.obstacle_tracks[i, prev, 1]),
            }
            for i in range(self.obstacle_tracks.shape[0])
        ]


def make_planning_fixture(
    n_obstacles: int = 2,
    n_steps: int = 12,
    dynamic: bool = True,
    seed: int = 0,
) -> PlanningFixture:
    """Corridor scenario: robot at origin heading +x, goal at (4, 0),
    obstacles crossing the corridor on linear tracks (v3's "crossing_pair"
    pattern). With dynamic=False the tracks are frozen at their t=0 positions
    — the MP-1 (M-004) contrast case.
    """
    rng = np.random.default_rng(seed)
    tracks = np.zeros((n_obstacles, n_steps, 2))
    for i in range(n_obstacles):
        x0 = 1.5 + rng.uniform(0.0, 2.0)
        y0 = (1.0 + 0.4 * i) * (1 if i % 2 == 0 else -1)
        vy = -0.18 * np.sign(y0)
        for t in range(n_steps):
            if dynamic:
                tracks[i, t] = (x0, y0 + vy * t)
            else:
                tracks[i, t] = (x0, y0)
    return PlanningFixture(
        start=(0.0, 0.0, 0.0),
        goal=(4.0, 0.0),
        obstacle_tracks=tracks,
        dt=0.5,
        v_candidates=np.linspace(0.1, 1.0, 5),
        omega_candidates=np.linspace(-1.0, 1.0, 7),
        dynamic=dynamic,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Knowledge distillation
# ---------------------------------------------------------------------------


@dataclass
class KDLogitsFixture:
    teacher_logits: np.ndarray  # (B, C) — sharp, mostly-correct
    student_logits: np.ndarray  # (B, C) — noisy
    labels: np.ndarray          # (B,)
    seed: int


def make_kd_logits_fixture(
    batch: int = 8, n_classes: int = 5, seed: int = 0
) -> KDLogitsFixture:
    """Teacher logits peak on the true label (margin ~4); student is the
    teacher + heavy noise. KD probes perturb the teacher and assert the loss
    responds (KD-1 teacher-signal-influence)."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, size=batch)
    teacher = rng.normal(0.0, 0.5, size=(batch, n_classes))
    teacher[np.arange(batch), labels] += 4.0
    student = teacher + rng.normal(0.0, 1.5, size=teacher.shape)
    return KDLogitsFixture(
        teacher_logits=teacher, student_logits=student,
        labels=labels, seed=seed,
    )
