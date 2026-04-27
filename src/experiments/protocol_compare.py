from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from conformal.conformal_localizer import ConformalLocalizer


@dataclass(frozen=True)
class LSudden3Stream:
    X_train: np.ndarray
    y_train: np.ndarray
    X_stream: np.ndarray
    y_stream: np.ndarray
    phase_idx: np.ndarray
    drift_positions: list[int]


def _rotation_matrix(n_features: int, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    rot = np.eye(n_features)
    rot[0, 0], rot[0, 1] = c, -s
    rot[1, 0], rot[1, 1] = s, c
    return rot


def _make_phase(
    rng: np.random.Generator,
    n_per_class: int,
    n_features: int,
    noise_std: float,
    rotation: float,
) -> tuple[np.ndarray, np.ndarray]:
    base_mean = np.zeros(n_features)
    base_mean[0] = 1.5
    rot = _rotation_matrix(n_features=n_features, angle=rotation)
    mean0 = rot @ base_mean
    mean1 = -rot @ base_mean

    x0 = rng.normal(loc=mean0, scale=noise_std, size=(n_per_class, n_features))
    x1 = rng.normal(loc=mean1, scale=noise_std, size=(n_per_class, n_features))
    X = np.vstack([x0, x1])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)]).astype(int)

    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def make_l_sudden3_stream(
    seed: int = 42,
    n_per_phase: int = 1_000,
    n_phases: int = 4,
    n_features: int = 10,
    noise_std: float = 0.6,
    drift_angle: float = np.pi / 2,
) -> LSudden3Stream:
    rng = np.random.default_rng(seed)
    phases_X, phases_y = [], []
    for phase in range(n_phases):
        Xp, yp = _make_phase(
            rng=rng,
            n_per_class=n_per_phase // 2,
            n_features=n_features,
            noise_std=noise_std,
            rotation=phase * drift_angle,
        )
        phases_X.append(Xp)
        phases_y.append(yp)

    X_train, y_train = phases_X[0], phases_y[0]
    X_stream = np.vstack(phases_X)
    y_stream = np.hstack(phases_y)
    phase_idx = np.repeat(np.arange(n_phases), n_per_phase)
    drift_positions = [n_per_phase * p for p in range(1, n_phases)]

    return LSudden3Stream(
        X_train=X_train,
        y_train=y_train,
        X_stream=X_stream,
        y_stream=y_stream,
        phase_idx=phase_idx,
        drift_positions=drift_positions,
    )


def make_drift_window_labels(n_stream: int, drift_positions: list[int], window: int) -> np.ndarray:
    labels = np.zeros(n_stream, dtype=int)
    for drift_pos in drift_positions:
        labels[drift_pos : drift_pos + window] = 1
    return labels


def _make_localizer(alpha: float, random_state: int, n_jobs: int) -> ConformalLocalizer:
    base_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return ConformalLocalizer(
        model=base_model,
        alpha=alpha,
        n_min_members=50,
        bootstrap_fraction=0.8,
        cover_search=20,
        cv_params={},
        n_jobs=n_jobs,
    )


def run_protocol_comparison(
    seed: int = 42,
    alpha: float = 0.2,
    drift_window: int = 50,
    n_jobs: int = -1,
) -> dict[str, float]:
    stream = make_l_sudden3_stream(seed=seed)

    # Notebook protocol: train once on phase 0 labels and score the full stream.
    notebook_localizer = _make_localizer(alpha=alpha, random_state=seed, n_jobs=n_jobs)
    notebook_localizer.fit(stream.X_train, stream.y_train)
    p_notebook = notebook_localizer.score_samples(stream.X_stream)
    score_notebook = 1.0 - p_notebook

    drift_window_labels = make_drift_window_labels(
        n_stream=stream.X_stream.shape[0],
        drift_positions=stream.drift_positions,
        window=drift_window,
    )
    post_drift_labels = (stream.phase_idx > 0).astype(int)

    notebook_auc_window = roc_auc_score(drift_window_labels, score_notebook)
    notebook_auc_post = roc_auc_score(post_drift_labels, score_notebook)

    # Original-script-like protocol: fit on before/after labels over the full stream.
    observation_state = (stream.phase_idx == 0).astype(int)
    original_like_localizer = _make_localizer(alpha=alpha, random_state=seed, n_jobs=n_jobs)
    original_like_localizer.fit(stream.X_stream, observation_state)
    p_original_like = original_like_localizer.score_samples(stream.X_stream)
    score_original_like = 1.0 - p_original_like
    original_like_auc_post = roc_auc_score(post_drift_labels, score_original_like)

    return {
        "notebook_auc_window": notebook_auc_window,
        "notebook_auc_post_drift": notebook_auc_post,
        "original_like_auc_post_drift": original_like_auc_post,
    }

