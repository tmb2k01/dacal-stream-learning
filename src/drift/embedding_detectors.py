"""Label-free drift detectors on embeddings / confidences.

Formalized from ``playground/CIFAR10C_conformal_drift_merged_upgrade_working.ipynb``
and ``playground/CIFAR10C_multi_conformal_mmd.ipynb``.  Each detector:

1. is fitted on a clean reference sample via :meth:`fit_reference`
   (calibration embeddings / confidences), which also calibrates its
   detection threshold under the no-drift null;
2. consumes one stream sample per :meth:`update` call, buffers a sliding
   test window and runs a two-sample test every ``test_every`` steps;
3. requires **zero** labels.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from drift.base import BaseDriftDetector, DriftEvent


class OnlinePCA:
    """Minimal PCA (SVD) projection fitted on clean reference embeddings."""

    def __init__(self, n_components: int = 32) -> None:
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        self.n_components = n_components
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> OnlinePCA:
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        _, _, vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        self.components_ = vt[: self.n_components]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise ValueError("OnlinePCA is not fitted yet")
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) @ self.components_.T


# ---------------------------------------------------------------------------
# Kernel two-sample statistics
# ---------------------------------------------------------------------------


def median_sigma(X: np.ndarray, max_samples: int = 500, rng=None) -> float:
    """Median pairwise distance heuristic for the RBF bandwidth."""
    rng = np.random.default_rng(rng)
    if len(X) > max_samples:
        X = X[rng.choice(len(X), size=max_samples, replace=False)]
    sq = (X**2).sum(axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * X @ X.T, 0.0)
    iu = np.triu_indices(len(X), k=1)
    return float(np.sqrt(np.median(d2[iu]))) + 1e-6


def _rbf(A: np.ndarray, B: np.ndarray, sigma: float) -> np.ndarray:
    aa = (A**2).sum(axis=1)[:, None]
    bb = (B**2).sum(axis=1)[None, :]
    return np.exp(-np.maximum(aa + bb - 2.0 * A @ B.T, 0.0) / (2.0 * sigma**2))


def mmd2_unbiased(X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
    """Unbiased estimate of the squared RBF-kernel MMD between samples X and Y."""
    n, m = len(X), len(Y)
    if n < 2 or m < 2:
        raise ValueError("MMD needs at least two samples per side")
    kxx = _rbf(X, X, sigma)
    kyy = _rbf(Y, Y, sigma)
    kxy = _rbf(X, Y, sigma)
    return float(
        (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
        + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
        - 2.0 * kxy.mean()
    )


# ---------------------------------------------------------------------------
# Generic windowed detector
# ---------------------------------------------------------------------------


class WindowedTwoSampleDetector(BaseDriftDetector):
    """Sliding-window two-sample detector against a fixed clean reference."""

    signal = "embedding"

    def __init__(
        self,
        window_size: int = 200,
        test_every: int = 50,
        alpha: float = 0.05,
        seed: int = 0,
    ) -> None:
        if window_size < 2:
            raise ValueError("window_size must be >= 2")
        if test_every < 1:
            raise ValueError("test_every must be >= 1")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.window_size = int(window_size)
        self.test_every = int(test_every)
        self.alpha = float(alpha)
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)
        self._window: deque[np.ndarray] = deque(maxlen=self.window_size)
        self._since_test = 0
        self._last_score = 0.0
        self.reference: np.ndarray | None = None
        self.threshold: float | None = None
        self.n_tests = 0

    # -- reference -------------------------------------------------------
    def fit_reference(self, reference: dict[str, Any] | np.ndarray) -> None:
        ref = reference[self.signal] if isinstance(reference, dict) else reference
        ref = self._prepare_reference(np.asarray(ref, dtype=np.float64))
        if len(ref) < 2:
            raise ValueError("reference must contain at least two samples")
        self.reference = ref
        self.threshold = self._calibrate_threshold(ref)

    def _prepare_reference(self, ref: np.ndarray) -> np.ndarray:
        return ref.reshape(len(ref), -1)

    def _transform(self, values: np.ndarray) -> np.ndarray:
        return values.reshape(len(values), -1)

    def _calibrate_threshold(self, ref: np.ndarray) -> float | None:
        return None

    # -- stream ----------------------------------------------------------
    def update(self, value: Any) -> DriftEvent:
        if self.reference is None:
            raise ValueError(f"{type(self).__name__} requires fit_reference() before update()")
        self._window.append(np.asarray(value, dtype=np.float64).reshape(-1))
        self._since_test += 1
        if len(self._window) < self.window_size or self._since_test < self.test_every:
            return DriftEvent(drift=False, warning=False, score=self._last_score)
        self._since_test = 0
        self.n_tests += 1
        window = self._transform(np.vstack(self._window))
        score, drift = self._test(window)
        self._last_score = float(score)
        if drift:
            self._on_drift(window)
        return DriftEvent(drift=bool(drift), warning=False, score=self._last_score)

    def _test(self, window: np.ndarray) -> tuple[float, bool]:
        raise NotImplementedError

    def _on_drift(self, window: np.ndarray) -> None:
        pass

    def reset(self) -> None:
        self._window.clear()
        self._since_test = 0
        self._last_score = 0.0

    def _ref_subsample(self, ref: np.ndarray, size: int) -> np.ndarray:
        replace = len(ref) < size
        return ref[self._rng.choice(len(ref), size=size, replace=replace)]


# ---------------------------------------------------------------------------
# MMD
# ---------------------------------------------------------------------------


class MMDDriftDetector(WindowedTwoSampleDetector):
    """RBF-kernel MMD² test on (PCA-projected) embeddings.

    The detection threshold is the ``1 - alpha`` quantile of MMD² between two
    disjoint window-sized subsamples of the clean reference (a bootstrap /
    permutation null matched to the test-window size).
    """

    signal = "embedding"

    def __init__(
        self,
        window_size: int = 200,
        test_every: int = 50,
        alpha: float = 0.05,
        n_permutations: int = 200,
        pca_components: int | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(window_size=window_size, test_every=test_every, alpha=alpha, seed=seed)
        if n_permutations < 10:
            raise ValueError("n_permutations must be >= 10")
        self.n_permutations = int(n_permutations)
        self.pca_components = pca_components
        self.pca: OnlinePCA | None = None
        self.sigma: float | None = None

    def _prepare_reference(self, ref: np.ndarray) -> np.ndarray:
        ref = ref.reshape(len(ref), -1)
        if self.pca_components is not None and ref.shape[1] > self.pca_components:
            self.pca = OnlinePCA(self.pca_components).fit(ref)
            ref = self.pca.transform(ref)
        self.sigma = median_sigma(ref, rng=self._rng)
        return ref

    def _transform(self, values: np.ndarray) -> np.ndarray:
        values = values.reshape(len(values), -1)
        return self.pca.transform(values) if self.pca is not None else values

    def _calibrate_threshold(self, ref: np.ndarray) -> float:
        w = self.window_size
        null = []
        for _ in range(self.n_permutations):
            if len(ref) >= 2 * w:
                perm = self._rng.permutation(len(ref))
                a, b = ref[perm[:w]], ref[perm[w : 2 * w]]
            else:
                a, b = self._ref_subsample(ref, w), self._ref_subsample(ref, w)
            null.append(mmd2_unbiased(a, b, self.sigma))
        return float(np.quantile(null, 1.0 - self.alpha))

    def _test(self, window: np.ndarray) -> tuple[float, bool]:
        ref_sub = self._ref_subsample(self.reference, len(window))
        stat = mmd2_unbiased(ref_sub, window, self.sigma)
        return stat, stat > self.threshold


class AdaptiveMMDDriftDetector(MMDDriftDetector):
    """MMD detector that re-learns its reference after every alarm.

    After a drift alarm the next ``rebuild_size`` stream samples become the
    new reference (and the null threshold is recalibrated on them), so the
    detector tracks the *new* concept instead of alarming on it forever.
    """

    signal = "embedding"

    def __init__(self, *args, rebuild_size: int | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rebuild_size = int(rebuild_size) if rebuild_size else 2 * self.window_size
        self._rebuild: list[np.ndarray] | None = None
        self.n_rebuilds = 0

    def update(self, value: Any) -> DriftEvent:
        if self._rebuild is not None:
            self._rebuild.append(np.asarray(value, dtype=np.float64).reshape(-1))
            if len(self._rebuild) >= self.rebuild_size:
                raw = np.vstack(self._rebuild)
                self._rebuild = None
                self.n_rebuilds += 1
                ref = self._transform(raw)
                self.sigma = median_sigma(ref, rng=self._rng)
                self.reference = ref
                self.threshold = self._calibrate_threshold(ref)
            return DriftEvent(drift=False, warning=False, score=self._last_score)
        return super().update(value)

    def _on_drift(self, window: np.ndarray) -> None:
        self._rebuild = []


# ---------------------------------------------------------------------------
# KS on confidences
# ---------------------------------------------------------------------------


class KSDriftDetector(WindowedTwoSampleDetector):
    """Two-sample Kolmogorov-Smirnov test on a scalar signal (default: confidence)."""

    signal = "confidence"

    def __init__(
        self,
        window_size: int = 200,
        test_every: int = 50,
        alpha: float = 0.05,
        signal: str = "confidence",
        seed: int = 0,
    ) -> None:
        super().__init__(window_size=window_size, test_every=test_every, alpha=alpha, seed=seed)
        self.signal = signal

    def _prepare_reference(self, ref: np.ndarray) -> np.ndarray:
        return ref.reshape(-1)

    def _test(self, window: np.ndarray) -> tuple[float, bool]:
        result = ks_2samp(self.reference, window.reshape(-1))
        return float(result.statistic), bool(result.pvalue < self.alpha)


# ---------------------------------------------------------------------------
# C2ST
# ---------------------------------------------------------------------------


class C2STDriftDetector(MMDDriftDetector):
    """Classifier two-sample test: cross-validated RF ROC-AUC of reference vs window."""

    def __init__(
        self,
        window_size: int = 200,
        test_every: int = 100,
        threshold_auc: float = 0.6,
        cv: int = 5,
        n_estimators: int = 50,
        max_depth: int = 3,
        pca_components: int | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(
            window_size=window_size,
            test_every=test_every,
            alpha=0.05,
            n_permutations=10,
            pca_components=pca_components,
            seed=seed,
        )
        if not 0.5 <= threshold_auc < 1.0:
            raise ValueError("threshold_auc must be in [0.5, 1)")
        self.threshold_auc = float(threshold_auc)
        self.cv = int(cv)
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)

    def _calibrate_threshold(self, ref: np.ndarray) -> float:
        return self.threshold_auc

    def _test(self, window: np.ndarray) -> tuple[float, bool]:
        ref_sub = self._ref_subsample(self.reference, len(window))
        X = np.vstack([ref_sub, window])
        y = np.hstack([np.zeros(len(ref_sub)), np.ones(len(window))])
        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.seed,
            n_jobs=1,
        )
        splitter = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.seed)
        auc = float(np.mean(cross_val_score(clf, X, y, cv=splitter, scoring="roc_auc")))
        return auc, auc > self.threshold_auc


# ---------------------------------------------------------------------------
# CUSUM
# ---------------------------------------------------------------------------


class CUSUMDriftDetector(BaseDriftDetector):
    """Two-sided CUSUM on block means of a scalar label-free signal.

    The signal (``confidence`` by default, or ``set_size``) is averaged over
    blocks of ``block_size`` samples, standardized with the reference
    mean / standard error, and accumulated as
    ``S+ = max(0, S+ + z - k)``, ``S- = max(0, S- - z - k)``.
    An alarm is raised when ``max(S+, S-) > h``.
    """

    signal = "confidence"

    def __init__(
        self,
        k: float = 0.5,
        h: float = 5.0,
        block_size: int = 25,
        signal: str = "confidence",
    ) -> None:
        if k < 0:
            raise ValueError("k must be >= 0")
        if h <= 0:
            raise ValueError("h must be > 0")
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.k = float(k)
        self.h = float(h)
        self.block_size = int(block_size)
        self.signal = signal
        self.mu0: float | None = None
        self.sigma0: float | None = None
        self._block: list[float] = []
        self._s_pos = 0.0
        self._s_neg = 0.0

    def fit_reference(self, reference: dict[str, Any] | np.ndarray) -> None:
        ref = reference[self.signal] if isinstance(reference, dict) else reference
        ref = np.asarray(ref, dtype=np.float64).reshape(-1)
        if len(ref) < 2:
            raise ValueError("reference must contain at least two samples")
        self.mu0 = float(ref.mean())
        self.sigma0 = float(ref.std() / np.sqrt(self.block_size)) + 1e-8

    def update(self, value: float) -> DriftEvent:
        if self.mu0 is None or self.sigma0 is None:
            raise ValueError("CUSUMDriftDetector requires fit_reference() before update()")
        self._block.append(float(np.asarray(value, dtype=np.float64).reshape(-1)[0]))
        score = max(self._s_pos, self._s_neg)
        if len(self._block) < self.block_size:
            return DriftEvent(drift=False, warning=False, score=score)
        z = (float(np.mean(self._block)) - self.mu0) / self.sigma0
        self._block.clear()
        self._s_pos = max(0.0, self._s_pos + z - self.k)
        self._s_neg = max(0.0, self._s_neg - z - self.k)
        score = max(self._s_pos, self._s_neg)
        drift = score > self.h
        warning = score > 0.5 * self.h
        if drift:
            self.reset()
        return DriftEvent(drift=drift, warning=warning and not drift, score=score)

    def reset(self) -> None:
        self._block.clear()
        self._s_pos = 0.0
        self._s_neg = 0.0
