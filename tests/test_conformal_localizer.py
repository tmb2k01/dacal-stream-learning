"""Unit tests for ConformalLocalizer and ConformalLocalizerCalibrator."""
import sys
import os

# Ensure src is on the path when running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor

from conformal.conformal_localizer import (
    ConformalLocalizer,
    evaluate_bootstrap,
    get_ib_bootstraps,
)
from conformal.localizer_calibrator import ConformalLocalizerCalibrator
from conformal.util import greedy_cover


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_binary_dataset(n=300, n_features=4, seed=0):
    """Return a balanced binary classification dataset."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


def make_fast_localizer(alpha=0.2):
    """Return a ConformalLocalizer with settings fast enough for unit tests."""
    return ConformalLocalizer(
        model=LogisticRegression(max_iter=200),
        n_min_members=5,
        bootstrap_fraction=0.5,
        cover_search=5,
        alpha=alpha,
        n_jobs=1,
    )


# ---------------------------------------------------------------------------
# greedy_cover
# ---------------------------------------------------------------------------

class TestGreedyCover:
    def test_returns_boolean_mask(self):
        M = np.eye(5)
        result = greedy_cover(M, np.ones(5), d=1)
        assert result.dtype == bool
        assert result.shape == (5,)

    def test_covers_every_column_at_least_d_times(self):
        rng = np.random.default_rng(42)
        M = rng.integers(0, 2, size=(20, 10)).astype(float)
        d = 2
        selected = greedy_cover(M, np.ones(20), d=d)
        # Each column covered by at least d selected rows
        coverage = M[selected].sum(axis=0)
        assert (coverage >= d).all() or selected.all()  # may exhaust rows

    def test_custom_selected_seed(self):
        M = np.eye(5)
        pre = np.array([True, False, False, False, False])
        result = greedy_cover(M, np.ones(5), d=1, selected=pre.copy())
        assert result[0]  # pre-selected row stays selected


# ---------------------------------------------------------------------------
# get_ib_bootstraps
# ---------------------------------------------------------------------------

class TestGetIbBootstraps:
    def test_shape(self):
        bootstraps = get_ib_bootstraps(
            population_size=100, oob_times=5, bootstrap_fraction=0.3, n_repititions=10
        )
        # rows = selected bootstraps, cols = bootstrap size
        assert bootstraps.ndim == 2
        assert bootstraps.shape[1] == int(0.3 * 100)

    def test_indices_in_range(self):
        n = 80
        bootstraps = get_ib_bootstraps(n, oob_times=3, bootstrap_fraction=0.2, n_repititions=5)
        assert bootstraps.min() >= 0
        assert bootstraps.max() < n


# ---------------------------------------------------------------------------
# evaluate_bootstrap
# ---------------------------------------------------------------------------

class TestEvaluateBootstrap:
    def test_returns_p_values_and_messages(self):
        X, y = make_binary_dataset(n=100)
        bootstrap = np.random.choice(100, size=50, replace=True)
        model = LogisticRegression(max_iter=200)
        p_i, msg = evaluate_bootstrap(X, y, bootstrap, model, n_samples=100, min_test_size=2)
        assert p_i.shape == (100,)
        assert isinstance(msg, list)

    def test_p_values_in_unit_interval(self):
        X, y = make_binary_dataset(n=100)
        bootstrap = np.random.choice(100, size=50, replace=True)
        model = LogisticRegression(max_iter=200)
        p_i, _ = evaluate_bootstrap(X, y, bootstrap, model, n_samples=100, min_test_size=2)
        valid = p_i[~np.isnan(p_i)]
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_warns_on_single_class_test_set(self):
        """If the OOB set has only one class, a warning should be returned."""
        X = np.random.randn(50, 2)
        y = np.zeros(50, dtype=int)
        y[:25] = 1
        # Force bootstrap to cover all class-1 samples so test set has only class 0
        bootstrap = np.where(y == 1)[0]
        bootstrap = np.concatenate([bootstrap, bootstrap])  # ensure repetitions
        model = LogisticRegression(max_iter=200)
        _, msg = evaluate_bootstrap(X, y, bootstrap, model, n_samples=50, min_test_size=2)
        assert any("WARN" in m for m in msg)


# ---------------------------------------------------------------------------
# ConformalLocalizer
# ---------------------------------------------------------------------------

class TestConformalLocalizer:
    def test_fit_predict_shape(self):
        X, y = make_binary_dataset(n=200)
        clf = make_fast_localizer()
        clf.fit(X, y)
        preds = clf.predict(X[:10])
        assert preds.shape == (10,)
        assert preds.dtype == bool

    def test_score_samples_range(self):
        X, y = make_binary_dataset(n=200)
        clf = make_fast_localizer()
        clf.fit(X, y)
        scores = clf.score_samples(X[:10])
        assert ((scores >= 0) & (scores <= 1)).all()

    def test_not_fitted_raises(self):
        clf = make_fast_localizer()
        with pytest.raises(ValueError, match="not fitted"):
            clf.score_samples(np.random.randn(5, 4))

    def test_get_info_keys(self):
        clf = make_fast_localizer()
        info = clf.get_info()
        for key in ("base model", "alpha", "min members", "bootstrap fraction"):
            assert key in info

    def test_alpha_respected(self):
        """With alpha=0 nothing should be flagged as anomalous."""
        X, y = make_binary_dataset(n=200)
        clf = make_fast_localizer(alpha=0.0)
        clf.fit(X, y)
        assert not clf.predict(X).any()

    def test_alpha_one_all_anomalous(self):
        """With alpha=1 everything should be flagged as anomalous."""
        X, y = make_binary_dataset(n=200)
        clf = make_fast_localizer(alpha=1.0)
        clf.fit(X, y)
        assert clf.predict(X).all()

    def test_requires_binary_labels(self):
        X = np.random.randn(50, 2)
        y = np.array([0, 1, 2] * 16 + [0, 1])  # 3 classes
        clf = make_fast_localizer()
        with pytest.raises(AssertionError):
            clf.fit(X, y)

    def test_cv_params_no_error(self):
        """GridSearchCV path should not raise when cv_params are provided."""
        X, y = make_binary_dataset(n=200)
        clf = ConformalLocalizer(
            model=LogisticRegression(max_iter=200),
            cv_params={"C": [0.1, 1.0]},
            cv_runs=2,
            n_min_members=5,
            bootstrap_fraction=0.5,
            cover_search=5,
            alpha=0.2,
            n_jobs=1,
        )
        clf.fit(X, y)
        assert clf.localizer_trained is not None


# ---------------------------------------------------------------------------
# ConformalLocalizerCalibrator
# ---------------------------------------------------------------------------

class TestConformalLocalizerCalibrator:
    def _make_calibrator(self, min_recalibrate_samples=10):
        return ConformalLocalizerCalibrator(
            model=LogisticRegression(max_iter=200),
            alpha=0.2,
            n_min_members=5,
            bootstrap_fraction=0.5,
            cover_search=5,
            n_jobs=1,
            min_recalibrate_samples=min_recalibrate_samples,
        )

    # --- predict_set before fitting ---

    def test_predict_set_before_fit_returns_none(self):
        cal = self._make_calibrator()
        out = cal.predict_set({"logits": 0.9}, x=np.random.randn(4))
        assert out["p_value"] is None
        assert out["anomaly"] is None
        assert out["prediction"] == {"logits": 0.9}

    # --- update / buffer accumulation ---

    def test_update_accumulates_buffer(self):
        cal = self._make_calibrator(min_recalibrate_samples=999)
        X, y = make_binary_dataset(n=20)
        for xi, yi in zip(X, y):
            cal.update({"x": xi}, yi)
        assert len(cal._buffer_y) == 20
        assert len(cal._buffer_X) == 20

    def test_update_without_x_key_ignored(self):
        cal = self._make_calibrator()
        cal.update({"logits": 0.5}, y_true=1)
        assert len(cal._buffer_y) == 0

    # --- recalibrate ---

    def test_recalibrate_fits_localizer(self):
        cal = self._make_calibrator(min_recalibrate_samples=999)
        X, y = make_binary_dataset(n=200)
        for xi, yi in zip(X, y):
            cal.update({"x": xi}, yi)
        cal.recalibrate()
        assert cal._localizer.localizer_trained is not None

    def test_recalibrate_with_empty_buffer_is_noop(self):
        cal = self._make_calibrator()
        cal.recalibrate()  # should not raise
        assert cal._localizer.localizer_trained is None

    def test_auto_recalibrate_on_threshold(self):
        """Recalibration is triggered automatically when buffer hits the threshold."""
        n = 200
        cal = self._make_calibrator(min_recalibrate_samples=n)
        X, y = make_binary_dataset(n=n)
        for xi, yi in zip(X, y):
            cal.update({"x": xi}, yi)
        assert cal._localizer.localizer_trained is not None

    # --- predict_set after fitting ---

    def test_predict_set_after_recalibrate(self):
        cal = self._make_calibrator(min_recalibrate_samples=999)
        X, y = make_binary_dataset(n=200)
        for xi, yi in zip(X, y):
            cal.update({"x": xi}, yi)
        cal.recalibrate()

        x_new = X[0]
        out = cal.predict_set({"logits": 0.7}, x=x_new)
        assert out["p_value"] is not None
        assert isinstance(out["anomaly"], bool)
        assert out["prediction"] == {"logits": 0.7}

    def test_predict_set_p_value_in_unit_interval(self):
        cal = self._make_calibrator(min_recalibrate_samples=999)
        X, y = make_binary_dataset(n=200)
        for xi, yi in zip(X, y):
            cal.update({"x": xi}, yi)
        cal.recalibrate()

        for xi in X[:20]:
            out = cal.predict_set({}, x=xi)
            p = np.atleast_1d(out["p_value"])
            assert (p >= 0.0).all() and (p <= 1.0).all()

    # --- reset ---

    def test_reset_clears_buffer(self):
        cal = self._make_calibrator(min_recalibrate_samples=999)
        X, y = make_binary_dataset(n=20)
        for xi, yi in zip(X, y):
            cal.update({"x": xi}, yi)
        cal.reset()
        assert cal._buffer_X == []
        assert cal._buffer_y == []

    def test_reset_clears_fitted_state(self):
        cal = self._make_calibrator(min_recalibrate_samples=999)
        X, y = make_binary_dataset(n=200)
        for xi, yi in zip(X, y):
            cal.update({"x": xi}, yi)
        cal.recalibrate()
        cal.reset()
        assert cal._localizer.localizer_trained is None

    def test_predict_set_returns_none_after_reset(self):
        cal = self._make_calibrator(min_recalibrate_samples=999)
        X, y = make_binary_dataset(n=200)
        for xi, yi in zip(X, y):
            cal.update({"x": xi}, yi)
        cal.recalibrate()
        cal.reset()
        out = cal.predict_set({}, x=X[0])
        assert out["p_value"] is None

