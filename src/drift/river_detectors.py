"""River-backed drift detector wrappers.

Each class adapts a ``river.drift`` algorithm to the ``BaseDriftDetector``
interface (``update(value) -> DriftEvent``, ``reset()``).
"""


from river.drift import ADWIN, PageHinkley
from river.drift.binary import DDM

from drift.base import BaseDriftDetector, DriftEvent


class ADWINDetector(BaseDriftDetector):
    """Wrapper around :class:`river.drift.ADWIN`.

    ADWIN is a change-detector and estimator for the mean of a stream of
    real-valued observations.  It raises a drift signal when two sub-windows
    of its adaptive window differ by more than can be explained by chance.

    Parameters
    ----------
    delta:
        Confidence parameter (smaller → less sensitive). Default ``0.002``.
    """

    def __init__(self, delta: float = 0.002) -> None:
        if not (0.0 < float(delta) < 1.0):
            raise ValueError("delta must be in (0, 1)")
        self.delta = float(delta)
        self._inner = ADWIN(delta=self.delta)
    def update(self, value: float) -> DriftEvent:
        self._inner.update(float(value))
        drift = bool(self._inner.drift_detected)
        score = float(self._inner.variance) if self._inner.width > 0 else 0.0
        if drift:
            self.reset()
        return DriftEvent(drift=drift, warning=False, score=score)

    def reset(self) -> None:
        self._inner = ADWIN(delta=self.delta)


class DDMDetector(BaseDriftDetector):
    """Wrapper around :class:`river.drift.binary.DDM` (Drift Detection Method).

    DDM monitors a binary error stream (0 = correct, 1 = error).
    Continuous real-valued inputs are thresholded at 0.5 before being
    forwarded to the inner detector so that scalar loss / score signals
    work out of the box.

    Parameters
    ----------
    warm_start:
        Number of samples to collect before starting drift detection.
        Default ``30``.
    warning_threshold:
        Warning level threshold (in standard deviations). Default ``2.0``.
    drift_threshold:
        Drift level threshold (in standard deviations). Default ``3.0``.
    """

    def __init__(
        self,
        warm_start: int = 30,
        warning_threshold: float = 2.0,
        drift_threshold: float = 3.0,
    ) -> None:
        if warm_start < 1:
            raise ValueError("warm_start must be >= 1")
        if warning_threshold <= 0:
            raise ValueError("warning_threshold must be > 0")
        if drift_threshold <= warning_threshold:
            raise ValueError("drift_threshold must be > warning_threshold")

        self.warm_start = int(warm_start)
        self.warning_threshold = float(warning_threshold)
        self.drift_threshold = float(drift_threshold)
        self._inner = DDM(
            warm_start=self.warm_start,
            warning_threshold=self.warning_threshold,
            drift_threshold=self.drift_threshold,
        )

    def update(self, value: float) -> DriftEvent:
        # DDM expects a binary label; threshold continuous values at 0.5.
        binary_value = bool(float(value) >= 0.5)
        self._inner.update(binary_value)
        drift = bool(self._inner.drift_detected)
        warning = bool(self._inner.warning_detected)
        if drift:
            self.reset()
        return DriftEvent(drift=drift, warning=warning, score=1.0 if binary_value else 0.0)

    def reset(self) -> None:
        self._inner = DDM(
            warm_start=self.warm_start,
            warning_threshold=self.warning_threshold,
            drift_threshold=self.drift_threshold,
        )


class PageHinkleyDetector(BaseDriftDetector):
    """Wrapper around :class:`river.drift.PageHinkley`.

    The Page-Hinkley test is a sequential hypothesis test that detects a
    persistent change in the mean of a stream of values.

    Parameters
    ----------
    delta:
        Minimum amplitude of change to detect. Default ``0.005``.
    threshold:
        Threshold above which drift is signalled. Default ``50.0``.
    min_instances:
        Minimum number of instances before drift can be detected.
        Default ``30``.
    alpha:
        Forgetting factor (exponential weighting). Default ``0.9999``.
    mode:
        Direction of change to detect: ``'up'``, ``'down'``, or ``'both'``.
        Default ``'both'``.
    """

    def __init__(
        self,
        delta: float = 0.005,
        threshold: float = 50.0,
        min_instances: int = 30,
        alpha: float = 0.9999,
        mode: str = "both",
    ) -> None:
        if delta < 0:
            raise ValueError("delta must be >= 0")
        if threshold <= 0:
            raise ValueError("threshold must be > 0")
        if min_instances < 1:
            raise ValueError("min_instances must be >= 1")
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        if mode not in {"up", "down", "both"}:
            raise ValueError("mode must be one of {'up', 'down', 'both'}")

        self.delta = float(delta)
        self.threshold = float(threshold)
        self.min_instances = int(min_instances)
        self.alpha = float(alpha)
        self.mode = mode
        self._inner = PageHinkley(
            delta=self.delta,
            threshold=self.threshold,
            min_instances=self.min_instances,
            alpha=self.alpha,
            mode=self.mode,
        )

    def update(self, value: float) -> DriftEvent:
        self._inner.update(float(value))
        drift = bool(self._inner.drift_detected)
        if drift:
            self.reset()
        return DriftEvent(drift=drift, warning=False, score=float(value))

    def reset(self) -> None:
        self._inner = PageHinkley(
            delta=self.delta,
            threshold=self.threshold,
            min_instances=self.min_instances,
            alpha=self.alpha,
            mode=self.mode,
        )



