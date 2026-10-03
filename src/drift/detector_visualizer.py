"""detector_visualizer.py – Visualization utilities for drift detectors"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from drift.base import BaseDriftDetector, DriftEvent

# ---------------------------------------------------------------------------
# Standalone helper
# ---------------------------------------------------------------------------


def replay_detector(
    detector: BaseDriftDetector,
    values: list[float],
) -> list[DriftEvent]:
    """Feed *values* into *detector* one-by-one and return all DriftEvents.

    The detector is *not* reset before or after the replay – call
    ``detector.reset()`` yourself if a clean run is needed.
    """
    return [detector.update(float(v)) for v in values]


# ---------------------------------------------------------------------------
# Main visualizer
# ---------------------------------------------------------------------------


class DetectorVisualizer:
    """Multi-panel figure showing how a drift detector triggers over time.

    Parameters
    ----------
    detector:
        The detector whose attributes (e.g. ``threshold_std``) are read to
        annotate the score panel.  Pass ``None`` to skip annotations.
    window:
        Window size for the rolling error rate (panel 4).  Defaults to 50.
    figsize:
        Passed directly to ``plt.subplots``.
    """

    def __init__(
        self,
        detector: BaseDriftDetector | None = None,
        window: int = 50,
        figsize: tuple[float, float] = (14, 10),
    ) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.detector = detector
        self.window = window
        self.figsize = figsize

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot(
        self,
        records: list[dict[str, Any]],
        title: str = "Detector Diagnostics",
        save_path: str | None = None,
    ) -> plt.Figure:
        """Build a diagnostic figure from ``SimulationResult.records``.

        Parameters
        ----------
        records:
            The ``records`` list returned by ``StreamSimulationEngine.run()``.
            Each entry must contain ``"drift_event"`` (a :class:`DriftEvent` or
            ``None``); a ``"correct"`` key is optional but enables panel 4.
        title:
            Super-title of the figure.
        save_path:
            If given, the figure is saved to this path before being returned.
        """
        steps = [r.get("step", i) for i, r in enumerate(records)]
        events = [r.get("drift_event") for r in records]
        corrects = [r.get("correct") for r in records]
        return self._build_figure(
            steps=steps,
            events=events,
            corrects=corrects,
            title=title,
            save_path=save_path,
        )

    def plot_events(
        self,
        events: list[DriftEvent | None],
        steps: list[int] | None = None,
        title: str = "Detector Diagnostics",
        save_path: str | None = None,
    ) -> plt.Figure:
        """Build a diagnostic figure from a raw list of :class:`DriftEvent`.

        Useful when events were collected via :func:`replay_detector` instead of
        an engine run.  Panel 4 (rolling error rate) is omitted.

        Parameters
        ----------
        events:
            List of :class:`DriftEvent` objects (or ``None`` entries for
            warmup steps that produced no event).
        steps:
            Optional explicit step indices.  Defaults to ``range(len(events))``.
        title:
            Super-title of the figure.
        save_path:
            If given, the figure is saved to this path before being returned.
        """
        if steps is None:
            steps = list(range(len(events)))
        elif len(steps) != len(events):
            raise ValueError("steps and events must have the same length")
        return self._build_figure(
            steps=steps,
            events=events,
            corrects=None,
            title=title,
            save_path=save_path,
        )

    # ------------------------------------------------------------------
    # Internal build
    # ------------------------------------------------------------------

    def _build_figure(
        self,
        steps: list[int],
        events: list[DriftEvent | None],
        corrects: list[bool | None] | None,
        title: str,
        save_path: str | None,
    ) -> plt.Figure:
        has_correct = corrects is not None and any(c is not None for c in corrects)
        n_panels = 4 if has_correct else 3
        height_ratios = [3, 1.5, 1.5, 2] if has_correct else [3, 1.5, 1.5]

        fig, axes = plt.subplots(
            n_panels,
            1,
            figsize=self.figsize,
            sharex=True,
            gridspec_kw={"height_ratios": height_ratios},
        )
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

        steps_arr = np.array(steps)
        scores = np.array([e.score if e is not None else np.nan for e in events], dtype=float)
        warnings = np.array([bool(e.warning) if e is not None else False for e in events])
        drifts = np.array([bool(e.drift) if e is not None else False for e in events])

        self._panel_score(axes[0], steps_arr, scores, warnings, drifts)
        self._panel_flags(axes[1], steps_arr, warnings, drifts)
        self._panel_cumulative(axes[2], steps_arr, drifts)

        if has_correct and n_panels == 4 and corrects is not None:
            correct_arr = np.array(
                [float(c) if c is not None else np.nan for c in corrects], dtype=np.float64
            )
            self._panel_rolling_error(axes[3], steps_arr, correct_arr)

        axes[-1].set_xlabel("Stream Step", fontsize=11)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    # ------------------------------------------------------------------
    # Individual panels
    # ------------------------------------------------------------------

    def _panel_score(
        self,
        ax: plt.Axes,
        steps: np.ndarray,
        scores: np.ndarray,
        warnings: np.ndarray,
        drifts: np.ndarray,
    ) -> None:
        """Panel 1 – detector score over time with threshold reference."""
        ax.plot(steps, scores, color="#4C72B0", linewidth=1.0, label="Score", zorder=2)

        # Shade warmup region when detector provides a warmup_size; otherwise fall back to leading NaNs.
        warmup_size = getattr(self.detector, "warmup_size", None) if self.detector is not None else None
        if isinstance(warmup_size, int) and 0 < warmup_size <= len(steps):
            ax.axvspan(steps[0], steps[warmup_size - 1], color="#DDDDDD", alpha=0.5, label="Warmup")
        else:
            first_finite = np.where(~np.isnan(scores))[0]
            if first_finite.size and int(first_finite[0]) > 0:
                ax.axvspan(steps[0], steps[int(first_finite[0])], color="#DDDDDD", alpha=0.5, label="Warmup")

        # Threshold line from detector attributes
        threshold = self._read_threshold()
        if threshold is not None:
            ax.axhline(
                threshold,
                color="crimson",
                linestyle="--",
                linewidth=1.2,
                label=f"Threshold ({threshold:.2g}σ)",
            )

        # Highlight warning and drift steps
        if warnings.any():
            ax.scatter(
                steps[warnings],
                scores[warnings],
                marker="^",
                color="orange",
                s=50,
                zorder=4,
                label="Warning",
            )
        if drifts.any():
            ax.scatter(
                steps[drifts],
                scores[drifts],
                marker="D",
                color="crimson",
                s=60,
                zorder=5,
                label="Drift trigger",
            )

        ax.set_ylabel("Score", fontsize=10)
        ax.set_title("Detector Score", fontsize=11)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.7)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(True, linestyle=":", alpha=0.5)

    def _panel_flags(
        self,
        ax: plt.Axes,
        steps: np.ndarray,
        warnings: np.ndarray,
        drifts: np.ndarray,
    ) -> None:
        """Panel 2 – binary warning / drift flag events."""
        warn_steps = steps[warnings]
        drift_steps = steps[drifts]

        if len(warn_steps):
            ax.vlines(
                warn_steps,
                0,
                1,
                colors="orange",
                linewidth=1.5,
                alpha=0.8,
                label=f"Warning ({len(warn_steps)})",
            )
        if len(drift_steps):
            ax.vlines(
                drift_steps,
                0,
                1,
                colors="crimson",
                linewidth=2.0,
                alpha=0.9,
                label=f"Drift ({len(drift_steps)})",
            )

        ax.set_ylim(-0.1, 1.5)
        ax.set_yticks([])
        ax.set_title("Warning & Drift Events", fontsize=11)
        if warn_steps.size or drift_steps.size:
            ax.legend(fontsize=8, loc="upper left", framealpha=0.7)
        ax.grid(True, axis="x", linestyle=":", alpha=0.4)

    def _panel_cumulative(
        self,
        ax: plt.Axes,
        steps: np.ndarray,
        drifts: np.ndarray,
    ) -> None:
        """Panel 3 – cumulative drift count (step function)."""
        cumulative = np.cumsum(drifts.astype(int))
        ax.step(steps, cumulative, where="post", color="#2ca02c", linewidth=1.8)
        ax.fill_between(steps, cumulative, step="post", alpha=0.15, color="#2ca02c")

        total = int(cumulative[-1]) if len(cumulative) else 0
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(f"Cumulative Drift Triggers  (total = {total})", fontsize=11)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(True, linestyle=":", alpha=0.5)

    def _panel_rolling_error(
        self,
        ax: plt.Axes,
        steps: np.ndarray,
        correct: np.ndarray,
    ) -> None:
        """Panel 4 – rolling error rate (1 – accuracy)."""
        error = np.where(~np.isnan(correct), 1.0 - correct, np.nan).astype(np.float64)
        w = self.window
        kernel = np.ones(w) / w
        valid = ~np.isnan(error)
        rolled = np.full_like(error, np.nan)
        if valid.sum() >= w:
            rolled_vals = np.convolve(error[valid].astype(float), kernel.astype(float), mode="valid")
            rolled_idx = np.where(valid)[0][w - 1 :]
            rolled[rolled_idx] = rolled_vals

        ax.plot(steps, rolled, color="#9467bd", linewidth=1.4, label=f"Rolling error (w={w})")
        ax.axhline(0.5, color="grey", linestyle=":", linewidth=1.0)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Error rate", fontsize=10)
        ax.set_title("Rolling Prediction Error Rate", fontsize=11)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.7)
        ax.grid(True, linestyle=":", alpha=0.5)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_threshold(self) -> float | None:
        """Try to read the threshold value from the attached detector."""
        if self.detector is None:
            return None
        return getattr(self.detector, "threshold_std", None)


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    from drift.baseline_detector import RunningStatsBaselineDetector

    random.seed(0)
    rng = random.Random(0)

    # Synthetic stream: 60 stable steps → 30 drifted steps → 30 stable again
    values = (
        [rng.gauss(0.1, 0.05) for _ in range(60)]
        + [rng.gauss(0.85, 0.1) for _ in range(30)]
        + [rng.gauss(0.1, 0.05) for _ in range(30)]
    )

    detector = RunningStatsBaselineDetector(warmup_size=30, threshold_std=2.5, consecutive=2)
    events = replay_detector(detector, values)

    viz = DetectorVisualizer(detector=detector, window=20)
    fig = viz.plot_events(events, title="RunningStatsBaselineDetector – Synthetic Demo")
    fig.savefig("detector_diagnostics_demo.png", dpi=150, bbox_inches="tight")
    print("Saved → detector_diagnostics_demo.png")
    plt.show()




