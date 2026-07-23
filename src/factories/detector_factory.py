from drift import (
    ADWINDetector,
    DDMDetector,
    PageHinkleyDetector,
    RunningStatsBaselineDetector,
)


class DriftDetectorFactory:
    @staticmethod
    def create(config):
        detector_cfg = config.get("drift_detector", {})
        if not detector_cfg or not detector_cfg.get("enabled", True):
            return None

        name = detector_cfg.get("name")
        if name in {None, "none"}:
            return None

        if name in {"baseline", "running_stats_baseline"}:
            return RunningStatsBaselineDetector(
                warmup_size=detector_cfg.get("warmup_size", 50),
                threshold_std=detector_cfg.get("threshold_std", 3.0),
                consecutive=detector_cfg.get("consecutive", 3),
                min_std=detector_cfg.get("min_std", 1e-8),
            )

        if name == "adwin":
            return ADWINDetector(delta=detector_cfg.get("delta", 0.002))

        if name == "ddm":
            return DDMDetector(
                warm_start=detector_cfg.get("warm_start", 30),
                warning_threshold=detector_cfg.get("warning_threshold", 2.0),
                drift_threshold=detector_cfg.get("drift_threshold", 3.0),
            )

        if name == "page_hinkley":
            return PageHinkleyDetector(
                delta=detector_cfg.get("delta", 0.005),
                threshold=detector_cfg.get("threshold", 50.0),
                min_instances=detector_cfg.get("min_instances", 30),
                alpha=detector_cfg.get("alpha", 0.9999),
                mode=detector_cfg.get("mode", "both"),
            )

        raise ValueError(f"Unsupported drift detector: {name!r}")
