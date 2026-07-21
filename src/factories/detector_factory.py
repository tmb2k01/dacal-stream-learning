from drift import RunningStatsBaselineDetector


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

        # if name == "adwin":
        #     return ADWINDetector(delta=detector_cfg["delta"])
        # if name == "ddm":
        #     return DDMDetector()
        # if name == "page_hinkley":
        #     return PageHinkleyDetector(
        #         delta=detector_cfg["delta"],
        #         threshold=detector_cfg["threshold"]
        #     )
        raise ValueError(f"Unsupported drift detector: {name!r}")
