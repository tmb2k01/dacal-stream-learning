class DriftDetectorFactory:
    @staticmethod
    def create(config):
        detector_cfg = config.get("drift_detector", {})
        if not detector_cfg or not detector_cfg.get("enabled", True):
            return None

        name = detector_cfg.get("name")
        if name in {None, "none"}:
            return None
        raise ValueError(f"Unsupported drift detector: {name!r}")
