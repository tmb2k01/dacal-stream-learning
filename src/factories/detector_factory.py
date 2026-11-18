from drift import (
    ADWINDetector,
    AdaptiveMMDDriftDetector,
    C2STDriftDetector,
    CUSUMDriftDetector,
    DDMDetector,
    KSDriftDetector,
    MMDDriftDetector,
    OracleDriftDetector,
    PageHinkleyDetector,
    RunningStatsBaselineDetector,
)

# Detectors that consume the 0/1 prediction error and thus need true labels.
SUPERVISED_DETECTORS = {"baseline", "running_stats_baseline", "adwin", "ddm", "page_hinkley"}
# Detectors that consume label-free signals (embeddings / confidences).
UNSUPERVISED_DETECTORS = {"mmd", "adaptive_mmd", "ks", "c2st", "cusum"}


class DriftDetectorFactory:
    @staticmethod
    def create(config, drift_positions=None):
        detector_cfg = config.get("drift_detector", {})
        if not detector_cfg or not detector_cfg.get("enabled", True):
            return None

        name = detector_cfg.get("name")
        if name in {None, "none"}:
            return None
        seed = detector_cfg.get("seed", config.get("seed", 0))

        detector = None
        if name in {"baseline", "running_stats_baseline"}:
            detector = RunningStatsBaselineDetector(
                warmup_size=detector_cfg.get("warmup_size", 50),
                threshold_std=detector_cfg.get("threshold_std", 3.0),
                consecutive=detector_cfg.get("consecutive", 3),
                min_std=detector_cfg.get("min_std", 1e-8),
            )
        elif name == "adwin":
            detector = ADWINDetector(delta=detector_cfg.get("delta", 0.002))
        elif name == "ddm":
            detector = DDMDetector(
                warm_start=detector_cfg.get("warm_start", 30),
                warning_threshold=detector_cfg.get("warning_threshold", 2.0),
                drift_threshold=detector_cfg.get("drift_threshold", 3.0),
            )
        elif name == "page_hinkley":
            detector = PageHinkleyDetector(
                delta=detector_cfg.get("delta", 0.005),
                threshold=detector_cfg.get("threshold", 50.0),
                min_instances=detector_cfg.get("min_instances", 30),
                alpha=detector_cfg.get("alpha", 0.9999),
                mode=detector_cfg.get("mode", "both"),
            )
        elif name in {"mmd", "adaptive_mmd"}:
            kwargs = {
                "window_size": detector_cfg.get("window_size", 200),
                "test_every": detector_cfg.get("test_every", 50),
                "alpha": detector_cfg.get("alpha", 0.05),
                "n_permutations": detector_cfg.get("n_permutations", 200),
                "pca_components": detector_cfg.get("pca_components"),
                "seed": seed,
            }
            if name == "adaptive_mmd":
                detector = AdaptiveMMDDriftDetector(
                    rebuild_size=detector_cfg.get("rebuild_size"), **kwargs
                )
            else:
                detector = MMDDriftDetector(**kwargs)
        elif name == "ks":
            detector = KSDriftDetector(
                window_size=detector_cfg.get("window_size", 200),
                test_every=detector_cfg.get("test_every", 50),
                alpha=detector_cfg.get("alpha", 0.05),
                signal=detector_cfg.get("signal", "confidence"),
                seed=seed,
            )
        elif name == "c2st":
            detector = C2STDriftDetector(
                window_size=detector_cfg.get("window_size", 200),
                test_every=detector_cfg.get("test_every", 100),
                threshold_auc=detector_cfg.get("threshold_auc", 0.6),
                cv=detector_cfg.get("cv", 5),
                n_estimators=detector_cfg.get("n_estimators", 50),
                max_depth=detector_cfg.get("max_depth", 3),
                pca_components=detector_cfg.get("pca_components"),
                seed=seed,
            )
        elif name == "cusum":
            detector = CUSUMDriftDetector(
                k=detector_cfg.get("k", 0.5),
                h=detector_cfg.get("h", 5.0),
                block_size=detector_cfg.get("block_size", 25),
                signal=detector_cfg.get("signal", "confidence"),
            )
        elif name == "oracle":
            positions = detector_cfg.get("drift_positions", drift_positions)
            if positions is None:
                raise ValueError("oracle detector needs drift_positions (from config or dataset)")
            detector = OracleDriftDetector(drift_positions=positions, delay=detector_cfg.get("delay", 0))
        else:
            raise ValueError(f"Unsupported drift detector: {name!r}")

        if detector is not None and "signal" in detector_cfg:
            detector.signal = detector_cfg["signal"]
        return detector
