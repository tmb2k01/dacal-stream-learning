import sys
from pathlib import Path

# Ensure src is on the path when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.protocol_compare import run_protocol_comparison


def test_protocol_comparison_outputs_valid_auc_values():
    metrics = run_protocol_comparison(seed=7, alpha=0.2, drift_window=30, n_jobs=1)

    expected_keys = {
        "notebook_auc_window",
        "notebook_auc_post_drift",
        "original_like_auc_post_drift",
    }
    assert set(metrics) == expected_keys

    for value in metrics.values():
        assert 0.0 <= value <= 1.0
