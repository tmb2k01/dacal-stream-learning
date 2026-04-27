import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import pandas as pd

from experiments.protocol_compare import run_protocol_comparison


if __name__ == "__main__":
    metrics = run_protocol_comparison(seed=42, alpha=0.2, drift_window=50, n_jobs=-1)
    table = pd.DataFrame(
        [
            {
                "Protocol": "Notebook (drift-window labels)",
                "ROC-AUC": metrics["notebook_auc_window"],
            },
            {
                "Protocol": "Notebook (all post-drift labels)",
                "ROC-AUC": metrics["notebook_auc_post_drift"],
            },
            {
                "Protocol": "Original-like (fit before/after, eval post-drift)",
                "ROC-AUC": metrics["original_like_auc_post_drift"],
            },
        ]
    )

    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
