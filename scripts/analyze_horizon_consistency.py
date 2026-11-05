"""Summarise horizon_consistency.py: are the first k steps of a longer forecast the same as a k-step forecast?

Differences are in MASE units of the trace. `relative` divides by the k-step forecast's own test MASE, so it
reads as "how large is the change compared with the model's error". A model counts as identical at a setting
when at least 99.9% of windows match to within 1e-4 MASE on every step.

    python scripts/analyze_horizon_consistency.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
DIR = Path(__file__).resolve().parent.parent / "outputs" / "why" / "horizon_consistency"


def main():
    rows = [r for p in DIR.glob("*__*.json") for r in json.loads(p.read_text())]
    d = pd.DataFrame(rows)
    d["relative"] = d.diff_mase / d.base_test_mase
    d["accuracy_change_pct"] = 100 * (d.long_test_mase / d.base_test_mase - 1)  # >0: the long forecast is worse
    d["identical"] = d.exact_frac >= 0.999
    d.to_csv(DIR / "horizon_consistency.csv", index=False)
    pd.set_option("display.width", 200)
    summary = d.groupby(["model", "k", "H"]).agg(
        identical=("identical", "sum"), n=("identical", "size"), median_relative=("relative", "median"),
        max_relative=("relative", "max"), median_acc_change=("accuracy_change_pct", "median"),
        worst_acc_change=("accuracy_change_pct", "max"), best_acc_change=("accuracy_change_pct", "min"),
        max_noise=("rerun_noise", "max")).reset_index()
    summary.to_csv(DIR / "horizon_consistency_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
