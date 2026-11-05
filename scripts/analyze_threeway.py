"""TimesFM-3.0 as is, fine-tuned and trained from scratch, each compared with the best tuned site model.

Reads the runs of why_tfm3.py and the table of analyze_dl_hpo.py.

    python scripts/analyze_threeway.py  -> outputs/why/tfm3/threeway_tfm3.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "outputs"


def main():
    site = pd.read_csv(OUT / "dl_hpo/headline_tuned.csv").set_index(["trace", "H"])
    rows = []
    for p in sorted((OUT / "why/ft").glob("tfm3__*.json")):
        r = json.loads(p.read_text())
        tuned = site.loc[(r["trace"], r["H"]), "tuned_mase"]
        rows.append(dict(trace=r["trace"], H=r["H"], L=r["L"], mode=r["mode"], test_mase=r["test_mase"],
                         test_mase_std=r["test_mase_std"], best_site_mase=tuned,
                         pct_vs_best_site=100 * (r["test_mase"] / tuned - 1)))
    t = pd.DataFrame(rows).sort_values(["H", "trace", "mode"])
    (OUT / "why/tfm3").mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "why/tfm3/threeway_tfm3.csv", index=False)
    print(t.pivot_table(index=["H", "trace"], columns="mode", values="pct_vs_best_site").round(1).to_string())
    print(t.groupby(["H", "mode"]).pct_vs_best_site.median().round(1).to_string())


if __name__ == "__main__":
    main()
