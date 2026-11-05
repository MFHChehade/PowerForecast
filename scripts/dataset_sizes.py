"""Size of every trace as used in the benchmark (after the loader's filtering).

    py -3 scripts/dataset_sizes.py  -> outputs/fm_sweep/dataset_sizes.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_fm_sweep as sw  # noqa: E402

rows = []
for tr in sw.TRACES:
    s = sw.load_trace(tr)
    n = np.array([len(x) for x in s])
    rows.append(dict(trace=tr, series=len(s), readings=int(n.sum()), median_len=int(np.median(n)),
                     min_len=int(n.min()), max_len=int(n.max())))
d = pd.DataFrame(rows)
d.to_csv(sw.OUT / "dataset_sizes.csv", index=False)
print(d.to_string(index=False))
