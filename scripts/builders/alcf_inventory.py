"""Task 1 — inventory ALCF .csv.gz files (streaming, no full decompression).

Writes outputs_alcf_power_forecast/alcf_inventory.csv and .md:
schema (n cols), row count (chunked; huge files sampled), date range, null fraction,
key timestamp columns.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
for _p in (str(_REPO), str(_REPO / "scripts")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from powerforecast import config

import pandas as pd

DL = config.ALCF_RAW_ROOT
OUT = config.OUTPUTS_ROOT / "outputs_alcf_power_forecast"
HUGE_MB = 200          # files larger than this are sampled, not fully counted
CHUNK = 200_000

FILES = [
    "ANL-ALCF-DJC-POLARIS_20250101_20251231.csv.gz",
    "ANL-ALCF-MACHINESTATUS-POLARIS_20250101_20251231.csv.gz",
    "ANL-ALCF-DJC-AURORA_20250127_20251231.csv.gz",
    "ANL-ALCF-MACHINESTATUS-AURORA_20250127_20251231.csv.gz",
    "ANL-ALCF-GRIDFTP-PROJECT-HOUR_20240101_20241231.csv.gz",
    "ANL-ALCF-GRIDFTP-USER-HOUR_20240101_20241231.csv.gz",
    "ANL-ALCF-DARSHAN-THETA_20230101_20231231.csv.gz",
    "ANL-ALCF-DARSHAN-MIRA_20200101_20201231.csv.gz",
    "ANL-ALCF-HARDWAREERROR-THETA_20240101_20240131.csv.gz",
    "ANL-ALCF-RE-MIRA_20200101_20201231.csv.gz",
    "ANL-ALCF-RE-INTREPID_20140101_20141231.csv.gz",
    "ANL-ALCF-DJC-THETA_20240101_20241231.csv.gz",
    "ANL-ALCF-DJC-THETAGPU_20240101_20241231.csv.gz",
    "ANL-ALCF-DJC-MIRA_20200101_20201231.csv.gz",
    "ANL-ALCF-DJC-COOLEY_20240101_20241231.csv.gz",
    "ANL-ALCF-DJC-INTREPID_20140101_20141231.csv.gz",
]
DATE_COL = {  # preferred column to derive the date range
    "DJC": "START_TIMESTAMP", "MACHINESTATUS": "START_TIMESTAMP",
    "GRIDFTP": "AT_DATE_ID", "DARSHAN": "RUN_DATE_ID", "RE": None, "HARDWAREERROR": None,
}


def family(name):
    for k in ("DJC", "MACHINESTATUS", "GRIDFTP", "DARSHAN", "HARDWAREERROR", "RE"):
        if f"-{k}-" in name or f"-{k}_" in name:
            return k
    return "?"


def header(path):
    with gzip.open(path, "rt", errors="replace") as fh:
        return [c.strip('"') for c in fh.readline().strip().split(",")]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for fn in FILES:
        p = DL / fn
        if not p.exists():
            rows.append({"file": fn, "status": "MISSING"}); continue
        size_mb = p.stat().st_size / 1e6
        cols = header(p)
        fam = family(fn)
        dcol = DATE_COL.get(fam)
        huge = size_mb > HUGE_MB
        n = 0
        dmin = dmax = None
        null_frac = {}
        first_chunk = None
        usecols = [dcol] if (dcol and not huge) else None
        try:
            it = pd.read_csv(p, compression="gzip", chunksize=CHUNK, usecols=usecols,
                             low_memory=False, on_bad_lines="skip")
            for ci, ch in enumerate(it):
                n += len(ch)
                if first_chunk is None:
                    first_chunk = pd.read_csv(p, compression="gzip", nrows=2000, low_memory=False, on_bad_lines="skip")
                    null_frac = (first_chunk.isna().mean().round(3)).to_dict()
                if dcol and dcol in ch.columns:
                    s = ch[dcol]
                    if fam in ("GRIDFTP", "DARSHAN"):
                        s = pd.to_numeric(s, errors="coerce")
                    else:
                        s = pd.to_datetime(s, errors="coerce")
                    lo, hi = s.min(), s.max()
                    dmin = lo if dmin is None else min(dmin, lo)
                    dmax = hi if dmax is None else max(dmax, hi)
                if huge and ci >= 1:  # sample only: stop after 2 chunks
                    n = f">={n} (sampled)"
                    break
        except Exception as exc:
            n = f"err: {exc}"
        ts_cols = [c for c in cols if "TIMESTAMP" in c or "DATE_ID" in c or c in ("START_TIME", "END_TIME")]
        high_null = {k: v for k, v in null_frac.items() if v > 0.5}
        rows.append({
            "file": fn, "family": fam, "size_mb": round(size_mb, 1), "n_cols": len(cols),
            "row_count": n, "date_col": dcol, "date_min": str(dmin), "date_max": str(dmax),
            "n_timestamp_cols": len(ts_cols), "key_timestamp_cols": ";".join(ts_cols[:6]),
            "cols_>50%_null": ";".join(list(high_null)[:8]),
        })
        print(f"  {fn}: rows={n} range=[{dmin}..{dmax}] cols={len(cols)}", flush=True)

    inv = pd.DataFrame(rows)
    inv.to_csv(OUT / "alcf_inventory.csv", index=False)
    lines = ["# ALCF Dataset Inventory", "",
             "Streaming inventory (no full decompression; files >200 MB sampled).", "",
             "| file | family | MB | cols | rows | date range | key timestamps |",
             "|---|---|---|---|---|---|---|"]
    for _, r in inv.iterrows():
        if r.get("status") == "MISSING":
            lines.append(f"| {r['file']} | — | — | — | MISSING | — | — |"); continue
        lines.append(f"| {r['file']} | {r['family']} | {r['size_mb']} | {r['n_cols']} | {r['row_count']} | "
                     f"{r['date_min'][:10]} → {r['date_max'][:10]} | {r['key_timestamp_cols']} |")
    lines += ["", "## Notes",
              "- **Primary forecasting inputs:** DJC-POLARIS, DJC-AURORA (per-job records) + MACHINESTATUS-POLARIS/AURORA (downtime windows). All 2025.",
              "- **No power/GPU telemetry columns** are present in any file → workload proxies used; `GPUS_REQUESTED`, `NODES_USED`, `USED_CORE_HOURS` are the GPU/power proxies.",
              "- **Temporal-overlap caveat:** GRIDFTP (2024) and Darshan (Theta 2023 / Mira 2020) do NOT overlap the 2025 Polaris/Aurora window, so they are inventoried but NOT merged as covariates.",
              "- HARDWAREERROR-THETA (~572 MB) sampled only; old systems (Theta/Mira/Intrepid/Cooley) are cross-system references, not Polaris/Aurora labels."]
    (OUT / "alcf_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT/'alcf_inventory.csv'} and .md")


if __name__ == "__main__":
    main()
