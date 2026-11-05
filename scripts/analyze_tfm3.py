"""TimesFM-3.0 on its own (not "the best FM") against the best tuned site model, for the talk.

Same rules as analyze_dl_hpo.py: TimesFM-3.0's context is chosen on validation; the best tuned model is
chosen on validation; gain = (MASE_tuned - MASE_tfm3) / MASE_tuned x 100, mean +- std over the tuned
model's 3 seeds; 95% CI from a paired bootstrap over the 300 test windows.

Writes outputs/why/tfm3/:
    headline_tfm3.csv        TimesFM-3.0 vs best tuned model, 16 settings
    context_tfm3.csv         TimesFM-3.0 restricted to each context L vs best tuned model
    learning_curves_tfm3.csv local models with more data vs TimesFM-3.0
    synth2_tfm3.csv          long synthetic series, excess over the optimum, TimesFM-3.0 only

    python scripts/analyze_tfm3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_dl_hpo as ad  # noqa: E402
import run_fm_sweep as sw  # noqa: E402

TF = "timesfm_3p0_330m"
W = sw.OUT.parent / "why"
OUT = W / "tfm3"
OUT.mkdir(exist_ok=True)


def gain(ref, other):
    return (ref - other) / ref * 100


def headline(allm):
    rows = []
    for (tr, H), g in allm.groupby(["trace", "H"]):
        t = g[g.group.isin(["dl", "ml"])].sort_values("val_mase").iloc[0]
        f = g[g.model == TF].iloc[0]
        e_f, e_t = ad.errs(f), ad.errs(t)
        per_seed = [gain(es.mean(), e_f.mean()) for es in ad.seed_errs(t)]
        idx = np.random.default_rng(0).integers(0, len(e_f), (2000, len(e_f)))
        boots = gain(e_t[idx].mean(1), e_f[idx].mean(1))
        rows.append(dict(trace=tr, H=H, best_tuned=t.model, tuned_mase=t.test_mase, tfm3_L=f.L,
                         tfm3_mase=f.test_mase, gain_mean=float(np.mean(per_seed)),
                         gain_std=float(np.std(per_seed, ddof=1)), ci_lo=float(np.percentile(boots, 2.5)),
                         ci_hi=float(np.percentile(boots, 97.5))))
    hl = pd.DataFrame(rows)
    hl["verdict"] = np.where(hl.ci_lo > 0, "TimesFM-3.0 better", np.where(hl.ci_hi < 0, "tuned better", "tie"))
    hl.to_csv(OUT / "headline_tfm3.csv", index=False)
    print(hl.round(2).to_string(index=False))
    for H in sw.HORIZONS:
        print(f"H={H}: verdicts", hl[hl.H == H].verdict.value_counts().to_dict())
    return hl


def context(allm):
    rows = []
    for (tr, H), g in allm.groupby(["trace", "H"]):
        t = g[g.group.isin(["dl", "ml"])].sort_values("val_mase").iloc[0]
        for L in sw.CONTEXTS:
            p = sw.cell_path(tr, H, L, TF).with_suffix(".json")
            if p.exists():
                r = json.loads(p.read_text())
                rows.append(dict(trace=tr, H=H, L=L, tuned=t.test_mase, tfm3=r["test_mase"],
                                 gain=gain(t.test_mase, r["test_mase"])))
    c = pd.DataFrame(rows)
    c.to_csv(OUT / "context_tfm3.csv", index=False)
    print("median gain by L:", c.groupby(["H", "L"]).gain.median().round(1).unstack(0).to_string())
    return c


def learning_curves(hl):
    b = pd.read_csv(W / "results" / "h1_learning_curves_best.csv")
    tf = hl.set_index(["trace", "H"]).tfm3_mase
    b["tfm3"] = [tf[(r.trace, int(r.H))] for r in b.itertuples()]
    b["tfm3_gain"] = gain(b.test_mase, b.tfm3)
    b.to_csv(OUT / "learning_curves_tfm3.csv", index=False)


def synth2():
    # oracle MASE per process is already in synth2_results.csv (why_synth2.py analyze)
    oracles = pd.read_csv(W / "synth2" / "synth2_results.csv").set_index("proc").oracle
    procs, ns = list(oracles.index), [1000, 4000, 16000, 64000]
    d = pd.DataFrame([json.loads(p.read_text()) for p in (W / "synth2").glob("*.json")])
    rows = []
    for p in procs:
        o = oracles[p]
        g = d[d.proc == p]
        f = g[(g.kind == "fm") & (g.model == TF)].iloc[0]
        row = dict(proc=p, oracle=o, tfm3_excess=(f.test_mase / o - 1) * 100)
        for n in ns:
            loc = g[(g.kind == "local") & (g.N == n)].sort_values("val_mase").iloc[0]
            row[f"local_excess_N{n}"] = (loc.test_mase / o - 1) * 100
        rows.append(row)
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "synth2_tfm3.csv", index=False)
    print(r.round(2).to_string(index=False))


def main():
    tuned, ref = ad.load_tuned(), ad.load_reference()
    cols = ["trace", "H", "model", "group", "L", "val_mase", "test_mase", "test_mase_std"]
    allm = pd.concat([tuned[cols], ref[cols]], ignore_index=True)
    pd.set_option("display.width", 220)
    hl = headline(allm)
    context(allm)
    learning_curves(hl)
    synth2()


if __name__ == "__main__":
    main()
