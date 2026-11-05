"""The ALCF 24-model zoo benchmark, as a class.

:class:`ZooBenchmark` owns the experiment grid (machines, contexts, horizons),
the resumable per-cell JSON cache, and the three run modes (stage A headline,
stage B 1-D ablations, full context-x-horizon grid). Cell execution order,
filenames, JSON fields, and printed output are identical to the historical
``alcf_model_zoo.py``, so existing partial caches are reused as-is.
"""
from __future__ import annotations

import json
import time

import pandas as pd

from powerforecast import config
from powerforecast.models import FAMILIES, FAMILY_OF, create
from powerforecast.utils import keep_awake, set_seed

from .metrics import score
from .splits import load_series, make_split

MACHINES = ["polaris", "aurora"]
BASE_CONTEXT, BASE_HORIZON = 336, 1
CONTEXTS = [24, 96, 168, 336, 512, 1024, 2048, 3500, 5000]
HORIZONS = [1, 6, 12, 24, 48, 168]

# h=168 foundation cells cost ~50 min each across the full context axis; the weekly
# horizon is covered at the base context by the 1-D horizon sweep. The context x horizon
# grid uses the operationally meaningful short/medium horizons.
GRID_HORIZONS = [1, 6, 24]

OUT = config.OUTPUTS_ROOT / "outputs_alcf_model_zoo"
PART = OUT / "partial"
FIG = OUT / "figures"
MET = OUT / "metrics"


class ZooBenchmark:
    """Resumable ALCF model-zoo benchmark over Polaris + Aurora hourly proxies."""

    def __init__(self, machines=None, out_root=None):
        self.machines = list(machines or MACHINES)
        self.out = out_root or OUT
        self.part = self.out / "partial"
        self.fig = self.out / "figures"
        self.met = self.out / "metrics"
        for d in (self.out, self.part, self.fig, self.met):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- cells ----

    def part_path(self, stage, machine, model, context, horizon):
        return self.part / f"{stage}_{machine}_{model}_c{context}_h{horizon}.json"

    def run_cell(self, stage, machine, model, context, horizon, split, forecaster=None):
        """Run one (stage, machine, model, context, horizon) cell, cached as JSON."""
        pp = self.part_path(stage, machine, model, context, horizon)
        if pp.exists():
            return json.loads(pp.read_text())
        fc = forecaster if forecaster is not None else create(model)
        t0 = time.time()
        pred = fc.predict(split, horizon)
        rec = {"stage": stage, "machine": machine, "model": model, "family": FAMILY_OF[model],
               "context": context, "horizon": horizon, "seconds": round(time.time() - t0, 2),
               **score(split["Yte"], pred, split["scale"])}
        pp.write_text(json.dumps(rec))
        print(f"  [{rec['seconds']:6.1f}s] {machine} {model} c{context} h{horizon} "
              f"MAE={rec['MAE']:.2f} MASE={rec['MASE']:.3f}", flush=True)
        return rec

    def _splits(self, context, horizon):
        return {m: make_split(load_series(m), context, horizon) for m in self.machines}

    def _run_local(self, stage, model, context, horizon, splits):
        for m in self.machines:
            try:
                self.run_cell(stage, m, model, context, horizon, splits[m])
            except Exception as exc:
                print(f"  [FAIL] {m} {model} c{context} h{horizon}: {exc}", flush=True)

    def _run_fm(self, stage, model, context, horizon, splits):
        """Load one foundation model, score every machine still missing, free it."""
        need = [m for m in self.machines if not self.part_path(stage, m, model, context, horizon).exists()]
        if not need:
            return
        fc = create(model)
        try:
            fc.load(context, horizon)
        except Exception as exc:
            print(f"  [SKIP fm] {model} c{context} h{horizon}: {exc}", flush=True)
            return
        for m in need:
            try:
                self.run_cell(stage, m, model, context, horizon, splits[m], forecaster=fc)
            except Exception as exc:
                print(f"  [FAIL] {m} {model} c{context} h{horizon}: {exc}", flush=True)
        fc.unload()

    # ------------------------------------------------------------ stages ----

    def run_stage_a(self, only_family=None, only_model=None):
        """Stage A: context=336, horizon=1, all models, both machines (headline table)."""
        keep_awake(); set_seed()
        splits = self._splits(BASE_CONTEXT, BASE_HORIZON)
        # non-FM first (no GPU contention)
        for fam in ("classical", "ml", "dl"):
            if only_family and fam != only_family:
                continue
            for model in FAMILIES[fam]:
                if only_model and model != only_model:
                    continue
                self._run_local("A", model, BASE_CONTEXT, BASE_HORIZON, splits)
        # FMs: load one at a time, both machines, then free
        if not only_family or only_family == "fm":
            for model in FAMILIES["fm"]:
                if only_model and model != only_model:
                    continue
                self._run_fm("A", model, BASE_CONTEXT, BASE_HORIZON, splits)

    def run_stage_b(self, models):
        """Stage B: 1-D context sweep (h=1) and horizon sweep (c=336) on top models."""
        keep_awake(); set_seed()
        print(f"Stage B models: {models}", flush=True)
        # CONTEXT sweep (horizon=1)
        for context in CONTEXTS:
            splits = self._splits(context, 1)
            for model in [m for m in models if FAMILY_OF[m] != "fm"]:
                self._run_local("B", model, context, 1, splits)
            for model in [m for m in models if FAMILY_OF[m] == "fm"]:
                self._run_fm("B", model, context, 1, splits)
        # HORIZON sweep (context=336)
        for horizon in HORIZONS:
            splits = self._splits(BASE_CONTEXT, horizon)
            for model in [m for m in models if FAMILY_OF[m] != "fm"]:
                self._run_local("B", model, BASE_CONTEXT, horizon, splits)
            for model in [m for m in models if FAMILY_OF[m] == "fm"]:
                self._run_fm("B", model, BASE_CONTEXT, horizon, splits)

    def run_grid(self, models, horizons=None):
        """Full context x horizon grid, written as 'B' cells so existing 1-D sweeps are
        reused. FMs are reloaded per cell (fresh memory) for crash-safety on 8 GB GPU."""
        keep_awake(); set_seed()
        horizons = horizons or GRID_HORIZONS
        print(f"Grid models={models} contexts={CONTEXTS} horizons={horizons}", flush=True)
        for context in CONTEXTS:
            for horizon in horizons:
                splits = self._splits(context, horizon)
                for model in models:
                    fam = FAMILY_OF[model]
                    # LSTM cuDNN workspace OOMs the 8 GB GPU and is too slow on CPU past
                    # ~2k steps; skip those corners (an LSTM gains nothing from 3.5-5k steps).
                    if model == "LSTM" and context > 2048:
                        continue
                    # FM inference OOMs the 8 GB GPU at the two largest contexts when the
                    # horizon is multi-step; the full 9-context sweep already covers h=1, so
                    # restrict the multi-horizon grid to context <= 2048 for foundation models.
                    if fam == "fm" and horizon > 1 and context > 2048:
                        continue
                    if fam == "fm":
                        self._run_fm("B", model, context, horizon, splits)
                    else:
                        need = [m for m in self.machines
                                if not self.part_path("B", m, model, context, horizon).exists()]
                        if need:
                            self._run_local("B", model, context, horizon, splits)

    # ----------------------------------------------------------- results ----

    def collect(self):
        rows = [json.loads(p.read_text()) for p in self.part.glob("*.json")]
        return pd.DataFrame(rows)

    def top_models(self, k=6):
        df = self.collect()
        a = df[(df.stage == "A") & (df.horizon == 1)]
        rank = a.groupby("model")["MASE"].mean().sort_values()
        chosen = list(rank.index[:k])
        for must in ("Persistence", "TimesFM_2p5"):
            if must not in chosen:
                chosen.append(must)
        return chosen

    def aggregate(self):
        df = self.collect()
        if df.empty:
            print("no partials yet"); return
        df.to_csv(self.met / "model_zoo_all_results.csv", index=False)
        # Stage A headline gain vs persistence
        a = df[(df.stage == "A") & (df.horizon == 1)].copy()
        gains = []
        for m in self.machines:
            sub = a[a.machine == m]
            pers = sub[sub.model == "Persistence"]["MAE"].values
            if not len(pers):
                continue
            pers = pers[0]
            for _, r in sub.iterrows():
                gains.append({"machine": m, "model": r.model, "family": r.family, "MAE": r.MAE,
                              "MASE": r.MASE, "sMAPE": r.sMAPE,
                              "gain_vs_persistence_pct": 100 * (pers - r.MAE) / pers})
        if not gains:
            print("(no Persistence baseline yet — run classical family to enable gains)")
            return df, None
        g = pd.DataFrame(gains).sort_values(["machine", "MAE"])
        g.to_csv(self.met / "stage_a_headline.csv", index=False)
        print("\n=== Stage A headline (sorted by MAE) ===")
        print(g.round(2).to_string(index=False))
        return df, g


def collect():
    """Module-level convenience used by the figure/report scripts."""
    return ZooBenchmark().collect()
