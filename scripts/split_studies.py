"""Copy every study from the per-model databases (studies/{model}.db) into one database per study
(studies/split/{model}__{trace}__H{H}.db), so cluster workers can resume the laptop search in parallel.
Trials left RUNNING by a killed process are dropped (they never finished and are not counted anyway).

    python scripts/split_studies.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import optuna  # noqa: E402

import run_dl_hpo as dl  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
FINISHED = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.FAIL)

n_st = n_tr = 0
for db in sorted(dl.STUDIES.glob("*.db")):
    src = f"sqlite:///{db.as_posix()}"
    for s in optuna.get_all_study_summaries(src):
        dst = f"sqlite:///{(dl.STUDIES / 'split' / f'{s.study_name}.db').as_posix()}"
        if (dl.STUDIES / "split" / f"{s.study_name}.db").exists():
            continue
        old = optuna.load_study(study_name=s.study_name, storage=src)
        new = optuna.create_study(study_name=s.study_name, storage=dst, direction="minimize")
        trials = [t for t in old.trials if t.state in FINISHED]
        new.add_trials(trials)
        n_st += 1; n_tr += len(trials)
print(f"copied {n_st} studies, {n_tr} finished trials")
