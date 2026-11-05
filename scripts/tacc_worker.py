"""Cluster task runner for the tuned-baseline study (TACC Vista).

A task is one (model, trace, horizon): its Optuna search followed by its 3-seed final run. Many workers
share one GPU node; each claims tasks with an atomic mkdir under outputs/dl_hpo/claims/, so any number
of workers and jobs can run at once. A task is finished when its final/*.json exists. Claims left by a
job that died are released by `python tacc_worker.py release` (run at the start of every job).

    python scripts/tacc_worker.py list
    python scripts/tacc_worker.py release
    python scripts/tacc_worker.py work --worker 3
    python scripts/tacc_worker.py status

Needs PF_STUDY_SPLIT=1 (one SQLite file per study).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_dl_hpo as dl  # noqa: E402
import run_fm_sweep as sw  # noqa: E402

CLAIMS = dl.OUT / "claims"
CLAIMS.mkdir(exist_ok=True)
ML = ["Ridge", "LightGBM"]


def tasks():
    """Most expensive first, so the long ones start early and the tail of the run stays short."""
    order = sorted(dl.SLOWEST) + sorted(dl.HEAVY - dl.SLOWEST) + \
        [m for m in dl.DL_MODELS if m not in dl.HEAVY] + ML
    return [(m, tr, H) for m in order for H in sw.HORIZONS for tr in sw.TRACES if sw.feasible(tr, H)]


def tag(m, tr, H):
    return f"{m}__{tr}__H{H}"


def finished(m, tr, H):
    return (dl.FINAL / f"{tr}_H{H}_{m}.json").exists()


def release():
    """Drop claims of unfinished tasks whose job is no longer running."""
    try:
        live = set(subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i"],
                                  capture_output=True, text=True).stdout.split())
    except FileNotFoundError:
        live = set()
    me = os.environ.get("SLURM_JOB_ID")
    n = 0
    for d in CLAIMS.iterdir():
        job = (d / "job").read_text().strip() if (d / "job").exists() else ""
        m, tr, H = d.name.split("__")
        if not finished(m, tr, int(H[1:])) and job not in live - {me} and not (d / "failed").exists():
            shutil.rmtree(d, ignore_errors=True)
            n += 1
    print(f"released {n} stale claims")


def work(worker, only=None):
    import run_ml_hpo as ml
    job = os.environ.get("SLURM_JOB_ID", "local")
    for m, tr, H in tasks():
        if finished(m, tr, H) or (only and m not in only):
            continue
        d = CLAIMS / tag(m, tr, H)
        try:
            d.mkdir()
        except FileExistsError:
            continue
        (d / "job").write_text(job)
        t0 = time.time()
        print(f"[w{worker}] start {tag(m, tr, H)}", flush=True)
        try:
            if m in ML:
                ml.run_hpo(m, tr, H)
                ml.run_final(m, tr, H)
            else:
                dl.run_hpo(m, tr, H, dl.n_trials(m))
                dl.run_final(m, tr, H)
            print(f"[w{worker}] done  {tag(m, tr, H)} in {(time.time() - t0) / 60:.1f} min", flush=True)
        except Exception:
            (d / "failed").write_text(traceback.format_exc())
            print(f"[w{worker}] FAILED {tag(m, tr, H)}\n{traceback.format_exc()}", flush=True)


def status():
    ts = tasks()
    done = [t for t in ts if finished(*t)]
    claimed = [t for t in ts if (CLAIMS / tag(*t)).exists() and not finished(*t)]
    failed = [t for t in ts if (CLAIMS / tag(*t) / "failed").exists()]
    print(f"tasks {len(ts)}  done {len(done)}  in progress {len(claimed) - len(failed)}  failed {len(failed)}  "
          f"queued {len(ts) - len(done) - len(claimed)}")
    for t in failed:
        print("  failed:", tag(*t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["list", "release", "work", "status"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--models", nargs="*", help="restrict this worker to these models (e.g. Ridge LightGBM)")
    a = ap.parse_args()
    if a.mode == "list":
        for t in tasks():
            print(tag(*t), "done" if finished(*t) else "")
    elif a.mode == "release":
        release()
    elif a.mode == "work":
        work(a.worker, a.models)
    else:
        status()


if __name__ == "__main__":
    main()
