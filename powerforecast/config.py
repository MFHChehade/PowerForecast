"""Central paths for the PowerForecast benchmark.

Every machine-specific location lives here so the rest of the code is portable.
Defaults are repo-relative (a fresh clone works once you place the datasets under
``data/``); override any of them with environment variables:

    PF_DATA_ROOT      input datasets            (default: <repo>/data)
    PF_ALCF_RAW       raw ALCF .csv.gz logs     (default: <repo>/data/alcf_raw)
    PF_OUTPUTS_ROOT   where results are written (default: <repo>/outputs)
    PF_PYTHON         python used for report subprocess steps (default: current)
    PF_LATEX_COMPILE  optional LaTeX-compile helper for auto-building PDF reports
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent

# Input datasets. Expected layout under DATA_ROOT:
#   mit_power_sample.parquet, power_sample.parquet,
#   extension_data/{google,bitbrains,lnn_ai_datacenter,ai_power_profiles}/...
DATA_ROOT = Path(os.environ.get("PF_DATA_ROOT", REPO_ROOT / "data"))
EXTENSION_ROOT = DATA_ROOT / "extension_data"

# Raw ALCF job-scheduler logs (the ANL-ALCF-DJC-*.csv.gz files).
ALCF_RAW_ROOT = Path(os.environ.get("PF_ALCF_RAW", DATA_ROOT / "alcf_raw"))

# All result folders (outputs_*) are written under here.
OUTPUTS_ROOT = Path(os.environ.get("PF_OUTPUTS_ROOT", REPO_ROOT / "outputs"))
OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)

# Used only by the optional report-auto-compilation steps.
PYTHON_EXE = Path(os.environ.get("PF_PYTHON", shutil.which("python") or "python"))
_latex = os.environ.get("PF_LATEX_COMPILE")
LATEX_COMPILE = Path(_latex) if _latex else None
