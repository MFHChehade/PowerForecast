# data/ — expected layout

Datasets are **not committed** (size and licensing). Place them here, or point
`PF_DATA_ROOT` at wherever you keep them (see `powerforecast/config.py`). Expected layout:

```
data/
├── mit_power_sample.parquet              # MIT Supercloud (measured GPU power)
├── power_sample.parquet                  # SURF Lisa (measured node power)
├── extension_data/
│   ├── google/aggregated_google.parquet
│   ├── bitbrains/aggregated_bitbrains.parquet
│   ├── lnn_ai_datacenter/lnn_inference_power.parquet
│   ├── lnn_ai_datacenter/lnn_training_power.parquet
│   └── ai_power_profiles/nlr_training_1s_resampled.parquet
└── alcf_raw/                             # raw ALCF job logs (only to rebuild the hourly series)
    ├── ANL-ALCF-DJC-POLARIS_20250101_20251231.csv.gz
    └── ANL-ALCF-DJC-AURORA_20250127_20251231.csv.gz
```

Each dataset has its own terms. They have not been reviewed for redistribution, so none is bundled here.

Sources: MIT Supercloud (MIT Lincoln Laboratory); SURF Lisa (SURF, NL); Google PowerData
2019 (`gs://powerdata_2019`); Bitbrains GWA-T-12 (Grid Workloads Archive); LNN signatures
(Mughees et al., IEEE TSG 2026); NLR H100 (Vercellino et al., arXiv:2604.07345); ALCF
Polaris/Aurora job logs (Argonne, public 2025).
