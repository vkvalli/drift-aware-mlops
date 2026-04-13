# Drift-Aware MLOps (Offline-First)

An offline-first, event-driven MLOps research system for drift-aware retraining under delayed labels, with baseline policies and a lightweight RL agent (LinUCB).

## Current Status (v2.2)

- Stable offline pipeline with canonical event logging
- Multi-task support:
  - `classification`
  - `regression`
  - `text_classification`
- Baselines + RL are implemented and evaluated
- AG News real-data path is calibrated (TF-IDF)
- Fairness metrics are integrated as evaluation signals (Adult demo scope)

## Architecture (Current)

The orchestrated flow is:

`Replay/Ingestion -> Monitor -> Policy Agent -> Trainer -> Canonical Logger`

Each stage writes modular outputs, and the run is consolidated into one canonical log:

- `artifacts/events.jsonl` (ordered canonical trace)
- `artifacts/dlq.jsonl` (invalid-schema/error events)
- `artifacts/checkpoints/` (checkpoint/resume metadata where applicable)

## Event Model (Canonical)

The system logs structured event types:

- `batch`
- `state`
- `action`
- `cost`
- `model_update`
- `run_meta` (header/metadata line)

IDs are deterministic and seeded for reproducibility (`b_`, `s_`, `a_`, `c_`, `u_`, `m_`).

## Policies

Implemented policy families:

- `no_retrain`
- `fixed_interval`
- `threshold_persistence`
- `fairness_trigger`
- `linucb`

Trainer action semantics:

- `defer`
- `fine_tune`
- `full_retrain`

## Tasks + Metrics

### Classification
- Metrics include `accuracy`, optional `balanced_accuracy`, `f1`
- Electricity and Airline experiments are supported

### Regression
- Metrics include `mae_pre`, `rmse_pre`, `r2_pre`
- California Housing path is supported

### Text Classification
- Feature mode: `tfidf`
- AG News real dataset path is supported
- Note: SBERT is not implemented in this release

### Fairness (Evaluation Scope)
- Fairness signals logged as `dpd_pre`, `eod_pre`
- Adult fairness demo is scoped to binary classification evaluation

## Quickstart

Use your project virtual environment (recommended).

### 1) Run a single offline pipeline

```bash
python -m src.runner.offline_run \
  --csv data/electricity-normalized.csv \
  --task classification \
  --label-column class \
  --window-size 200 \
  --max-rows 5000 \
  --label-delay-batches 2 \
  --ref-window 3 \
  --policy threshold_persistence \
  --threshold 0.01 \
  --consecutive-k 3 \
  --seed 111 \
  --clean-artifacts \
  --canonical-output artifacts/events.jsonl
```
### 2) Summarize + plot a run

```bash
python tools/summarize_run.py --events artifacts/events.jsonl --out artifacts/run_summary.txt
python tools/plot_run.py --events artifacts/events.jsonl --outdir artifacts/plots
```

### 3) Run a policy matrix

```bash
python -m tools.run_matrix \
  --csv data/electricity-normalized.csv \
  --task classification \
  --label-column class \
  --window-size 200 \
  --max-rows 20000 \
  --label-delay-batches 2 \
  --ref-window 3 \
  --seeds 111,222,333 \
  --policies no_retrain,fixed_interval,threshold_persistence,linucb
```

```bash
LATEST_MATRIX="$(ls -td runs/matrix_* | head -n 1)"
python tools/aggregate_matrix.py --matrix-dir "$LATEST_MATRIX" --out "$LATEST_MATRIX/summary.csv"
```

## Dataset Preparation Helpers
Examples:

```bash
python tools/prepare_california_housing.py --out data/california_housing.csv
python tools/prepare_ag_news.py --source hf --out data/ag_news_real.csv --rows 60000 --seed 42
python tools/prepare_adult.py --out data/adult.csv --protected-attr sex
```

## Reproducibility Notes

Implemented policy families:

- Use fixed seeds for deterministic IDs and reproducible runs
- Normalize logged_at precision in deterministic comparisons
- Keep one run per folder under runs/ (no overwrite)
- Use .venv for consistent dependency behavior locally and in tests

## Testing

Run full tests in your virtual environment:
```bash
.venv/bin/python -m unittest
```

## Scope Notes (Important)

- Offline-first canonical pipeline is the primary supported mode
- Redis stream architecture remains a planned/extendable path, not the default runtime
- Fairness integration is operational and logged; claims are scoped to implemented evaluation settings
- Text path uses TF-IDF in this version

## Docs

- docs/design_decisions.md — architecture and design lock points
- docs/experiments_log.md — full chronological experiment log
- docs/experiments_summary.md — thesis-facing milestone summaries
- docs/implementation-plan.md — implementation roadmap and phase tracking
