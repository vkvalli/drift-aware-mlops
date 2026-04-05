# Drift-Aware MLOps

Offline, event-driven retraining framework with deterministic replay, baseline policies, and RL policy support.

Current state (v2.0.1):
- End-to-end offline chain is working: `replay -> monitor -> policy -> trainer -> logger`
- Task modes: `classification`, `regression`, `text_classification`
- Datasets validated in smoke/matrix runs:
  - Electricity (classification)
  - Airline Delay (classification + fairness eval fields)
  - California Housing (regression)
  - AG News (text classification)
  - Adult from OpenML (fairness evaluation benchmark)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
.venv/bin/python -m pip install -r requirements-dev.txt
```

Run tests:

```bash
.venv/bin/python -m unittest
```

## Core CLI

Main runner:

```bash
.venv/bin/python -m src.runner.offline_run --help
```

Matrix runner:

```bash
.venv/bin/python -m tools.run_matrix --help
```

## Quick Start: Single Run

Always write run outputs to a unique `runs/<name>` folder.

### 1) Classification (Electricity)

```bash
RUN_ID="elec_smoke_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="runs/$RUN_ID"
mkdir -p "$RUN_DIR"

.venv/bin/python -m src.runner.offline_run \
  --csv data/electricity-normalized.csv \
  --task classification \
  --label-column class \
  --target-column class \
  --dataset-id electricity \
  --window-size 200 \
  --max-rows 40000 \
  --label-delay-batches 2 \
  --seed 111 \
  --replay-output "$RUN_DIR/replay.jsonl" \
  --state-output "$RUN_DIR/state.jsonl" \
  --actions-output "$RUN_DIR/actions.jsonl" \
  --trainer-output "$RUN_DIR/trainer_output.jsonl" \
  --canonical-output "$RUN_DIR/events.jsonl" \
  --processed-index-file "$RUN_DIR/processed_actions.jsonl" \
  --dlq "$RUN_DIR/dlq.jsonl"
```

### 2) Regression (California Housing)

```bash
.venv/bin/python tools/prepare_california_housing.py --out data/california_housing.csv

RUN_ID="california_smoke_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="runs/$RUN_ID"
mkdir -p "$RUN_DIR"

.venv/bin/python -m src.runner.offline_run \
  --csv data/california_housing.csv \
  --task regression \
  --label-column MedHouseVal \
  --target-column MedHouseVal \
  --perf-direction lower_better \
  --dataset-id california_housing \
  --window-size 200 \
  --max-rows 40000 \
  --label-delay-batches 2 \
  --seed 111 \
  --replay-output "$RUN_DIR/replay.jsonl" \
  --state-output "$RUN_DIR/state.jsonl" \
  --actions-output "$RUN_DIR/actions.jsonl" \
  --trainer-output "$RUN_DIR/trainer_output.jsonl" \
  --canonical-output "$RUN_DIR/events.jsonl" \
  --processed-index-file "$RUN_DIR/processed_actions.jsonl" \
  --dlq "$RUN_DIR/dlq.jsonl"
```

### 3) Text Classification (AG News)

```bash
.venv/bin/python tools/prepare_ag_news.py --out data/ag_news.csv --rows 60000

RUN_ID="agnews_smoke_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="runs/$RUN_ID"
mkdir -p "$RUN_DIR"

.venv/bin/python -m src.runner.offline_run \
  --csv data/ag_news.csv \
  --task text_classification \
  --feature-mode tfidf \
  --label-column class \
  --dataset-id ag_news \
  --window-size 200 \
  --max-rows 40000 \
  --label-delay-batches 2 \
  --seed 111 \
  --replay-output "$RUN_DIR/replay.jsonl" \
  --state-output "$RUN_DIR/state.jsonl" \
  --actions-output "$RUN_DIR/actions.jsonl" \
  --trainer-output "$RUN_DIR/trainer_output.jsonl" \
  --canonical-output "$RUN_DIR/events.jsonl" \
  --processed-index-file "$RUN_DIR/processed_actions.jsonl" \
  --dlq "$RUN_DIR/dlq.jsonl"
```

### 4) Fairness Eval Benchmark (Adult from OpenML)

```bash
.venv/bin/python tools/prepare_adult.py --out data/adult.csv --target-column income --rows 60000

RUN_ID="adult_openml_smoke_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="runs/$RUN_ID"
mkdir -p "$RUN_DIR"

.venv/bin/python -m src.runner.offline_run \
  --csv data/adult.csv \
  --task classification \
  --label-column income \
  --target-column income \
  --protected-attr sex \
  --dataset-id adult_openml \
  --window-size 200 \
  --max-rows 40000 \
  --label-delay-batches 2 \
  --seed 111 \
  --replay-output "$RUN_DIR/replay.jsonl" \
  --state-output "$RUN_DIR/state.jsonl" \
  --actions-output "$RUN_DIR/actions.jsonl" \
  --trainer-output "$RUN_DIR/trainer_output.jsonl" \
  --canonical-output "$RUN_DIR/events.jsonl" \
  --processed-index-file "$RUN_DIR/processed_actions.jsonl" \
  --dlq "$RUN_DIR/dlq.jsonl"
```

## Plots + Run Summary

```bash
.venv/bin/python tools/plot_run.py --events "$RUN_DIR/events.jsonl" --outdir "$RUN_DIR/plots"
.venv/bin/python tools/summarize_run.py --events "$RUN_DIR/events.jsonl" --out "$RUN_DIR/run_summary.txt"
```

## Matrix Runs (Baselines)

```bash
.venv/bin/python -m tools.run_matrix \
  --csv data/electricity-normalized.csv \
  --task classification \
  --label-column class \
  --target-column class \
  --window-size 200 \
  --max-rows 60000 \
  --label-delay-batches 2 \
  --ref-window 3 \
  --seeds 111,222,333 \
  --policies no_retrain,fixed_interval,threshold_persistence
```

Aggregate:

```bash
LATEST_MATRIX="$(ls -td runs/matrix_* | head -n 1)"
.venv/bin/python tools/aggregate_matrix.py --matrix-dir "$LATEST_MATRIX" --out "$LATEST_MATRIX/summary.csv"
```

## Registry Logger + Deterministic Replay

Registry logger:

```bash
.venv/bin/python -m services.registry_logger.main \
  --input runs/<run>/trainer_output.jsonl \
  --canonical runs/<run>/events.jsonl \
  --checkpoint runs/<run>/checkpoints/registry_logger.json \
  --dlq runs/<run>/dlq.jsonl
```

Replay tool:

```bash
.venv/bin/python -m tools.replay \
  --source runs/<run>/events.jsonl \
  --out runs/<run>/replay_out.jsonl \
  --from-id state:s_000010 \
  --to-id action:a_000120 \
  --deterministic
```

## Thesis Logging Guardrail

Append chunk-level implementation notes to `docs/research_log.md`:

```bash
.venv/bin/python tools/log_chunk_update.py --help
```

## Where Results Live

- Rolling experiment details: `docs/experiments_log.md`
- Final condensed tables: `docs/experiments_summary.md`
- Architecture and metric rationale: `docs/design_decisions.md`

## Branch Policy

- `main`: active development
- `thesis-review-build`: review-safe snapshot branch
