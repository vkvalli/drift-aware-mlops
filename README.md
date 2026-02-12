# Drift-Aware MLOps (Offline MVP)

## Quick Start (Offline Replay + Monitor + Policy + Trainer)

Run the offline pipeline to generate stage logs and a canonical trace:

```bash
python3 -m src.runner.offline_run \
  --csv tests/data/sample.csv \
  --window-size 2 \
  --max-rows 5000 \
  --label-delay-batches 2 \
  --ref-window 2 \
  --threshold 1.0 \
  --seed 1 \
  --replay-output artifacts/replay.jsonl \
  --state-output artifacts/state.jsonl \
  --actions-output artifacts/actions.jsonl \
  --trainer-output artifacts/trainer_output.jsonl \
  --canonical-output artifacts/events.jsonl
```

This produces stage outputs:
- `artifacts/replay.jsonl` (BatchEvent + LabelEvent)
- `artifacts/state.jsonl` (StateEvent)
- `artifacts/actions.jsonl` (ActionEvent)
- `artifacts/trainer_output.jsonl` (CostEvent + ModelUpdateEvent)

And a canonical trace:
- `artifacts/events.jsonl` (RunMeta header + all events in order)

Subset mode (`--max-rows`) enables fast deterministic runs on large datasets.

## Replay Safety / Idempotency

Use these flags to ensure deterministic replays:

```bash
python3 -m src.runner.offline_run \
  --csv tests/data/sample.csv \
  --seed 1 \
  --max-rows 5000 \
  --clean-artifacts \
  --processed-index-file artifacts/processed_actions.jsonl
```

- `--clean-artifacts` clears prior outputs before run.
- `--processed-index-file` stores processed action IDs for idempotent trainer re-runs.

## Aggregate Run Reports

After generating per-run reports, rebuild `runs/summary.txt`:

```bash
python3 tools/aggregate_runs.py --runs-dir runs --out runs/summary.txt
```

To sync the run-parameter table in `docs/experiments_log.md`:

```bash
python3 tools/update_experiments_log.py --summary runs/summary.txt --log docs/experiments_log.md
```

## Baselines + Matrix Runs

Baselines supported by the policy agent:
- `no_retrain`
- `fixed_interval` (every N windows)
- `threshold_persistence` (drift threshold + K consecutive windows)

Run a small matrix (about 6–10 runs):

```bash
python3 tools/run_matrix.py \
  --csv data/electricity.csv \
  --window-size 50 \
  --max-rows 5000 \
  --seeds 111,222 \
  --thresholds 0.01,0.1 \
  --ks 1,3 \
  --fixed-intervals 10 \
  --cooldowns 0 \
  --policies no_retrain,fixed_interval,threshold_persistence
```

Aggregate to CSV:

```bash
python3 tools/aggregate_matrix.py \
  --matrix-dir runs/matrix_YYYYMMDD_HHMMSS \
  --out runs/summary.csv
```

Repro tips:
- Seeds are deterministic and recorded in `run_meta`.
- `events.jsonl` includes the run parameters for each run.

## Registry Logger + Deterministic Replay (FileBus)

The registry logger consolidates stage outputs into the canonical log. In Phase A it uses the FileBus backend (JSONL files) so it can run without Redis.

### Registry Logger

```bash
python3 -m services.registry_logger.main \
  --source artifacts/trainer_output.jsonl \
  --canonical artifacts/events.jsonl \
  --checkpoint artifacts/checkpoints/registry_logger.json \
  --dlq artifacts/dlq.jsonl
```

Notes:
- The canonical log is append-only and never rewritten.
- The checkpoint includes `source_file` and `source_size_bytes` to avoid resuming against a different input file.
- If the canonical log is large (>200MB), the logger warns and you can provide `--seen-index artifacts/index/seen_ids.txt` to avoid a full scan.
- Invalid events are routed to the DLQ and the logger continues.

### Deterministic Replay

Replay uses canonical synthetic IDs (e.g., `state:s_000123`) and replays in file order (byte offset order).

```bash
python3 -m tools.replay \
  --canonical artifacts/events.jsonl \
  --out artifacts/replay_out.jsonl \
  --from-id state:s_000010 \
  --to-id action:a_000120 \
  --checkpoint artifacts/checkpoints/registry_logger.json \
  --deterministic
```
