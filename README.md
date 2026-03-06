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

## Task Modes

The offline runner supports multiple task types through shared event contracts:

- `--task classification` (default tabular classification)
- `--task regression` (logs `mae_pre`, `rmse_pre`, `r2_pre`)
- `--task text_classification` with `--feature-mode tfidf`

Useful task-related flags:
- `--target-column` for explicit target field
- `--feature-mode tabular|tfidf`
- `--protected-attr` for fairness diagnostics (`dpd_pre`, `eod_pre`)

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

## Thesis-Ready Chunk Changelog

Use this after each implementation chunk to append a structured entry to `docs/research_log.md`:

```bash
python3 tools/log_chunk_update.py \
  --chunk-id "X.Y" \
  --files-changed "src/trainer/trainer.py,tools/build_rl_dataset.py" \
  --why-changed "Added regression metrics support,Enabled lower-is-better reward direction" \
  --verify-commands "python3 -m unittest tests/test_rl_dataset_regression.py,python3 tools/build_rl_dataset.py --help" \
  --expected-output "Regression test passes,CLI shows mae_pre/rmse_pre perf fields" \
  --new-metrics "mae_pre,rmse_pre,r2_pre" \
  --new-plots "" \
  --new-log-fields "task,target_column,perf_direction" \
  --limitations "Smoke-only verification in local run" \
  --next-chunk "Text classification support"
```

## Review Build Branch Workflow

Stable review snapshots are tracked on `thesis-review-build`.

- `main` can keep moving with active development.
- `thesis-review-build` should receive only review-safe fixes and documentation polish.

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
