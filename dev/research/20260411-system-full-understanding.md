---
date: 2026-04-11T18:37:39-07:00
topic: "Full system understanding of drift-aware-mlops (folder + local repo + remote repo)"
tags: [research, codebase, mlops, drift-monitoring, fairness, rl, experiments]
status: complete
last_updated: 2026-04-11
git_commit: adbf328
---

# Research: Full system understanding of drift-aware-mlops

**Date**: 2026-04-11T18:37:39-07:00

## Research Question

Gain comprehensive understanding of the entire workspace, local git repository state, remote repository tracking state, and how the implemented system works end-to-end.

## Summary

The repository is an offline-first, deterministic MLOps experimentation system for drift-aware retraining policy evaluation. The active architecture is a schema-validated JSONL pipeline (`replay -> monitor -> policy_agent -> trainer -> canonical events`). It includes deterministic IDs, DLQ-backed validation, policy baselines plus LinUCB, fairness-aware state/reward support, and experiment orchestration/reporting scripts. Local and remote branches are synced and point to `https://github.com/vkvalli/drift-aware-mlops`.

## Detailed Findings

### 1) Repository and Git Topology (Local + Remote)

- Current branch: `main` tracking `origin/main`; workspace has untracked `node_modules/` (`git status --short --branch` at 2026-04-11).
- Remote configured: `origin https://github.com/vkvalli/drift-aware-mlops` (`git remote -v`).
- Local branches with upstreams: `main`, `fairness`, `thesis-review-build`, `codex/implementation-plan`, `codex/main-isolated` (`git branch -vv`).
- Remote branches discovered: `origin/main`, `origin/fairness`, `origin/thesis-review-build`, `origin/codex/implementation-plan`, `origin/codex/main-isolated` (`git for-each-ref refs/remotes/origin`).
- Upstream tracking has no ahead/behind markers across local heads (`git for-each-ref --format='%(refname:short) %(upstream:short) %(upstream:track)' refs/heads`).
- Tags present: `v2.0.1`, `v2.0.0`, `v1.1.0`, `v1.0.0`, `setup-complete` (`git tag --sort=-creatordate`).
- Commit counts: `53` commits on `main`, `57` across all refs (`git rev-list --count`).
- HEAD at research time: `adbf328` (`git rev-parse --short HEAD`).

### 2) Runtime Architecture (Implemented Path)

- Orchestrator entrypoint wires the full chain and writes run metadata first: [`src/runner/offline_run.py:141`].
- Pipeline sequence in code is explicit:
  - replay stage: [`src/runner/offline_run.py:185`]
  - monitor stage: [`src/runner/offline_run.py:242`]
  - policy stage: [`src/runner/offline_run.py:258`]
  - trainer stage: [`src/runner/offline_run.py:282`]
  - canonical merge append after each stage: [`src/runner/offline_run.py:240`], [`src/runner/offline_run.py:256`], [`src/runner/offline_run.py:280`], [`src/runner/offline_run.py:300`]
- Design doc confirms architecture lock and deployment note: offline JSONL is active for thesis evidence; Redis Streams remains a target topology [`docs/design_decisions.md:556`], [`docs/design_decisions.md:567`].

### 3) Ingestion and Label Delay Semantics

- Replay creates `BatchEvent` windows and delayed `LabelEvent`s with deterministic IDs and configurable lag [`src/ingestion/replay.py:29`], [`src/ingestion/replay.py:55`], [`src/ingestion/replay.py:72`], [`src/ingestion/replay.py:77`].
- Every emitted record is validated via schema helper before persistence [`src/ingestion/replay.py:68`], [`src/ingestion/replay.py:82`].

### 4) Monitoring and State Construction

- Monitor loads batch events, computes drift statistics, and emits `StateEvent`s [`src/monitor/monitor.py:33`], [`src/monitor/monitor.py:221`], [`src/monitor/monitor.py:340`].
- Drift readiness requires reference history and task-specific logic (`tabular` vs `text_classification`) [`src/monitor/monitor.py:284`], [`src/monitor/monitor.py:291`], [`src/monitor/monitor.py:309`].
- Fairness signal is lag-aware and only becomes active with support thresholds (`fairness_min_support`, `fairness_min_group_support`) [`src/monitor/monitor.py:317`], [`src/monitor/monitor.py:325`], [`src/monitor/monitor.py:333`].

### 5) Policy Layer

- Policy engine supports: `no_retrain`, `fixed_interval`, `threshold_persistence` (default branch), `fairness_trigger`, and `linucb` [`src/policy/policy_agent.py:56`], [`src/policy/policy_agent.py:122`], [`src/policy/policy_agent.py:125`], [`src/policy/policy_agent.py:174`], [`src/policy/policy_agent.py:184`], [`src/policy/policy_agent.py:97`].
- Fairness-trigger implementation includes hysteresis, persistence counter, and cooldown behavior [`src/policy/policy_agent.py:126`], [`src/policy/policy_agent.py:134`], [`src/policy/policy_agent.py:146`], [`src/policy/policy_agent.py:161`].
- LinUCB policy path can use fairness features in context vector (`fairness_core`, `fairness_ready`) [`src/policy/policy_agent.py:17`], [`src/policy/policy_agent.py:25`], [`src/policy/policy_agent.py:116`].

### 6) Training, Cost, and Model Updates

- Trainer reads actions + optional states, applies idempotency via processed-action index, and conditionally executes updates [`src/trainer/trainer.py:506`], [`src/trainer/trainer.py:524`], [`src/trainer/trainer.py:561`], [`src/trainer/trainer.py:941`].
- Supports `classification`, `regression`, `text_classification` loading/feature paths (`tabular` or TF-IDF) [`src/trainer/trainer.py:82`], [`src/trainer/trainer.py:134`], [`src/trainer/trainer.py:648`], [`src/trainer/trainer.py:772`].
- Emits `CostEvent` with pre-update metrics and fairness diagnostics when available (`dpd_pre`, `eod_pre`) [`src/trainer/trainer.py:704`], [`src/trainer/trainer.py:722`], [`src/trainer/trainer.py:889`], [`src/trainer/trainer.py:934`].
- Full retrain and fine-tune differ in modeled resource/cost values and training semantics [`src/trainer/trainer.py:694`], [`src/trainer/trainer.py:700`], [`src/trainer/trainer.py:811`].

### 7) Event Contracts, Validation, and Logging

- Event-type-to-model mapping and schema naming are centralized in validation module [`src/schemas/validate.py:15`], [`src/schemas/validate.py:31`].
- Invalid payloads are routed to DLQ with expected schema and offset metadata [`src/schemas/validate.py:52`], [`src/schemas/validate.py:73`].
- Pydantic compatibility layer provides fallback validation/model behavior when unavailable [`src/schemas/compat.py:21`], [`src/schemas/compat.py:24`], [`src/schemas/compat.py:104`].
- JSONL logger standardizes `event_type` and `logged_at` for persisted records [`src/logger/jsonl_logger.py:22`], [`src/logger/jsonl_logger.py:27`].

### 8) Registry Logger Service Path

- Service consumes JSONL via `FileBus`, validates schema, deduplicates by canonical ID, checkpoints byte offset, and appends canonical records [`services/registry_logger/main.py:85`], [`services/registry_logger/main.py:93`], [`services/registry_logger/main.py:100`], [`services/registry_logger/main.py:114`], [`services/registry_logger/main.py:125`], [`services/registry_logger/main.py:134`].
- Validation wrapper delegates to shared schema map and parser [`services/registry_logger/validate.py:7`], [`services/registry_logger/validate.py:11`].

### 9) RL Components

- LinUCB implementation is a 2-action contextual bandit (`defer`, `fine_tune`) with standard UCB score over per-action linear models [`src/rl/linucb.py:6`], [`src/rl/linucb.py:9`], [`src/rl/linucb.py:21`].
- Design decisions document records fairness-aware RL state/reward integration and current lock status [`docs/design_decisions.md:545`], [`docs/design_decisions.md:547`], [`docs/design_decisions.md:550`], [`docs/design_decisions.md:553`].

### 10) Experiment Operations and Reporting

- Single-run wrapper delegates to offline runner in timestamped `runs/run_*` directory [`tools/run_experiment.py:25`], [`tools/run_experiment.py:30`], [`tools/run_experiment.py:77`].
- Matrix runner executes policy/seed/threshold sweeps and writes run folders under `runs/matrix_*` [`tools/run_matrix.py:67`], [`tools/run_matrix.py:79`], [`tools/run_matrix.py:82`], [`tools/run_matrix.py:145`].
- Run summarization computes event counts, update/cost aggregates, and metric summaries from canonical logs [`tools/summarize_run.py:24`], [`tools/summarize_run.py:33`], [`tools/summarize_run.py:88`].

### 11) Testing and Quality Gate Coverage

- CI uses Python 3.11 and runs `python -m unittest` on push/PR to `main` [`/.github/workflows/ci.yml:1`], [`/.github/workflows/ci.yml:17`], [`/.github/workflows/ci.yml:23`].
- End-to-end tests assert event counts/ordering and deterministic replay outcomes [`tests/test_offline_e2e.py:58`], [`tests/test_offline_e2e.py:59`], [`tests/test_offline_e2e.py:92`], [`tests/test_offline_e2e.py:104`].
- Policy tests include fairness trigger and LinUCB fairness-feature behavior [`tests/test_policy_agent.py:162`], [`tests/test_policy_agent.py:239`], [`tests/test_policy_agent.py:297`].
- Monitor and trainer tests cover drift behavior, fairness lag handling, determinism, and idempotency [`tests/test_monitor.py:63`], [`tests/test_monitor.py:96`], [`tests/test_trainer.py:144`], [`tests/test_trainer.py:172`].

### 12) Workspace/Data Shape

- Total tracked file paths discoverable by `rg --files`: `16471`.
- Python LOC in `src + services`: `3024`.
- Data/artifact footprint at research time:
  - `data`: `11G`
  - `runs`: `313M`
  - `artifacts`: `4.4M`
- Large source dataset present: `data/airline/airline_2018_2019.csv` at ~`5.0G`.
- `runs` contains multiple completed experiments; `summary.csv` found in 63 run folders.

## Code References

- `src/runner/offline_run.py:141` - Orchestrator `main`, run metadata and stage chaining.
- `src/ingestion/replay.py:29` - Replay event generation with delayed labels.
- `src/monitor/monitor.py:221` - Monitor entrypoint and drift/fairness state computation.
- `src/policy/policy_agent.py:48` - Policy execution across heuristic/fairness/RL families.
- `src/trainer/trainer.py:506` - Trainer execution, idempotency, metrics, and model updates.
- `src/schemas/validate.py:52` - Validation + DLQ routing.
- `src/schemas/compat.py:104` - Parse/validation compatibility path.
- `src/logger/jsonl_logger.py:22` - JSONL event write normalization.
- `src/bus/file_bus.py:6` - FileBus read/publish abstraction.
- `services/registry_logger/main.py:85` - Registry logger consumer loop with checkpointing and dedup.
- `services/registry_logger/validate.py:11` - Service-level schema validation entrypoint.
- `src/rl/linucb.py:9` - LinUCB policy core.
- `tools/run_matrix.py:67` - Matrix orchestrator.
- `tools/summarize_run.py:24` - Run summarizer.
- `tests/test_offline_e2e.py:58` - E2E ordering and determinism assertions.

## Architecture Insights

- The repository is built around reproducible offline experimentation rather than online serving; deterministic replay and append-only logs are central system invariants.
- Schema-validated event contracts unify module boundaries and support both strict validation and compatibility fallback.
- Policy experimentation is intentionally modular: deterministic baselines, fairness-triggered control, and contextual bandit paths are swappable over the same state/action/cost contracts.
- The same canonical event stream powers training traces, RL dataset generation, and run-level summarization.

## Historical Context (from dev/log/)

- No `dev/log/` directory exists in this workspace at research time.
- Historical context is instead maintained in docs:
  - `docs/design_decisions.md` (architecture lock + change log)
  - `docs/experiments_log.md` (run-by-run thesis experiment history)
  - `docs/research_log.md` (chunked implementation/research milestones)

## Related Research

- None yet under `dev/research/` before this document.

## Open Questions

- None required to describe current implemented behavior.
