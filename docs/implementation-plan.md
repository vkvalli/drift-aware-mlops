# Implementation Plan

> Status note (2026-03-18): this file is maintained as a historical execution record.  
> Some early bootstrap tasks (compose/service skeleton paths) have since evolved.  
> For the current reproducible workflow and active operating points, use `README.md`, `docs/design_decisions.md`, and `docs/experiments_summary.md`.
> Repro note: historical command snippets may use `python3`; current standard is `.venv/bin/python`.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Establish the repo skeleton, core compose services (Redis + registry-logger), minimal README with architecture + schemas, and a tiny Electricity replay sample for step-1 readiness.

**Architecture:** Compose-first, event-driven services using Redis Streams; registry-logger persists an append-only JSONL log with Redis stream IDs. Schema-first development with Pydantic/dataclasses; strict validation before emit; DLQ for schema mismatches.

**Tech Stack:** Python 3.11, Redis Streams, Docker Compose, Pydantic, pytest, ruff, black, GitHub Actions, pre-commit.

**Changelog:** 2026-02-06 — Added CI/hooks, schema-first enforcement, structured logging, deterministic seeds, healthchecks, debug endpoints, and --max-rows subset mode.
**Changelog:** 2026-03-06 — Added multi-modal extension phase (regression + text + fairness eval), thesis-ready chunk changelog guardrail, and thesis-review-build freeze workflow.
**Changelog:** 2026-03-07 — Clarified implementation scope: AG News text path is TF-IDF-only (no SBERT yet), and Adult fairness diagnostics are binary-classification scoped.

---

### Task 0: CI / Git Hooks (Early)

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.pre-commit-config.yaml`

**Step 1: Write the failing test**

```python
# tests/test_ci_hooks.py
from pathlib import Path


def test_ci_and_precommit_exist():
    assert Path(".github/workflows/ci.yml").exists()
    assert Path(".pre-commit-config.yaml").exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ci_hooks.py -v`
Expected: FAIL (files missing)

**Step 3: Write minimal implementation**

- Add a GitHub Action that runs `pytest` on pushes to `main` and on PRs.
- Add pre-commit with `black` and `ruff` hooks.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ci_hooks.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add .github/workflows/ci.yml .pre-commit-config.yaml tests/test_ci_hooks.py
git commit -m "chore: add CI and pre-commit hooks"
```

---

### Task 1: Repo Skeleton + Base Folders

**Files:**
- Create: `services/registry_logger/__init__.py`
- Create: `services/registry_logger/main.py`
- Create: `services/ingestion/__init__.py`
- Create: `services/monitor/__init__.py`
- Create: `services/policy/__init__.py`
- Create: `services/trainer/__init__.py`
- Create: `schemas/__init__.py`
- Create: `data/replay/README.md`
- Modify: `.gitignore`

**Step 1: Write the failing test**

```python
# tests/test_repo_skeleton.py
from pathlib import Path

def test_repo_skeleton_exists():
    expected = [
        Path("services/registry_logger"),
        Path("services/ingestion"),
        Path("services/monitor"),
        Path("services/policy"),
        Path("services/trainer"),
        Path("schemas"),
        Path("data/replay"),
    ]
    for p in expected:
        assert p.exists(), f"Missing {p}"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_repo_skeleton.py -v`
Expected: FAIL with missing paths

**Step 3: Write minimal implementation**

- Create the directories and `__init__.py` files.
- Add `.gitignore` entries for `data/raw`, `data/replay/*.parquet`, `mlruns`, `logs`.
- Add `data/replay/README.md` explaining sample replay files and size limit.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_repo_skeleton.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add services schemas data .gitignore tests/test_repo_skeleton.py
git commit -m "chore: add repo skeleton"
```

---

### Task 2: Docker Compose With Redis + Registry-Logger

**Files:**
- Modify: `docker-compose.yml`
- Create: `services/registry_logger/Dockerfile`
- Create: `services/registry_logger/requirements.txt`

**Step 1: Write the failing test**

```python
# tests/test_docker_compose.py
from pathlib import Path
import yaml

def test_compose_has_redis_and_logger():
    data = yaml.safe_load(Path("docker-compose.yml").read_text())
    services = data.get("services", {})
    assert "redis" in services
    assert "registry-logger" in services
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_docker_compose.py -v`
Expected: FAIL (missing services)

**Step 3: Write minimal implementation**

- Add `redis` service (ports + volume) and `registry-logger` service (build from `services/registry_logger`).
- Keep Prometheus/Grafana intact.
- `registry-logger` just starts a Python process that sleeps (stub) so container stays alive.
- Add a minimal `requirements.txt` for registry-logger (redis client + pydantic).
- Add healthcheck entries for Redis and registry-logger to avoid race start issues.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_docker_compose.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add docker-compose.yml services/registry_logger tests/test_docker_compose.py
git commit -m "feat: add redis and registry-logger compose services"
```

---

### Task 3: Minimal README With Architecture + Event Schemas

**Files:**
- Create: `README.md`
- Source: `docs/design_decisions.md`

**Step 1: Write the failing test**

```python
# tests/test_readme.py
from pathlib import Path

def test_readme_exists_and_has_sections():
    text = Path("README.md").read_text()
    assert "Architecture Overview" in text
    assert "Event Schemas" in text
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_readme.py -v`
Expected: FAIL (README missing)

**Step 3: Write minimal implementation**

- Create `README.md` with two sections:
  - **Architecture Overview**: short summary copied from `docs/design_decisions.md`.
  - **Event Schemas**: copy the event schema section (batch, state, action, model_update, cost, label, dlq).
-  - Add `label_delay_batches` to `BatchEvent` in the schema text (one-line note).
- Keep it concise and link to `docs/design_decisions.md` for full detail.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_readme.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add README.md tests/test_readme.py
git commit -m "docs: add minimal README with architecture and schemas"
```

---

### Task 4: Pin Dependency Versions Per Service

**Files:**
- Create: `services/ingestion/requirements.txt`
- Create: `services/monitor/requirements.txt`
- Create: `services/policy/requirements.txt`
- Create: `services/trainer/requirements.txt`

**Step 1: Write the failing test**

```python
# tests/test_service_requirements.py
from pathlib import Path

```

---

### Phase X: Multi-Modal Extension + Thesis Guardrails

#### Guardrail 1 — Thesis-Ready Chunk Changelog

**Files:**
- Create: `tools/log_chunk_update.py`
- Modify: `docs/research_log.md`
- Modify: `README.md`

**Verify:**
- `python3 tools/log_chunk_update.py --help`
- Expected: required fields shown (`--chunk-id`, `--files-changed`, `--why-changed`, `--verify-commands`, `--expected-output`)

**Commit template:**
- `chore: add thesis-ready chunk changelog tooling`

#### Guardrail 2 — Freeze Review Build Branch

**Files:**
- Modify: `README.md`
- Modify: `docs/implementation-plan.md`

**Commands:**
- `git checkout -b thesis-review-build`
- `git push -u origin thesis-review-build`

**Verify:**
- `git branch --list thesis-review-build`
- `git branch -r | rg thesis-review-build`
- Expected: branch present locally and remotely

**Commit template:**
- `chore: add thesis review build branch workflow docs`

#### Regression Support (California Housing)

**Files:**
- Modify: `src/trainer/compat.py`
- Modify: `src/trainer/trainer.py`
- Modify: `src/schemas/cost.py`
- Modify: `src/runner/offline_run.py`
- Modify: `tools/run_matrix.py`
- Modify: `tools/build_rl_dataset.py`
- Modify: `tools/aggregate_matrix.py`
- Add: `tools/prepare_california_housing.py`
- Add tests: `tests/test_regression_trainer.py`, `tests/test_rl_dataset_regression.py`

**Interfaces:**
- `--task {classification,regression,text_classification}`
- `--target-column`
- `--perf-direction {higher_better,lower_better}`
- `--perf-field` supports `mae_pre`, `rmse_pre`, `r2_pre`

**Verify:**
- `python3 -m src.runner.offline_run ... --task regression --dataset-id california_housing ...`
- `python3 tools/build_rl_dataset.py --events ... --perf-field mae_pre --perf-direction lower_better ...`
- Expected: `events.jsonl` contains `mae_pre`; RL dataset has non-null `delta_perf` and `reward`

**Commit template:**
- `feat: add regression task support with MAE/RMSE reward direction`

#### Text Support (AG News)

**Files:**
- Add: `tools/prepare_ag_news.py`
- Modify: `src/trainer/compat.py`
- Modify: `src/trainer/trainer.py`
- Modify: `src/monitor/monitor.py`
- Modify: `src/runner/offline_run.py`
- Add tests: `tests/test_text_trainer.py`, `tests/test_text_monitor_signals.py`

**Interfaces:**
- `--task text_classification`
- `--feature-mode tfidf` (default for text path)
- SBERT embedding mode is explicitly out of scope for this phase

**Added state summaries:**
- `doc_len_mean`
- `vocab_entropy_proxy`

**Verify:**
- `python3 -m src.runner.offline_run ... --task text_classification --dataset-id ag_news ...`
- `python3 tools/build_rl_dataset.py --events ... --perf-field f1_pre ...`
- Expected: complete event chain and valid RL dataset

**Commit template:**
- `feat: add AG News text-classification path with TF-IDF + incremental updates`

#### Fairness Evaluation Support (Adult, evaluation-only)

**Files:**
- Add: `tools/prepare_adult.py`
- Modify: `src/schemas/cost.py`
- Modify: `src/trainer/trainer.py`
- Modify: `tools/aggregate_matrix.py`
- Add: `tools/plot_fairness_tradeoff.py`
- Add tests: `tests/test_fairness_metrics.py`, `tests/test_fairness_aggregation.py`

**Interfaces:**
- `--protected-attr sex` (optional)

**Added metrics:**
- `dpd_pre`
- `eod_pre`

**Scope note:**
- Fairness diagnostics in this phase are scoped to binary classification runs (Adult demo), not multiclass/regression fairness.

**Verify:**
- `python3 -m src.runner.offline_run ... --dataset-id adult --protected-attr sex ...`
- `python3 tools/aggregate_matrix.py ...`
- `python3 tools/plot_fairness_tradeoff.py ...`
- Expected: fairness fields in `cost` events and summary output

**Commit template:**
- `feat: add fairness evaluation metrics (DPD/EOD) and reporting for Adult`


def test_service_requirements_pinned():
    files = [
        Path("services/ingestion/requirements.txt"),
        Path("services/monitor/requirements.txt"),
        Path("services/policy/requirements.txt"),
        Path("services/trainer/requirements.txt"),
    ]
    for f in files:
        text = f.read_text()
        assert "==" in text, f"Unpinned deps in {f}"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_service_requirements.py -v`
Expected: FAIL (files missing)

**Step 3: Write minimal implementation**

- Add per-service `requirements.txt` with exact versions (e.g., `redis==4.x`, `pydantic==1.10.x`).
- Keep lists minimal per service.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_service_requirements.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add services/*/requirements.txt tests/test_service_requirements.py
git commit -m "chore: pin service dependencies"
```

---

### Task 5: Electricity Tiny Replay Sample (100–500 Batches)

**Files:**
- Create: `scripts/make_replay_sample.py`
- Create: `data/replay/electricity_demo.csv`
- Create: `data/replay/electricity_demo.meta.json`
- Test: `tests/test_replay_sample.py`

**Step 1: Write the failing test**

```python
# tests/test_replay_sample.py
from pathlib import Path
import json
import pandas as pd


def test_replay_sample_exists_and_is_small():
    csv_path = Path("data/replay/electricity_demo.csv")
    meta_path = Path("data/replay/electricity_demo.meta.json")
    assert csv_path.exists()
    assert meta_path.exists()

    df = pd.read_csv(csv_path)
    assert 100 <= len(df) <= 500
    assert "batch_id" in df.columns
    assert "label_available" in df.columns

    meta = json.loads(meta_path.read_text())
    assert meta["dataset"] == "electricity"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_replay_sample.py -v`
Expected: FAIL (files missing)

**Step 3: Write minimal implementation**

- Add `scripts/make_replay_sample.py` that reads a local source CSV (placeholder path configurable), slices 100–500 rows into batches, and writes:
  - `data/replay/electricity_demo.csv`
  - `data/replay/electricity_demo.meta.json`
- If the full dataset is not available locally, generate a deterministic synthetic sample with the same column names and a `label_available` flag for delayed labels.
- The script must accept `--seed` and log it in metadata for determinism.
- Emit `label_delay_batches=D` for every batch in the replay output.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_replay_sample.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/make_replay_sample.py data/replay/electricity_demo.* tests/test_replay_sample.py
git commit -m "feat: add tiny electricity replay sample"
```

---

### Task 6: Schema-First Enforcement + Structured Logging

**Files:**
- Create: `schemas/events.py`
- Create: `schemas/logging.py`
- Create: `tests/test_schema_validation.py`

**Step 1: Write the failing test**

```python
# tests/test_schema_validation.py
from schemas.events import BatchEvent


def test_batch_event_validation():
    payload = {
        "schema_version": "1.0",
        "event_id": "evt-1",
        "batch_id": "batch-1",
        "replay_id": "demo1",
        "label_available": False,
        "timestamp": "2026-02-06T00:00:00Z",
        "data": {"rows": 10},
    }
    BatchEvent.model_validate(payload)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_schema_validation.py -v`
Expected: FAIL (schemas missing)

**Step 3: Write minimal implementation**

- Implement Pydantic models for events (batch/state/action/model_update/cost/label/dlq) with schema_version.
- Add structured logging helper to emit JSON logs with fields: `timestamp`, `service`, `level`, `event_id`.
- Add a helper `validate_or_dlq()` that validates payloads before emit and routes invalid payloads to DLQ.
- Ensure registry-logger persists JSONL records with Redis stream IDs for traceability.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_schema_validation.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add schemas tests/test_schema_validation.py
git commit -m "feat: add event schemas and structured logging"
```

---

### Task 7: Minimal Debug Endpoints (Monitor + Policy)

**Files:**
- Create: `services/monitor/debug_server.py`
- Create: `services/policy/debug_server.py`
- Test: `tests/test_debug_endpoints.py`

**Step 1: Write the failing test**

```python
# tests/test_debug_endpoints.py
from services.monitor.debug_server import build_app as build_monitor_app
from services.policy.debug_server import build_app as build_policy_app


def test_debug_app_builds():
    assert build_monitor_app() is not None
    assert build_policy_app() is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_debug_endpoints.py -v`
Expected: FAIL (files missing)

**Step 3: Write minimal implementation**

- Add a tiny FastAPI or Flask app with `/health` or `/debug` endpoint.
- Endpoint returns current consumer offsets and last-event IDs (stubbed until services exist).

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_debug_endpoints.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add services/monitor/debug_server.py services/policy/debug_server.py tests/test_debug_endpoints.py
git commit -m "feat: add debug endpoints for monitor and policy"
```

---

### Task 8: Smoke Check For Step-1 Readiness

**Files:**
- Modify: `docs/implementation-plan.md`

**Step 1: Write the failing test**

```python
# tests/test_step1_readiness.py
from pathlib import Path

def test_step1_readiness():
    assert Path("docker-compose.yml").exists()
    assert Path("README.md").exists()
    assert Path("data/replay/electricity_demo.csv").exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_step1_readiness.py -v`
Expected: FAIL (until steps complete)

**Step 3: Write minimal implementation**

- Ensure all prior tasks are complete.
- Add a short completion note to `docs/implementation-plan.md` once done.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_step1_readiness.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add docs/implementation-plan.md tests/test_step1_readiness.py
git commit -m "test: add step-1 readiness smoke check"
```

---

**Plan complete and saved to `docs/implementation-plan.md`. Two execution options:**

1. Subagent-Driven (this session) - I dispatch a fresh subagent per task, review between tasks, fast iteration
2. Parallel Session (separate) - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
