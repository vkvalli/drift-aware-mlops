# Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Establish the repo skeleton, core compose services (Redis + registry-logger), minimal README with architecture + schemas, and a tiny Electricity replay sample for step-1 readiness.

**Architecture:** Compose-first, event-driven services using Redis Streams; registry-logger persists an append-only JSONL log with Redis stream IDs. Schema-first development with Pydantic/dataclasses; strict validation before emit; DLQ for schema mismatches.

**Tech Stack:** Python 3.11, Redis Streams, Docker Compose, Pydantic, pytest.

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

### Task 4: Electricity Tiny Replay Sample (100–500 Batches)

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

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_replay_sample.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/make_replay_sample.py data/replay/electricity_demo.* tests/test_replay_sample.py
git commit -m "feat: add tiny electricity replay sample"
```

---

### Task 5: Smoke Check For Step-1 Readiness

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
