# Thesis Project Update for Dr. Anli Ji

Date: 2026-02-04

## Purpose (Advisor Review)

This document summarizes the project architecture and core design decisions to support an advisor review of:  
**(1) scope appropriateness, (2) thesis reasonableness, and (3) defensible claims.**

---

## 1) Architecture Overview (1 page max)

**MVP: event-driven retraining loop with delayed labels and multi-objective signals**

```mermaid
flowchart LR
  A["Ingestion / Replay"] -->|batch events| B["Monitor"]
  B -->|state events| C["Policy-Agent"]
  C -->|action events| D["Trainer / Updater"]
  D -->|model_update + cost events| E["Registry + Logger"]
  A -->|label events (delayed)| B

  subgraph Bus["Redis Streams"]
    S1["stream:batch"]
    S2["stream:state"]
    S3["stream:action"]
    S4["stream:model_update"]
    S5["stream:cost"]
    S6["stream:label (optional)"]
    S7["stream:dlq"]
  end

  A --> S1
  B --> S2
  C --> S3
  D --> S4
  D --> S5
  A --> S6

  S1 --> B
  S2 --> C
  S3 --> D
  S4 --> E
  S5 --> E
  S6 --> B
```

**Key property:** signals are produced independently of decisions, decisions are logged, and outcomes feed back later. This enables delayed-label learning without leakage.

---

## 2) Design Decisions Summary (2–3 pages max)

### 2.1 Core Architecture

**Services (Compose-first MVP):**
- `ingestion` → emits time-ordered `batch` events with label availability flags.  
- `monitor` → computes drift/perf/fairness signals and emits `state` events.  
- `policy-agent` → consumes `state` and emits `action` events.  
- `trainer` → executes actions and emits `model_update` + `cost` events.  
- `registry-logger` → consumes all streams and writes an append-only log.

**Message Bus:** Redis Streams for low-overhead, deterministic replay.  
**Replay:** stream IDs + checkpointed offsets for reproducibility.  
**DLQ:** invalid events routed to `stream:dlq` with reason codes.

### 2.2 State / Action / Reward

**State Vector Contract (fixed core + extensible extras):**
- Core fields (always present): `state_id`, `batch_id`, `window_id`, `decision_timestamp`, `model_version`, `label_available`, `state_mode`, `drift_core`, `perf_core`, `fairness_core`, `budget_status`, `label_delay`.
- Extras namespace for expanded signals: `extras.drift`, `extras.fairness`, `extras.perf`, `extras.system`, `extras.debug`.
- **Policy rule:** primary experiments use **core fields only**.

**Action Semantics (intent-level):**
- `defer`, `fine_tune`, `partial_retrain`, `full_retrain`.  
- **Execution reporting:** trainer logs `effective_action_type`, `executed_as_requested`, `reason_code` if action is modified due to constraints.  
- `defer` emits a zero-cost event for trace completeness.

**Reward / Objectives (Phase-1 baseline):**
- Objective vector: `O_perf`, `O_fair`, `O_cost`, `O_energy`.  
- Scalarized reward for training:  
  `R = w1 * ΔO_perf - w2 * O_cost - w3 * O_energy - w4 * O_fair_gap - w5 * action_penalty`  
- Terms normalized over sliding window; ground-truth metrics supersede proxies when available.

### 2.3 Evaluation Scope

**Primary:** offline replay (deterministic).  
**Appendix:** synthetic online simulation to stress bursty drift and label delays.  

**Baselines:** fixed-interval, drift-threshold, no-retrain, fairness-threshold, cost-aware retrain.  
**Metrics:** per-decision objective vector + aggregate trends; bootstrap confidence intervals.  
**Ablations:** policy and system ablations to isolate fairness/cost/telemetry effects.

### 2.4 Non-Goals (Explicit)

- No production deployment on Kubernetes in the thesis MVP.  
- No attempt to claim real-world causal impact beyond benchmark-based evaluation.  
- No full observability or testing sections included here (kept concise for advisor review).  
- No long literature survey or implementation code included in this summary.

---

## Advisor Checkpoints (What Feedback I Need)

1. **Is the architecture appropriate and defensible for a thesis?**  
2. **Is the scope reasonable for completion?**  
3. **Are the evaluation claims defensible with offline replay + stress appendix?**  

If anything looks over-scoped or under-scoped, I will adjust immediately.
