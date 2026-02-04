current benchmark datasets:
Tabular: Electricity, Airline Delay, Google Cluster Trace
Text: IMDB, AG News
Image: Rotated-MNIST, CIFAR-10C (optional later)
Fairness add-on: Adult

think about adding:
1. uncertainty modeling (confidence intervals, quantiles)
2. explicit budget constraints
3. action costs broken down (compute vs latency vs risk)
4. delayed feedback modeling
5. deployment friction (cool-down periods, rollback risk)
6. multi-model portfolio handling
7. partial retraining variants (layer freezing, LoRA, etc.)

Architecture (MVP, Compose-first)

We’ll implement a minimal event-driven MLOps pipeline using Docker Compose and Redis Streams to coordinate five services: ingestion/replay, monitor, policy-agent, trainer/updater, and registry+logger. Each service owns a single responsibility and communicates only via immutable events, enforcing a clean separation between signals, decisions, and outcomes—critical for delayed labels and multi-objective retraining decisions. The Ingestion service reads time-ordered dataset slices and emits batch events (including timestamps, window IDs, and label-availability flags). Monitor consumes batch events, computes drift/performance signals and fairness-related signals when labels are available (or proxy group-shift signals when labels are delayed), and emits state events containing the policy state vector. Policy-Agent consumes state events and emits action events (e.g., defer, fine_tune, partial_retrain, full_retrain), subject to system constraints such as cooldown windows. Trainer consumes action events, runs the requested update routine, and emits model_update and cost events (training time, estimated $ cost, energy/CO₂, and latency impact). Registry+Logger subscribes to all event streams and writes an append-only log for reproducibility, deterministic replay, and post-hoc reward computation. Deterministic replay is supported via Redis Streams IDs, checkpointed consumer offsets, and explicit model_version fields carried through events to preserve causal attribution.

Data Flow (Event Schemas + Stream Topology)

Core principle: immutable, append-only events with explicit schema validation, versioned model lineage, and deterministic replay. Cost events are logged for post-hoc evaluation and delayed reward computation, not for immediate policy state construction. Schema evolution is handled via versioned readers to preserve backward compatibility during replay.

Streams (Redis Streams)
- stream:batch (from ingestion)
- stream:state (from monitor)
- stream:action (from policy-agent)
- stream:model_update (from trainer)
- stream:cost (from trainer)
- stream:label (optional / future, for delayed labels)
- stream:dlq (internal, invalid schema or poison messages)

Stream Topology
| Stream | Producer | Consumers | Consumer Group |
|---|---|---|---|
| stream:batch | Ingestion | Monitor | monitor-cg |
| stream:state | Monitor | Policy-Agent | policy-cg |
| stream:action | Policy-Agent | Trainer | trainer-cg |
| stream:model_update | Trainer | Serving, Logger | serving-cg, logger-cg |
| stream:cost | Trainer | Logger | logger-cg |
| stream:label | Ingestion | Monitor (delayed processing) | monitor-label-cg |

Event Schemas (minimal fields)

batch
- batch_id, window_id, timestamp_start, timestamp_end
- label_available (bool), dataset_id
- model_version (current serving version)
- schema_version

state
- state_id, batch_id, window_id
- drift_metrics (KS/PSI/MMD)
- perf_metrics (ground-truth when labels available; proxy metrics otherwise)
- fairness_metrics when label_available=true else fairness_proxy
- label_delay (explicit delay metadata)
- budget_status (explicit budget and cooldown metadata)
- model_version, schema_version

budget_status example
```python
{
  "remaining_usd": 157.50,
  "spent_this_window_usd": 12.30,
  "cooldown_active": false,
  "cooldown_expires_at": null
}
```

perf_metrics example
```python
# When label_available=true
{
  "mode": "ground_truth",
  "accuracy": 0.89,
  "f1": 0.87,
  "calibration_error": 0.04
}

# When label_available=false
{
  "mode": "proxy",
  "predicted_confidence_mean": 0.82,
  "predicted_entropy": 0.45,
  "model_uncertainty": 0.12
}
```

fairness_metrics / fairness_proxy example
```python
# When label_available=true
{
  "mode": "ground_truth",
  "dpd": 0.08,
  "eod": 0.12,
  "subgroup_accuracy": {
    "group_A": 0.92,
    "group_B": 0.85
  }
}

# When label_available=false
{
  "mode": "proxy",
  "group_covariate_shift": {
    "group_A": 0.03,
    "group_B": 0.15
  },
  "predicted_label_dist": {
    "group_A": {"pos": 0.6, "neg": 0.4},
    "group_B": {"pos": 0.4, "neg": 0.6}
  }
}
```

label_delay example
```python
{
  "expected_delay_sec": 3600,
  "time_since_batch_sec": 1200,
  "labels_pending": true
}
```

action
- action_id, state_id
- action_type = defer | fine_tune | partial_retrain | full_retrain
- cooldown_expires_at
- model_version, schema_version

model_update
- action_id, new_model_version
- artifact_path, training_data_window
- eval_summary (key metrics)
- schema_version

cost
- action_id, train_time_s, estimated_$
- kwh, co2_kg, latency_impact_ms
- schema_version

label (optional / future)
- label_id, batch_id
- labels (or pointer)
- timestamp_observed
- schema_version

Validation + Idempotency
- Each consumer validates event schema before processing.
- Idempotency keys: batch_id, state_id, action_id.
- Registry+Logger writes a canonical log of all events; replay is driven by stream IDs + consumer offsets.

Observability Hooks
- Monitor and Trainer export Prometheus metrics per event type (processing latency, error rate, action distribution).
- Grafana dashboards show drift metrics, action counts, cost curves, and fairness proxy deltas.

Model Versioning Contract
- Every event carries model_version (input lineage) or new_model_version (output lineage), preserving causal attribution for delayed rewards and post-hoc evaluation.

Error Handling & Resilience (Bus/Schema Guarantees Only)

- Delivery semantics: at-least-once via Redis Streams consumer groups; consumers must be idempotent using action_id/state_id/batch_id keys.
- Poison message handling: invalid schema or non-parseable payloads are routed to a dead-letter stream (stream:dlq) with error metadata for inspection.
- Backpressure control: consumers track lag; if lag exceeds thresholds, policy-agent emits defer actions and trainer pauses new work.
- Schema validation: every consumer validates payloads against schema_versioned readers before processing; invalid versions are rejected to DLQ.
- Replay safety: deterministic replay uses stream IDs + stored consumer offsets; side effects (training, deployment) are disabled unless replay mode is explicitly enabled.

State Vector Contract (Locked Core + Extensible Extras)

The state event uses a fixed core schema (stable fields required for all policies and experiments) plus an extensible extras namespace for optional signals. Core fields ensure deterministic replay, consistent ablations, and stable policy interfaces; optional fields allow incremental addition of new drift/fairness/uncertainty signals without breaking backward compatibility.

Fixed core fields (always present)

Identifiers / timing
- state_id, batch_id, window_id
- decision_timestamp
- model_version

Availability flags
- label_available
- state_mode = ground_truth | proxy

Core signals (compact, numeric-first)
- drift_core (1–3 numbers max; e.g., PSI_mean, PSI_max, or MMD)
- perf_core (if labels: accuracy/F1; else: confidence/entropy)
- fairness_core (if labels: DPD/EOD; else: group_shift_max)

Constraints
- budget_status (remaining, cooldown flags)
- label_delay (pending + time since batch)

Extensible extras (optional, namespaced)
- extras.drift (full per-feature drift, multiple metrics)
- extras.fairness (per-group metrics tables)
- extras.perf (calibration breakdowns, per-class metrics)
- extras.system (queue depth, GPU utilization)
- extras.debug (anything you don’t want the agent to rely on)

Policy-agent rule: it may only use fixed core fields in the primary thesis experiments. Extras are for later or ablations.
