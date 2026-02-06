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
| stream:dlq | Any consumer | Logger / Inspector | dlq-cg |

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
- effective_action_type (optional)
- executed_as_requested (optional, bool)
- reason_code (optional)
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

Action Semantics (Locked Contract)

Actions are emitted by the Policy-Agent and consumed by the Trainer. Each action is intent-level (what to do), not an implementation plan (how to do it).

Action Types
- defer: no training is executed; monitoring continues.
- fine_tune: update existing model weights on recent data; no architecture changes.
- partial_retrain: retrain a subset of the model (e.g., last layers, LoRA adapters).
- full_retrain: train from scratch or from a baseline checkpoint on full training window.

Required Fields (in action event)
- action_id, state_id
- action_type
- cooldown_expires_at
- model_version
- schema_version

Execution reporting: Trainer records effective_action_type and executed_as_requested in the emitted cost (or model_update) metadata when the requested action cannot be executed (e.g., insufficient window, cooldown active), along with a reason_code.

Defer accounting: defer emits a cost event to preserve a complete per-decision trace (training costs set to zero in MVP).

Reward / Objectives (Locked Contract)

We optimize retraining decisions using multi-objective signals: accuracy, fairness, operational cost, and energy. The policy is evaluated on a vector of objectives for transparency, with an optional scalarization for training.

Objective Vector (reported for every decision)
- O_perf: model performance (accuracy/F1 when labels available; proxy metrics otherwise)
- O_fair: fairness (DPD/EOD when labels available; proxy group-shift otherwise)
- O_cost: compute cost (estimated_$, train time, latency impact)
- O_energy: energy footprint (kwh, co2_kg)

Primary Reward (Phase-1 baseline)

A weighted scalarization is used for the policy training loop to enable stable learning under delayed feedback, while all objectives are logged and evaluated as a vector for Pareto analysis:

R = w1 * ΔO_perf - w2 * O_cost - w3 * O_energy - w4 * O_fair_gap - w5 * action_penalty

All objective terms are normalized to comparable scales over a sliding window prior to scalarization.

ΔO_perf is computed relative to the most recent deployed model evaluated on the same window (or proxy-equivalent window when labels are delayed).

O_fair_gap is defined such that higher values indicate worse fairness (e.g., absolute DPD/EOD).

action_penalty discourages unnecessary retraining frequency and oscillation between actions.

Delayed Feedback Handling

When labels are delayed, rewards are computed asynchronously using the label stream and appended to the decision trace using state_id and model_version. The policy-agent does not see these rewards at decision time, preventing leakage.

When both proxy and ground-truth metrics exist for the same decision, ground-truth metrics supersede proxies for reward computation.

Reporting & Evaluation

- Report both scalar reward and the full objective vector.
- Plot Pareto fronts of (performance, fairness, cost, energy).
- Use ablations to compare fixed thresholds vs learned policies under identical budgets.

Observability & Metrics (Expanded)

Goals: detect drift and retraining impacts, ensure decision traceability, and enforce operational budgets (latency, cost, energy).

Core SLIs (per service)
- Ingestion: batch ingest latency, batch rate, schema validation failures.
- Monitor: state emission latency, drift metric compute time, label-lag distribution.
- Policy-Agent: decision latency, action rate by type, action oscillation rate, cooldown overrides.
- Trainer: training duration, success/failure rate, model_update emission lag.
- Registry+Logger: event write latency, log backlog, replay throughput.

Event Integrity Metrics
- Stream lag per consumer group (batch/state/action/model_update/cost/label/dlq).
- DLQ rate and top reason_code counts.
- Idempotency replays per action_id/state_id/batch_id.

Model Quality Metrics
- O_perf and O_fair components over time (proxy vs ground-truth tracked separately).
- Drift core trends (PSI_mean, PSI_max, or MMD).
- Fairness proxy shift vs ground-truth deltas when labels arrive.

Cost & Energy Metrics
- Estimated $ per action and rolling spend.
- kWh and CO₂ per action and rolling totals.
- Latency impact per action and service p95/p99.

Decision Traceability
- Join keys: state_id -> action_id -> model_update -> cost -> labels.
- Model lineage: model_version and new_model_version tracked across events.

Dashboards (Grafana)
- Drift & quality dashboard (drift_core, perf_core, fairness_core, proxy vs GT).
- Decision dashboard (action mix, cooldown events, action oscillation rate).
- Cost & energy dashboard (per-action + rolling budget burn).
- Reliability dashboard (stream lag, DLQ, error rates).

Alerts (Prometheus)
- Stream lag above threshold for any consumer group.
- DLQ rate spike or recurring reason_code.
- Drift_core crossing alert threshold for N consecutive decision intervals with no subsequent action.
- Cost or energy budget burn-rate exceeds planned envelope.

In replay mode, observability metrics are recorded but alerts are disabled to avoid false positives during offline evaluation.

Testing Strategy

Scope: validate correctness of drift metrics, fairness metrics, event schemas, and policy semantics; ensure deterministic replay and safe handling of delayed labels.

Unit Tests

Drift metrics: PSI/KS/MMD calculations on synthetic shifts; verify monotonicity with shift magnitude.

Fairness metrics: DPD/EOD on toy datasets; verify behavior with sparse groups.

Schema validation: all event types validate against versioned schemas; invalid payloads routed to DLQ.

Policy logic: action_type validity, cooldown enforcement, idempotency on repeated action_id.

Integration Tests

End-to-end event chain: batch -> state -> action -> model_update + cost.

Delayed labels: emit label events later; verify reward computation uses ground-truth when available.

Replay determinism: re-run with same stream IDs and offsets; verify identical action sequences and effective_action_type (metric values may differ within numerical tolerance).

E2E Scenario Tests

No-drift steady state: expect defer actions and low costs.

Drift spike: expect fine_tune or retrain depending on constraints.

Fairness shock: fairness_core degrades; expect action shift under same budget.

Cooldown active: verify effective_action_type reflects deferral with reason_code.

Missing fairness signal: verify policy degrades gracefully using performance-only signals.

Performance & Reliability Tests

Stream lag under load; ensure backpressure causes policy to defer.

Trainer failure simulation: ensure DLQ capture and idempotent recovery without duplicate model_update.

Acceptance Criteria

Schema conformance: 100% events valid in normal runs.

Determinism: replay matches original action sequence and effective_action_type.

Drift sensitivity: detect seeded shifts with expected metric deltas.

Deployment Flow (Compose-first, MVP)
Goal: run a deterministic, event-driven loop locally with minimal operational overhead while preserving production-like separation of concerns.

Services (docker-compose)

redis (Redis Streams)

ingestion (dataset replay)

monitor (drift + state)

policy-agent (decision engine)

trainer (update executor)

registry-logger (canonical event log + artifacts)

prometheus + grafana (metrics + dashboards)

Startup Order

redis

registry-logger (initializes canonical event log and artifact directories)

ingestion

monitor

policy-agent

trainer

prometheus + grafana

Deployment Steps (local)

docker-compose up redis registry-logger

docker-compose up ingestion monitor policy-agent trainer

docker-compose up prometheus grafana

Runtime Controls

Replay mode toggle (environment variable) disables side effects (training/deploy) while still writing events.

Cooldown enforcement is handled by policy-agent + trainer using cooldown_expires_at.

Model artifacts are written to a versioned registry path; model_version is emitted with every event for traceability.

Service configuration (thresholds, budgets, cooldown durations) is provided via environment variables or versioned config files mounted into containers.

Rollback / Safety

If trainer fails, no model_update is emitted; a reason_code is logged in the cost event.

If drift_core exceeds thresholds with no subsequent action, an alert fires (disabled in replay mode).

Registry-logger can reconstruct full decision traces for post-mortem analysis.

Kubernetes orchestration and horizontal scaling are explicitly out of scope for the MVP and left as future work.

Evaluation Protocol — Scope & Experimental Conditions

The evaluation uses offline replay as the primary protocol, with a synthetic online simulation appendix to stress behaviors under bursty drift and varied label delay distributions. Offline replay treats the event log as the source of truth: we re-emit batch and label events in timestamp order, ensuring deterministic processing via Redis Stream IDs and checkpointed consumer offsets. The policy sees only the state vector core fields and never observes rewards at decision time, preserving delayed feedback constraints. Each experiment is run for multiple random seeds (policy initialization and stochastic components where applicable) and fixed budgets to isolate policy effects from resource variance. For comparability, all variants share identical data windows, label delay patterns, and cost/energy models; the only difference is the decision policy (threshold baselines vs learned policy). We report the objective vector per decision and aggregate across time windows.

Datasets and modalities follow the existing benchmark set (tabular, text, image), plus fairness-sensitive variants where labels are available for subgroup metrics. Drift is introduced either through the natural dataset chronology (offline) or through controlled replay perturbations (e.g., injected covariate shift spikes, subgroup distribution shifts, and delayed label arrivals) in the appendix.

The synthetic appendix replays the same log with stochastic perturbations (e.g., burst drift, uneven label latency), to test stability and robustness of action oscillations and budget adherence. We explicitly separate “primary results” (offline replay) from “stress results” (simulation appendix) to keep claims grounded in reproducible evidence.

Evaluation Protocol — Baselines, Metrics, and Plots

We compare the proposed policy against a full baseline set to cover standard practice, fairness-aware triggers, and cost-aware scheduling. Baselines include: (1) Fixed-interval retraining (e.g., every N windows), (2) Drift-threshold retraining (trigger when PSI/MMD exceeds a threshold), (3) No-retrain (serve last model), (4) Fairness-triggered retrain (trigger when DPD/EOD exceeds a threshold), and (5) Cost-aware retrain (trigger when a drift-utility score—expected performance gain minus retraining cost—exceeds a budgeted threshold). The fairness baseline uses static thresholds and does not account for cost or energy trade-offs. All baselines consume the same state core fields and budgets to keep comparisons consistent.

Metrics are reported at two levels: per-decision and aggregate over time. Per-decision metrics include the objective vector components (O_perf, O_fair, O_cost, O_energy), action type, cooldown status, and label delay. Aggregate metrics include cumulative cost/energy, average performance and fairness gaps, action frequency, and oscillation rate. We compute trade-off curves between accuracy/fairness and cost/energy, and report confidence intervals across random seeds and replay runs.

Plots are standardized for comparability:
- Time-series of O_perf, O_fair, cost/energy, and drift_core.
- Action timeline with overlay of drift events and cooldown windows.
- Pareto fronts across (performance, fairness, cost, energy).
- Budget burn curves comparing policies under fixed budget constraints.
- Stability plots (action oscillation rate vs drift intensity).

All plots are normalized by decision window to enable comparison across datasets and horizons.

Evaluation Protocol — Ablation Study Plan

Goal: isolate the impact of policy inputs, reward terms, and system signals on performance, fairness, and cost/energy trade-offs.

Policy Ablations
- Remove fairness_core from state: policy uses perf_core + drift_core only, measuring fairness regression under drift.
- Remove cost/energy terms from reward: O_cost/O_energy excluded from scalarization.
- Disable cooldown: allow actions every window to quantify oscillation penalties.
- Replace RL policy with static threshold rules (drift/fairness/cost), using the same state signals but no learning.

System Ablations
- Disable label stream: policy and rewards rely on proxy metrics only.
- Disable cost/energy telemetry: set O_cost/O_energy to constant baseline.
- Remove fairness proxy: fairness_core missing when labels delayed (policy must ignore fairness until labels arrive).
- Disable budget constraint: allow unrestricted retraining to measure upper-bound performance.

Reporting
- For each ablation, report objective vector deltas relative to the full system.
- Highlight which signals reduce oscillation and which improve fairness under drift.
- Ablations are interpreted qualitatively (directional effects) and quantitatively (objective deltas), rather than optimized for any single metric.

Threats to Validity

Internal Validity
- Implementation bias: policy behavior may reflect implementation choices (feature scaling, cooldown logic) rather than algorithmic merit.
- Replay determinism: subtle ordering differences or race conditions in consumers could alter action sequences.
- Reward leakage: accidental use of future labels or cost signals at decision time would inflate performance.

These threats are mitigated through deterministic replay, strict schema enforcement, and idempotent event processing.

External Validity
- Dataset representativeness: benchmark datasets may not reflect production drift patterns or business constraints.
- System constraints: budgets and label delays are modeled; real systems may have different distributions.
- Deployment context: results from Compose-based experiments may not fully generalize to large-scale distributed environments.

External validity is partially addressed through heterogeneous datasets and synthetic stress testing, though generalization remains limited.

Construct Validity
- Proxy metrics: fairness and performance proxies under label delay may not align with ground-truth outcomes.
- Objective weighting: scalarized rewards may obscure trade-offs despite reporting vector metrics.
- Drift metrics: PSI/MMD thresholds may not capture all operationally relevant shifts.

We mitigate construct validity risks by reporting both proxy and ground-truth metrics and by analyzing directional rather than absolute effects.

Conclusion Validity
- Statistical power: limited seeds or windows may inflate variance and reduce confidence in differences.
- Multiple comparisons: evaluating many baselines and ablations can increase false positives without correction.

We reduce conclusion validity threats by reporting confidence intervals and focusing on consistent trends across seeds and datasets.

Statistical Analysis (Bootstrap-Only)

We use non-parametric bootstrap resampling for all confidence intervals and comparisons, avoiding distributional assumptions. For each policy and dataset, we bootstrap over decision windows and seeds using paired resampling across policies under identical replay conditions to estimate 95% confidence intervals for the objective vector components and aggregate metrics (cumulative cost/energy, average performance/fairness gaps, oscillation rate). Pairwise policy comparisons are reported as bootstrap confidence intervals on the delta of metrics (policy A minus policy B). We focus on effect sizes and interval overlap rather than p-values.

Bootstrap Procedure
- Resample decision windows with replacement within each run.
- Resample runs/seeds to reflect policy stochasticity.
- Aggregate across datasets separately, then report macro-averaged effects.

Reporting
- 95% bootstrap confidence intervals for each metric and for policy deltas.
- Median and interquartile range (IQR) for skewed metrics (cost, energy).
- Emphasis on consistent directional effects across seeds and datasets.

We avoid formal multiple-hypothesis testing and instead interpret bootstrap intervals conservatively, focusing on consistent effects rather than marginal significance.

Different datasets are used to validate different aspects of the proposed architecture: tabular datasets validate drift and replay behavior, text datasets stress retraining cost asymmetry, cluster traces evaluate cost-aware scheduling, and fairness datasets test subgroup-level trade-offs. No single dataset is expected to validate all system properties.
