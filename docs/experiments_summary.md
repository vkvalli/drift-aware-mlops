# Experiments Summary (Thesis-Facing)

This document is the condensed, non-redundant summary of final experimental results and locked decisions.  
For full chronological history, intermediate runs, and superseded tables, see `docs/experiments_log.md`.
All final tables in this document are sourced from the referenced CSV artifacts under `runs/` and are the authoritative thesis comparison set.

## 1) Final Baseline and RL Scope

- Datasets:
  - Electricity (`accuracy`-oriented)
  - Airline (`F1`/balanced-accuracy oriented due to class imbalance)
  - AG News text path uses TF-IDF features in this phase (no SBERT mode yet)
- Baselines:
  - `no_retrain`
  - `fixed_interval`
  - `threshold_persistence` (including full-retrain variants where applicable)
- RL policy:
  - `LinUCB`

## 2) Locked Policy Decisions

- Electricity heuristic baseline for comparisons:
  - **Drift-Threshold + Persistence**
- Airline RL operating point (locked):
  - reward mode: `rolling` with `ΔF1` (`perf_window=10`)
  - `w_cost=0.25`, `action_penalty=0.02`, `normalize_perf=true`, `defer_bonus=0.01`, `defer_threshold=0.2`
  - `LinUCB(alpha=0.25)`
  - exploration: `warm_start_n=20`, `warm_start_k=8`, `epsilon1=0.02`, `epsilon1_steps=30`, `epsilon2=0.02`
  - coverage target: `mean update_count >= 8`

## 3) Final Airline Comparison (120k, 3 seeds)

Source: `runs/airline_120k_policy_comparison.csv`

| Policy | Mean Updates | Mean Action Rate | Mean Cost (USD) | Mean F1 | Mean Balanced Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed_interval (N=10) | 58.00 | 0.0991 | 87.00 | 0.1632 | 0.5091 |
| threshold_persistence (T=1.8, K=3) | 67.00 | 0.1145 | 100.50 | 0.1541 | 0.4999 |
| linucb (locked) | 8.67 | 0.0148 | 13.00 | 0.2166 | 0.5117 |

Conclusion: at the locked operating point, LinUCB is materially lower-cost and lower-action-rate while remaining competitive (and in this run higher) on F1.

## 4) Key Ablations That Changed Final Decisions

### 4.1 Airline metric choice (`Δaccuracy` vs `ΔF1`)

- `Δaccuracy` on imbalanced airline led to over-updating and poor task alignment.
- `ΔF1` produced stable update rates and stronger minority-class-aware behavior.
- This is treated as metric-to-task alignment (construct validity), not outcome chasing.

### 4.2 Reward shaping (`raw` vs `rolling`)

- `rolling` reward produced smoother signal and better stability than `raw`.
- Result: `rolling` retained as default RL reward.

### 4.3 Shadow reward variant

- Shadow-buffer reward (update-only credit on held-out shadow batches) was more causally isolated but lower-signal/higher-variance.
- Result: keep shadow as robustness ablation; keep rolling as default.

### 4.4 Airline class weighting

- Weighted training improved F1 for retraining policies versus unweighted runs.
- Result: retain class weighting in airline trainer path.

## 5) Delayed-Label Stress (Cross-Dataset)

Source: `runs/delay_stress_summary.csv`

| Dataset | label_delay_batches | action_rate | total_cost_usd | update_count | perf_mean | reward_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| electricity | 0 | 0.0300 | 4.5 | 3.0 | 0.6233 (accuracy) | 0.0125 |
| electricity | 2 | 0.0300 | 4.5 | 3.0 | 0.6249 (accuracy) | 0.0145 |
| electricity | 5 | 0.0300 | 4.5 | 3.0 | 0.6295 (accuracy) | 0.0211 |
| electricity | 10 | 0.0300 | 3.0 | 2.0 | 0.6423 (accuracy) | 0.0536 |
| airline | 0 | 0.0148 | 13.0 | 8.67 | 0.2159 (F1) | 0.0010 |
| airline | 2 | 0.0148 | 13.0 | 8.67 | 0.2166 (F1) | 0.0010 |
| airline | 5 | 0.0148 | 13.0 | 8.67 | 0.2178 (F1) | 0.0010 |
| airline | 10 | 0.0148 | 11.5 | 7.67 | 0.2173 (F1) | 0.0011 |

Conclusion: both datasets remained stable across label-delay stress under conservative calibration; airline update coverage dips slightly only at the highest delay.

## 6) What Is Considered Final vs Superseded

- Final for thesis comparison:
  - Airline 120k policy table
  - Locked Airline RL operating point and gates
  - Delayed-label stress summary
- Active extension phase (in progress):
  - Multi-modal generalization (`classification`, `regression`, `text_classification`)
  - Evaluation-only fairness diagnostics (`dpd_pre`, `eod_pre`) scoped to binary classification runs on the Adult OpenML benchmark
- Superseded:
  - early electricity calibration tables
  - early LinUCB low-coverage runs before the `update_count >= 8` gate was satisfied
  - intermediate reward sweeps used only for calibration
