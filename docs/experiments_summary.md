# Experiments Summary (Thesis-Facing)

This document contains milestone-level outcomes only.
For the full chronological history and all run folders, see `docs/experiments_log.md`.

## 1) Milestone Timeline (What Changed)

| Milestone | Scope | Source Artifact(s) | Outcome |
| --- | --- | --- | --- |
| M1 | Offline chain wired (`replay -> monitor -> policy -> trainer`) | `runs/run_seed_111/events.jsonl` (+ paired early runs) | Deterministic end-to-end JSONL pipeline established. |
| M2 | Electricity baseline calibration | `runs/matrix_20260211_063634/summary.csv` | Threshold persistence stabilized cost/action behavior (K tuning). |
| M3 | LinUCB operating-point calibration (Electricity/Airline path) | `runs/matrix_20260301_054818/summary.csv` | Coverage gate and conservative action-rate gates satisfied. |
| M4 | Airline policy comparison (120k) | `runs/airline_120k_policy_comparison.csv` | Locked LinUCB is lower-cost/lower-action while competitive on F1/balanced accuracy. |
| M5 | Delayed-label stress suite | `runs/delay_stress_summary.csv` | Stable behavior across delays with conservative calibration. |
| M6 | Multi-modal smoke extension | `runs/matrix_20260306_023101/summary.csv`, `runs/matrix_20260306_024457/summary.csv`, `runs/matrix_20260306_024630/summary.csv` | Regression/text/fairness-eval paths run end-to-end. |
| M7 | RL fairness term integration (Step 1) | `runs/adult_rl_wfair_sweep_20260410_161019/wfair_sweep_summary.csv`, `runs/adult_rl_step1_compare_20260410_165138/step1_comparison.csv`, `runs/adult_rl_covboost_20260410_173352/coverage_boost_ab_summary.csv` | Fairness-aware reward implemented and measurable, but no action-policy shift yet under conservative eval schedule. |

## 2) Locked Decisions

- RL algorithm: `LinUCB`
- Airline reward default: rolling `ΔF1` + cost shaping (`w_cost=0.25`, `action_penalty=0.02`, `normalize_perf=true`, `defer_bonus=0.01`, `defer_threshold=0.2`)
- Exploration schedule (locked airline operating point): `warm_start_n=20`, `warm_start_k=8`, `epsilon1=0.02`, `epsilon1_steps=30`, `epsilon2=0.02`
- Coverage gate: `mean update_count >= 8`
- Fairness in RL reward: implemented as optional term (`w_fair * fair_norm`), not locked as mandatory default.

## 3) Final Airline Comparison (120k, 3 seeds)

Source: `runs/airline_120k_policy_comparison.csv`

| Policy | Mean Updates | Mean Action Rate | Mean Cost (USD) | Mean F1 | Mean Balanced Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed_interval (N=10) | 58.00 | 0.0991 | 87.00 | 0.1632 | 0.5091 |
| threshold_persistence (T=1.8, K=3) | 67.00 | 0.1145 | 100.50 | 0.1541 | 0.4999 |
| linucb (locked) | 8.67 | 0.0148 | 13.00 | 0.2166 | 0.5117 |

## 4) Fairness Objective Step-1 Result (Adult)

### 4.1 `w_fair` Sweep (same policy/schedule)

Source: `runs/adult_rl_wfair_sweep_20260410_161019/wfair_sweep_summary.csv`

| w_fair | action_rate | updates | cost_usd | f1_mean | dpd_mean | eod_mean | reward_mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0.0217 | 4.33 | 6.50 | 0.3472 | 0.0065 | 0.0132 | -0.0163 |
| 1.0 | 0.0217 | 4.33 | 6.50 | 0.3472 | 0.0065 | 0.0132 | -0.0307 |
| 10.0 | 0.0217 | 4.33 | 6.50 | 0.3472 | 0.0065 | 0.0132 | -0.1597 |

### 4.2 Coverage-Boost A/B (`w_fair=0` vs `w_fair=1`)

Source: `runs/adult_rl_covboost_20260410_173352/coverage_boost_ab_summary.csv`

| config | action_rate | updates | cost_usd | f1_mean | dpd_mean | eod_mean | reward_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nofair | 0.0217 | 4.33 | 6.50 | 0.3472 | 0.0065 | 0.0132 | -0.0163 |
| wfair1 | 0.0217 | 4.33 | 6.50 | 0.3472 | 0.0065 | 0.0132 | -0.1016 |

Interpretation: Step-1 fairness term is operational (reward changes), but policy behavior remained unchanged under the current conservative evaluation schedule. This is reported as an identifiability/coverage result, not a wiring failure.

## 5) Delayed-Label Stress (Cross-Dataset)

Source: `runs/delay_stress_summary.csv`

| Dataset | label_delay_batches | action_rate | total_cost_usd | update_count | perf_mean | reward_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| electricity | 0 | 0.0300 | 4.5 | 3.0 | 0.6233 (accuracy) | 0.0125 |
| electricity | 10 | 0.0300 | 3.0 | 2.0 | 0.6423 (accuracy) | 0.0536 |
| airline | 0 | 0.0148 | 13.0 | 8.67 | 0.2159 (F1) | 0.0010 |
| airline | 10 | 0.0148 | 11.5 | 7.67 | 0.2173 (F1) | 0.0011 |

## 6) Scope Boundaries (Current Thesis Lock)

- Text mode is TF-IDF based in this phase (`feature_mode=tfidf`); SBERT is not implemented.
- Fairness metrics (`dpd_pre`, `eod_pre`) are benchmarked on binary classification (Adult) and should not be over-claimed as multiclass/regression fairness guarantees.
- Full run-by-run chronology and superseded calibrations are intentionally kept in `docs/experiments_log.md`.
