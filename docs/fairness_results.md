# Fairness Evaluation Results - Adult OpenML

**Experiment:** Adult OpenML dataset, seeds (111, 222, 333), `window_size=200`, `max_rows=40000`, `label_delay=2`

## Results (Mean Across Seeds)

| Policy | Accuracy | Bal. Acc | F1 | Cost ($) | DPD | EOD | Updates | Action Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fairness_trigger (F=0.05, T=0.01) | 0.5594 | 0.5314 | 0.2746 | 99.00 | 0.0154 | 0.0586 | 66.00 | 0.3300 |
| fixed_interval (N=10) | 0.5072 | 0.5215 | 0.2780 | 30.00 | 0.0097 | 0.0439 | 20.00 | 0.1000 |
| no_retrain | 0.7532 | 0.5000 | 0.0000 | 0.00 | 0.0000 | 0.0000 | 0.00 | 0.0000 |
| threshold_persistence (T=0.01, K=3) | 0.5560 | 0.5395 | 0.3047 | 292.50 | 0.0185 | 0.0705 | 195.00 | 0.9750 |

> **Note on no_retrain:** DPD and EOD are reported from the logged evaluation stream. When the system never updates, those values should be read as a degenerate baseline rather than as evidence of fairness.

## Interpretation

`threshold_persistence (T=0.01, K=3)` achieves the highest mean F1 (0.3047) among the retraining policies in this run.
`fixed_interval (N=10)` has the lowest mean DPD (0.0097) among policies that actually retrain.
`fixed_interval (N=10)` is the lowest-cost nontrivial policy at $30.00 mean cost.

## Methodological Notes

- `fairness_trigger` uses a lagged fairness proxy derived only from batches whose labels would already have arrived under the configured delay. It does not read current-batch labels at decision time.
- DPD and EOD in the table are post-hoc evaluation metrics computed on model predictions; they are distinct from the lagged proxy used by the trigger policy.
- All tables and plots in this file are generated from the aggregated seed runs, not from hardcoded values.
