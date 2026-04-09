# Fairness Evaluation Results — Adult OpenML

**Experiment:** Adult OpenML dataset, 3 seeds (111 / 222 / 333), `window_size=200`, `max_rows=40000`

## Results (Mean Across 3 Seeds)

| Policy                    | Accuracy | Bal. Acc | F1     | Cost ($) | DPD    | EOD    | Updates |
|---------------------------|----------|----------|--------|----------|--------|--------|---------|
| no_retrain                | 0.7532   | 0.5000   | 0.0000 | 0.0      | 0.0000 | 0.0000 | 0.0     |
| fixed_interval (N=10)     | 0.5072   | 0.5215   | 0.2780 | 30.0     | 0.0097 | 0.0439 | 20.0    |
| threshold_persistence     | 0.5560   | 0.5395   | 0.3047 | 292.5    | 0.0185 | 0.0705 | 195.0   |

> **Note on no_retrain:** DPD and EOD are 0.0 because no model is ever fit, so no predictions are made and fairness metrics cannot be computed from model outputs.

## Interpretation

`threshold_persistence` achieves the best F1 (0.305) and balanced accuracy (0.540) among the retrained policies, but at a dramatically higher cost ($292.50 vs $30.00) and slightly worse fairness — its DPD (0.0185) and EOD (0.0705) are roughly double those of `fixed_interval`. `fixed_interval` (N=10) represents a substantially better cost-fairness tradeoff: it retrains only 20 times at $30 total while keeping DPD and EOD low, at the cost of a modest reduction in F1 (0.278 vs 0.305). For deployment scenarios where fairness and budget are primary constraints, `fixed_interval` is the preferred policy; `threshold_persistence` is only justified when maximising predictive performance outweighs cost and fairness concerns.

## Limitations

**(a) Prediction-based fairness metrics, not ground-truth outcomes.**
DPD and EOD are computed from model predictions on each evaluation batch (post-update), not from ground-truth group outcomes. This means the metrics reflect model bias rather than outcome disparity in the data itself, and may understate or overstate real-world fairness impact.

**(b) `fairness_core` in `StateEvent` is a label-distribution proxy, not prediction-based DPD.**
The `fairness_core` field computed by the monitor uses the difference in positive-label rates between protected groups in the incoming data window — a label-distribution proxy for DPD. It is not the same as the prediction-level DPD/EOD reported in the results table, and should not be interpreted as a direct fairness guarantee.

**(c) Binary classification only.**
All experiments use binary classification (income >50K vs ≤50K). The fairness metrics (DPD, EOD) are defined for binary outcomes; extension to multi-class settings requires additional methodology.

**(d) Synthetic / OpenML Adult dataset may not reflect real deployment distributions.**
The Adult dataset is a widely-used benchmark but is derived from 1994 US Census data. Its demographic composition, label distribution, and feature correlations may not generalise to modern or domain-specific deployment environments. Results should be interpreted as indicative rather than prescriptive.

**(e) No fairness-aware policy (`fairness_trigger`) was run in this batch — that is future work.**
The `fairness_trigger` policy (which retrains when `fairness_core` exceeds a threshold) was implemented but not included in this evaluation batch. A direct comparison of `fairness_trigger` against `fixed_interval` and `threshold_persistence` on DPD/EOD metrics is left as future work.
