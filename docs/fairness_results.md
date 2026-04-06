# Fairness Evaluation Results — Adult OpenML

**Experiment:** Adult OpenML dataset, 3 seeds (111 / 222 / 333), `window_size=200`, `max_rows=40000`, `label_delay=2`

## Results (Mean Across 3 Seeds)

| Policy                    | Accuracy | Bal. Acc | F1     | Cost ($) | DPD    | EOD    | Updates |
|---------------------------|----------|----------|--------|----------|--------|--------|---------|
| no_retrain                | 0.7532   | 0.5000   | 0.0000 | 0.0      | 0.0000 | 0.0000 | 0.0     |
| fixed_interval (N=10)     | 0.5072   | 0.5215   | 0.2780 | 30.0     | 0.0097 | 0.0439 | 20.0    |
| threshold_persistence     | 0.5560   | 0.5395   | 0.3047 | 292.5    | 0.0185 | 0.0705 | 195.0   |

> **Note on no_retrain:** DPD and EOD are 0.0 because no model is ever fit, so no predictions are made and fairness metrics cannot be computed from model outputs.

## Interpretation

`threshold_persistence` achieves the best F1 (0.305) and balanced accuracy (0.540) among the retrained policies, but at a dramatically higher cost ($292.50 vs $30.00) and slightly worse fairness — its DPD (0.0185) and EOD (0.0705) are roughly double those of `fixed_interval`. `fixed_interval` (N=10) represents a substantially better cost-fairness tradeoff: it retrains only 20 times at $30 total while keeping DPD and EOD low, at the cost of a modest reduction in F1 (0.278 vs 0.305). For deployment scenarios where fairness and budget are primary constraints, `fixed_interval` is the preferred policy; `threshold_persistence` is only justified when maximising predictive performance outweighs cost and fairness concerns. The `no_retrain` baseline shows high accuracy (0.753) due to the majority-class label distribution, but its balanced accuracy of 0.500 and F1 of 0.000 confirm it never predicts the minority class.

## Limitations

**(a) DPD/EOD are computed on model predictions at eval time (post-hoc), not during training.**
Fairness metrics are computed from model predictions on each evaluation batch after each update, not from ground-truth group outcomes and not as a training constraint. This means the metrics reflect model bias at inference time rather than outcome disparity in the data itself, and may understate or overstate real-world fairness impact.

**(b) `no_retrain` shows 0.0 fairness metrics because no model is ever fit, so no predictions are made.**
Since `no_retrain` never calls `fine_tune`, no model is trained and no predictions are generated. DPD and EOD are therefore undefined and reported as 0.0 — this should not be interpreted as the policy being "fair", but rather as the absence of a model.

**(c) `threshold_persistence` over-retrains (195 updates vs 20 for `fixed_interval`) due to a low drift threshold on this dataset — threshold needs tuning per dataset.**
The drift threshold of 0.01 is too sensitive for the Adult OpenML dataset, causing nearly every batch to trigger a retrain. This inflates cost and may degrade fairness by repeatedly fitting on small, potentially unrepresentative windows. Threshold selection must be validated per dataset before deployment.

**(d) This is evaluation-only fairness, not fairness-constrained training.**
No fairness penalty or constraint is applied during model training. The reported DPD and EOD are post-hoc measurements of the trained model's output disparity. Reducing these metrics would require fairness-aware training objectives (e.g., adversarial debiasing, reweighting), which are not implemented here.

**(e) Binary protected attribute (`sex`) only; multiclass fairness not supported.**
All experiments use a binary `sex` attribute (Male/Female). The DPD and EOD formulations used here are defined for binary protected groups and binary classification outcomes. Extension to multiclass protected attributes or multiclass prediction targets requires additional methodology.

**(f) Synthetic label proxy used in monitor `fairness_core` signal (not model-prediction DPD).**
The `fairness_core` field in `StateEvent` is computed by the monitor as the difference in positive-label rates between protected groups in the incoming data window — a label-distribution proxy for DPD. It is not the same as the prediction-level DPD/EOD reported in the results table above, and should not be interpreted as a direct fairness guarantee on model outputs.
