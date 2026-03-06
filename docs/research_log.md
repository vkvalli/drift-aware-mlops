Identified research gap:

Despite extensive research on data and concept drift detection, cost-aware retraining, and fairness in non-stationary environments, existing approaches address these challenges in isolation. Prior work on retraining decisions primarily optimizes predictive performance under cost constraints, treating retraining as a trade-off between accuracy degradation and computational expense, while largely ignoring fairness dynamics, energy consumption, and broader operational constraints. Conversely, fairness-focused studies demonstrate that algorithmic fairness is itself non-stationary and can deteriorate over time due to dataset and concept drift, yet they stop short of formulating fairness-aware retraining as an explicit decision problem with operational trade-offs.

Similarly, drift detection and monitoring frameworks provide increasingly robust signals—often under limited or delayed label availability—but typically treat drift detection as a binary trigger for retraining rather than as one of several competing inputs into a principled scheduling policy. In distributed and federated learning settings, recent work highlights group-specific and client-specific concept drift, further complicating fairness preservation, yet these methods focus on adaptation mechanisms rather than end-to-end retraining orchestration within production MLOps pipelines.

As a result, there is currently no unified framework that models model retraining as a sequential, multi-objective decision problem that jointly considers predictive performance, fairness, and operational cost (including computational and energy-related constraints) under real-world drift conditions. In particular, existing systems lack adaptive, policy-based retraining mechanisms that can dynamically balance these objectives over time, account for delayed or partial feedback, and operate within realistic production environments. This gap motivates the development of an autonomous, drift-aware MLOps agent that integrates heterogeneous drift and fairness signals with resource-aware decision-making to determine not only when to retrain models, but also how and to what extent retraining should occur.

next steps:

Do this for fairness:

Step 1: Pick Adult dataset (fastest fairness dataset)
	•	Binary classification.
	•	Protected attribute: sex.
	•	Already tabular.
	•	Easy to compute DPD/EOD.

Step 2: Add fairness metrics to evaluation only

Log:
	•	dpd_pre
	•	eod_pre

No RL change yet.

Step 3: Show fairness drift over time

Plot fairness metric vs batch.

Step 4: Add fairness to reward (lightweight)

Modify reward:
R = Δperf - λ_cost * cost - λ_fair * fairness_gap
Try 2–3 λ_fair values.

Show trade-off curve:
	•	performance vs fairness vs cost.

That’s enough.

What I’d do (min effort, max defensibility)
	1.	Keep electricity + airline as your main, fully analyzed RL results.
	2.	Add AG News (text) with the same pipeline:
	•	windows, drift signal, policy decisions, reward based on ΔF1 or Δaccuracy with cost.
	3.	Add Rotated MNIST (image) as a “stress modality”:
	•	even a small run showing drift triggers and the agent behaving reasonably is enough.
	4.	Put Google Cluster Trace as “future work / appendix” unless you really want to spend days on data engineering.

This gets you “multiple modalities” without ballooning scope.


for regression task:
Best dataset pick

Use one of these (in this order):
	1.	Ames Housing (Kaggle “House Prices: Advanced Regression Techniques”)

	•	Great classic dataset, widely used, tons of features.
	•	Target: SalePrice.

	2.	California Housing (scikit-learn built-in)

	•	Even easier if you want zero download hassle.
	•	Target: MedHouseVal (median house value).

If your priority is “least moving parts”: pick California Housing.

⸻

How this maps to your current pipeline

Right now you log classification metrics (accuracy/F1). For regression you swap:
	•	Perf metric: MAE or RMSE (I recommend MAE, it’s more stable and interpretable).
	•	Reward delta: improvement means error goes down.

So instead of Δaccuracy you do:
	•	Δperf = (MAE_pre - MAE_post)  (positive = improvement)

or in rolling mode:
	•	Δperf = rolling(MAE_pre_prev) - rolling(MAE_pre_now)

Then keep cost penalty exactly the same.

⸻

Minimal “regression support” changes you’ll need
	1.	Trainer supports regression model:
	•	SGDRegressor (partial_fit works like your SGDClassifier)
	2.	Eval metrics logged in CostEvent:
	•	mae_pre, maybe rmse_pre
	3.	RL dataset builder:
	•	--perf-field mae_pre
	•	interpret delta as “lower is better” (invert sign)

That’s it.

⸻

Recommendation

Pick California Housing first (fastest to integrate), then if time allows add Ames for realism.




## Chunk X.1 (2026-03-06T06:26:14.284818+00:00)

### What Changed
- docs/implementation-plan.md
- tools/log_chunk_update.py
- README.md

### Why It Changed
- Added tracked multi-modal phase to implementation plan
- Added thesis-ready chunk logger CLI
- Documented guardrail usage

### What To Verify
- rg -n 'Phase X: Multi-Modal Extension' docs/implementation-plan.md
- python3 tools/log_chunk_update.py --help

### Expected Output
- Phase section present in implementation plan
- CLI help lists required arguments

### New Metrics
- mae_pre
- rmse_pre
- r2_pre
- dpd_pre
- eod_pre

### New Plots
- artifacts/plots/fairness_tradeoff.svg

### New Log Fields
- task
- target_column
- perf_direction
- protected_attr

### Known Limitations
- Regression/text/fairness integrations pending verification

### Next Chunk
- Complete multi-task trainer and matrix aggregation updates


## Chunk X.2 (2026-03-06T06:31:39.496157+00:00)

### What Changed
- src/trainer/trainer.py
- src/trainer/compat.py
- src/monitor/monitor.py
- src/runner/offline_run.py
- tools/run_matrix.py
- tools/build_rl_dataset.py
- tools/aggregate_matrix.py
- src/schemas/cost.py
- tools/prepare_california_housing.py
- tools/prepare_ag_news.py
- tools/prepare_adult.py
- tools/plot_fairness_tradeoff.py
- tests/test_regression_trainer.py
- tests/test_text_trainer.py
- tests/test_fairness_metrics.py

### Why It Changed
- Generalized pipeline to multi-task runtime mode
- Added regression metrics and lower-better reward support
- Added text classification TF-IDF path
- Added fairness diagnostics logging and reporting

### What To Verify
- python3 -m unittest
- python3 -m unittest tests/test_regression_trainer.py tests/test_text_trainer.py tests/test_fairness_metrics.py
- git branch --list thesis-review-build

### Expected Output
- All tests pass (with known skips)
- New multi-task tests pass
- thesis-review-build branch exists

### New Metrics
- mae_pre
- rmse_pre
- r2_pre
- dpd_pre
- eod_pre

### New Plots
- artifacts/plots/fairness_tradeoff.svg

### New Log Fields
- task
- feature_mode
- target_column
- protected_attr
- perf_direction
- doc_len_mean
- vocab_entropy_proxy

### Known Limitations
- Smoke runs for California Housing and AG News not executed in this environment

### Next Chunk
- Run modality-specific matrices and update experiments_summary tables
