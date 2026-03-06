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


## Chunk MILESTONE-A-2026-03-06 (2026-03-06T10:39:24.310474+00:00)

### What Changed
- tools/plot_run.py
- runs/california_smoke_20260306_023033/events.jsonl
- runs/california_smoke_20260306_023033/plots/drift_timeseries.png
- runs/california_smoke_20260306_023033/plots/action_timeline.png
- runs/california_smoke_20260306_023033/plots/cost_curve.png
- runs/matrix_20260306_023101/summary.csv

### Why It Changed
- Unblocked regression plotting in headless run
- Completed California Housing smoke + mini matrix acceptance checks

### What To Verify
- SCIKIT_LEARN_DATA=.cache/sklearn_data .venv/bin/python tools/prepare_california_housing.py --out data/california_housing.csv
- .venv/bin/python -m src.runner.offline_run --csv data/california_housing.csv --task regression --label-column MedHouseVal --target-column MedHouseVal --perf-direction lower_better --dataset-id california_housing --window-size 200 --max-rows 40000 --label-delay-batches 2 --seed 111 --replay-output runs/california_smoke_20260306_023033/replay.jsonl --state-output runs/california_smoke_20260306_023033/state.jsonl --actions-output runs/california_smoke_20260306_023033/actions.jsonl --trainer-output runs/california_smoke_20260306_023033/trainer_output.jsonl --canonical-output runs/california_smoke_20260306_023033/events.jsonl --processed-index-file runs/california_smoke_20260306_023033/processed_actions.jsonl --dlq runs/california_smoke_20260306_023033/dlq.jsonl
- .venv/bin/python tools/plot_run.py --events runs/california_smoke_20260306_023033/events.jsonl --outdir runs/california_smoke_20260306_023033/plots
- .venv/bin/python tools/aggregate_matrix.py --matrix-dir runs/matrix_20260306_023101 --out runs/matrix_20260306_023101/summary.csv

### Expected Output
- Regression dataset CSV created
- Smoke run emits cost/model_update events
- Plot artifacts generated in run plots folder
- Matrix summary contains mae_/rmse_ fields

### New Metrics
- mae_pre
- rmse_pre
- r2_pre

### New Plots
- drift_timeseries.png
- action_timeline.png
- cost_curve.png

### New Log Fields
- cost.mae_pre
- cost.rmse_pre
- cost.r2_pre

### Known Limitations
- California Housing prep requires one-time network access unless dataset already present

### Next Chunk
- Milestone B: AG News smoke and matrix


## Chunk MILESTONE-B-2026-03-06 (2026-03-06T10:45:50.456999+00:00)

### What Changed
- src/monitor/monitor.py
- tests/test_text_monitor_signals.py
- runs/agnews_smoke_20260306_024442/events.jsonl
- runs/agnews_smoke_20260306_024442/plots/drift_timeseries.png
- runs/matrix_20260306_024457/summary.csv

### Why It Changed
- Fix degenerate text drift signal in monitor
- Complete AG News smoke + mini matrix acceptance checks

### What To Verify
- .venv/bin/python -m unittest tests/test_text_monitor_signals.py tests/test_monitor.py
- .venv/bin/python -m src.runner.offline_run --csv data/ag_news.csv --task text_classification --feature-mode tfidf --label-column class --dataset-id ag_news --window-size 200 --max-rows 40000 --label-delay-batches 2 --seed 111 --replay-output runs/agnews_smoke_20260306_024442/replay.jsonl --state-output runs/agnews_smoke_20260306_024442/state.jsonl --actions-output runs/agnews_smoke_20260306_024442/actions.jsonl --trainer-output runs/agnews_smoke_20260306_024442/trainer_output.jsonl --canonical-output runs/agnews_smoke_20260306_024442/events.jsonl --processed-index-file runs/agnews_smoke_20260306_024442/processed_actions.jsonl --dlq runs/agnews_smoke_20260306_024442/dlq.jsonl
- .venv/bin/python tools/aggregate_matrix.py --matrix-dir runs/matrix_20260306_024457 --out runs/matrix_20260306_024457/summary.csv

### Expected Output
- Unit tests pass for text monitor drift
- Smoke run produces non-constant drift_core after warmup
- Matrix summary includes accuracy/f1 fields

### New Metrics
- drift_core(text)=avg(abs(doc_len_shift)
- abs(entropy_shift))

### New Plots
- drift_timeseries.png
- action_timeline.png
- cost_curve.png

### New Log Fields
- state.doc_len_mean
- state.vocab_entropy_proxy

### Known Limitations
- Text drift magnitude remains small on synthetic AG News generator

### Next Chunk
- Milestone C: Adult fairness smoke + tradeoff plot


## Chunk MILESTONE-C-2026-03-06 (2026-03-06T10:47:35.878744+00:00)

### What Changed
- data/adult.csv
- runs/adult_smoke_20260306_024613/events.jsonl
- runs/matrix_20260306_024630/summary.csv
- runs/matrix_20260306_024630/plots/fairness_tradeoff.png

### Why It Changed
- Validate Adult fairness evaluation path end-to-end
- Generate fairness tradeoff artifact from matrix summary

### What To Verify
- .venv/bin/python tools/prepare_adult.py --out data/adult.csv --target-column income --rows 60000
- .venv/bin/python -m src.runner.offline_run --csv data/adult.csv --task classification --label-column income --target-column income --protected-attr sex --dataset-id adult --window-size 200 --max-rows 40000 --label-delay-batches 2 --seed 111 --replay-output runs/adult_smoke_20260306_024613/replay.jsonl --state-output runs/adult_smoke_20260306_024613/state.jsonl --actions-output runs/adult_smoke_20260306_024613/actions.jsonl --trainer-output runs/adult_smoke_20260306_024613/trainer_output.jsonl --canonical-output runs/adult_smoke_20260306_024613/events.jsonl --processed-index-file runs/adult_smoke_20260306_024613/processed_actions.jsonl --dlq runs/adult_smoke_20260306_024613/dlq.jsonl
- .venv/bin/python tools/aggregate_matrix.py --matrix-dir runs/matrix_20260306_024630 --out runs/matrix_20260306_024630/summary.csv
- .venv/bin/python tools/plot_fairness_tradeoff.py --summary runs/matrix_20260306_024630/summary.csv --out runs/matrix_20260306_024630/plots/fairness_tradeoff.png

### Expected Output
- Adult dataset prepared
- Smoke run emits fairness metrics in cost events
- Matrix summary contains dpd/eod fields
- Fairness tradeoff plot generated

### New Metrics
- dpd_pre
- eod_pre
- tp_pre
- fp_pre
- tn_pre
- fn_pre

### New Plots
- fairness_tradeoff.png

### New Log Fields
- cost.dpd_pre
- cost.eod_pre

### Known Limitations
- Adult preparation currently uses synthetic generator unless external input CSV is provided

### Next Chunk
- Milestone D: Update experiments_log modality subsections


## Chunk MILESTONE-D-2026-03-06 (2026-03-06T10:49:02.357977+00:00)

### What Changed
- docs/experiments_log.md

### Why It Changed
- Add compact modality subsections with setup
- metrics
- policy rows
- and plot links
- Document fairness as minimal evaluation-only demo

### What To Verify
- rg -n 'Multi-Modal Smoke Validation|California Housing \(Regression\)|AG News \(Text Classification\)|Adult \(Fairness Evaluation Demo\)' docs/experiments_log.md
- rg -n '^/|^[A-Za-z]:\\\\' docs/*.md

### Expected Output
- New modality sections present in experiments log
- No absolute machine-specific paths in docs

### New Metrics
- MAE/RMSE (regression)
- accuracy/F1 (text)
- DPD/EOD (fairness)

### New Plots
- runs/california_smoke_20260306_023033/plots/drift_timeseries.png
- runs/agnews_smoke_20260306_024442/plots/drift_timeseries.png
- runs/matrix_20260306_024630/plots/fairness_tradeoff.png

### New Log Fields
- cost.mae_pre
- cost.rmse_pre
- cost.dpd_pre
- cost.eod_pre
- state.doc_len_mean
- state.vocab_entropy_proxy

### Known Limitations
- Smoke tables are for pipeline verification; not all modalities are hyperparameter-tuned for final benchmark claims

### Next Chunk
- Milestone E/F: release notes and cleanup checks
