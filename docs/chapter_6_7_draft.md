# CHAPTER 6
## LIMITATIONS AND FUTURE WORK

### 6.1 Limitations

#### 6.1.1 Fairness is evaluation-first in the current system
DARE currently treats fairness as an evaluation-first objective. Fairness metrics (DPD, EOD) are logged and analyzed in the canonical event pipeline, and fairness-trigger baselines are implemented as deterministic control policies. An optional RL fairness shaping term is also implemented, but current Step-1 results show reward-level sensitivity without reliable action-policy shifts under conservative operating settings. Therefore, the thesis does not claim fully fairness-optimized RL behavior at this stage.

#### 6.1.2 RL policy class and action space are intentionally constrained
The learned policy used in this thesis is a contextual bandit (LinUCB) calibrated for conservative operation. In the current configuration, learned actions are limited to `defer` and `fine_tune`, while `full_retrain` is implemented and evaluated primarily in heuristic baselines. This scope keeps calibration and interpretation tractable, but it limits expressiveness relative to broader RL formulations.

#### 6.1.3 Offline-first evidence path
Thesis evidence is produced through the offline JSONL/FileBus path with deterministic replay. Redis Streams topology is documented as a deployment target, but not the active backend for reported results. As a consequence, production concerns such as multi-service runtime contention, distributed failures, and live operational drift response are not fully evaluated in this thesis.

#### 6.1.4 Cost and energy signals are modeled, not physically measured
Cost and energy terms are tracked in the event schema and used in policy analysis, but they are modeled from fixed execution assumptions in the current trainer path. This design is suitable for controlled policy comparison, but it does not yet provide hardware-measured cost and energy accounting.

#### 6.1.5 Extension tasks are less mature than the main RL headline
Regression (California Housing), text classification (AG News), and fairness benchmarking (Adult OpenML) validate architectural generalization and logging correctness, but they are not as mature as the Airline RL line of evidence. The strongest thesis-facing RL claims remain anchored to the Airline 120k comparison.

#### 6.1.6 External validity and scaling boundaries
Although the thesis evaluates multiple datasets and delayed-label settings, results are still bounded by selected datasets, conservative calibration gates, and offline replay conditions. Generalization to high-throughput production environments and broader domain distributions remains an open validation step.

### 6.2 Future Work

#### 6.2.1 Fairness-action coupling beyond evaluation-first
The immediate next step is to move from fairness evaluation-first analysis toward reliable fairness-action coupling in RL. This includes improving identifiability and coverage so that fairness shaping can alter action behavior without violating cost or stability constraints. The fairness-trigger baseline remains the deterministic reference control path during this transition.

#### 6.2.2 Learned policies with richer action space
Future policy work should extend learned decisions to include `full_retrain` under explicit budget and stability constraints. This would allow direct learning of escalation behavior currently handled by rule-based policies.

#### 6.2.3 Online deployment validation on Redis Streams
A production-aligned next phase is to execute the same event contracts in the Redis Streams topology with end-to-end observability under realistic service load. This will test queue lag behavior, recovery semantics, and policy robustness outside offline replay.

#### 6.2.4 Measured operational cost and energy accounting
Future experiments should integrate measured hardware telemetry (wall-clock, GPU/CPU energy, and deployment latency impact) to replace or calibrate proxy cost terms and strengthen operational claims.

#### 6.2.5 Maturity expansion for extension tasks
California, AG News, and Adult should move from smoke/mini-matrix validation to full multi-seed, policy-complete benchmark lines with tuned baselines and locked reporting tables, similar to the Airline protocol.

#### 6.2.6 Broader statistical and robustness analysis
Subsequent work can increase seed coverage and add stronger uncertainty reporting for policy deltas across datasets and delays, while preserving the reproducibility guarantees of the canonical event pipeline.

# CHAPTER 7
## CONCLUSION

### 7.1 Summary of Contributions

This thesis introduced DARE (Drift-Aware Retraining Engine), an event-driven framework that formalizes model retraining as a sequential decision problem under distribution shift, delayed labels, and operational constraints. The work makes four central contributions:

1. A reproducible offline-first retraining architecture with schema-validated events, deterministic replay, and canonical trace logging.
2. A unified policy evaluation framework spanning no-retrain, fixed-interval, threshold-persistence, fairness-trigger, and learned contextual-bandit policies.
3. A conservative RL calibration protocol that emphasizes controlled intervention rates and sufficient update coverage before comparative claims.
4. Cross-task architectural validation across classification, regression, text, delayed-label stress settings, and fairness-evaluation pathways.

### 7.2 Main Findings

The strongest thesis-facing result is the Airline 120k policy comparison. Under the locked operating point, LinUCB achieved substantially lower intervention and cost than heuristic baselines while improving task-aligned performance:

- action rate: `0.0148` (vs `0.0991` fixed interval, `0.1145` threshold persistence),
- mean cost: `$13.0` (vs `$87.0` and `$100.5`),
- mean F1: `0.2166` (vs `0.1632` and `0.1541`),
- mean balanced accuracy: `0.5117` (vs `0.5091` and `0.4999`).

Electricity experiments provided calibration evidence for reward design, conservative action control, and replay robustness. Delayed-label stress tests across Electricity and Airline showed stable conservative behavior as label delay increased from 0 to 10 batches, with only minor update-coverage reduction at longest delays.

Extension-task results showed that the same architecture supports regression, text classification, and fairness diagnostics without changing core event contracts. These results are interpreted as architectural generalization evidence rather than final RL headline benchmarks.

### 7.3 Fairness Positioning in Final Claims

Fairness is evaluation-first in the current thesis scope, with optional RL fairness shaping ablation implemented. Fairness shaping changed reward values in controlled sweeps, but did not yet produce reliable action-policy shifts under current conservative settings. The fairness-trigger baseline remains the deterministic fairness control path.

### 7.4 Final Perspective

The central conclusion of this thesis is that retraining should be treated as a decision process, not a fixed schedule. DARE demonstrates that decision-aware retraining can preserve predictive quality while significantly reducing unnecessary operational intervention when policies are calibrated against explicit cost, coverage, and delayed-feedback constraints. This establishes a reproducible foundation for the next stage of research: fairness-action coupling, richer learned action spaces, and production-scale deployment validation.
