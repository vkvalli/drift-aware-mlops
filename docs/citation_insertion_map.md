# Citation Insertion Map (for `DARE.pdf`, Apr 13, 2026)

This map is based on the current draft at `/Users/csuftitan/Downloads/DARE.pdf`.

## Quick Rule

- Use claim-level citations across all chapters (not only Chapter 2).
- Add section/page-level pin cites only for direct quotes, reproduced figures/tables, or exact borrowed equations.
- For your own experimental findings, use internal references (Table/Figure callouts), not external citations.

## Existing Reference IDs You Already Have

- `[1]` Regol et al. (When to retrain a machine learning model)
- `[2]` Kasundra et al. (Monitoring/retraining language models)
- `[3]` Pham et al. (Time to Retrain? concept drift detection)
- `[4]` Dong et al. (Data drift mitigation in ML pipelines)
- `[5]` Kumar et al. (Cost-efficient retraining under drift)
- `[6]` Sánchez-Mompó et al. (Green MLOps / energy)
- `[7]` Rank et al. (Performative RL in shifting environments)
- `[8]` Deho et al. (Fairness under covariate drift)

---

## Chapter 1: Insertions

1. Sentence cue: "models ... face non-stationary data"  
Add: `[3], [4]`  
Why: Drift prevalence in deployed systems.

2. Sentence cue: "retraining ... has real costs ... compute spend, energy usage"  
Add: `[5], [6]`  
Why: Operational and energy trade-offs.

3. Sentence cue: "labels can arrive late ... act using proxies"  
Add: `[1], [2]`  
Why: Delayed/limited feedback in retraining loops.

4. Sentence cue: "fixed schedules ... threshold rules brittle"  
Add: `[1], [3]`  
Why: Threshold sensitivity and trigger brittleness.

5. Sentence cue: "retraining as a decision process"  
Add: `[7]` (+ add a foundational contextual-bandit/LinUCB citation in bibliography)  
Why: Sequential policy framing.

6. Sentence cue: "fairness evaluation-first ... optional RL fairness shaping ablation"  
Add: `[8]`  
Why: Fairness degradation under drift motivates fairness monitoring.

7. Contribution list item on multi-objective trade-off (performance/cost/energy/fairness)  
Add: `[5], [6], [8]`  
Why: Each objective is externally grounded.

---

## Chapter 3: Insertions

1. Sentence cue: "three constraints ... non-stationary data, delayed supervision, bounded resources"  
Add: `[1], [3], [5]`  
Why: Problem assumptions should be sourced.

2. Sentence cue: "retraining as a sequential decision process"  
Add: `[7]`  
Why: RL/sequential control rationale.

3. Sentence cue: "delayed label events ... act under partial information"  
Add: `[1], [2]`  
Why: Delayed feedback realism.

4. Sentence cue: "drift estimated ... rolling reference"  
Add: `[3], [4]`  
Why: Drift-signal design grounding.

5. Sentence cue: "fairness treated as a lagged monitoring signal"  
Add: `[8]`  
Why: Fairness is non-stationary under drift.

6. Sentence cue: "balance intervention benefits against operational burden"  
Add: `[5], [6]`  
Why: Cost and energy burdens.

7. Sentence cue: "reward ... improve utility while avoiding unnecessary intervention"  
Add: `[5]` (and `[6]` if energy explicitly included in sentence)  
Why: Cost-aware reward shaping motivation.

8. Sentence cue: fairness-positioning text ("optional shaping", "deterministic fairness-trigger baseline")  
Add: internal evidence callouts to Chapter 5 fairness ablation table + fairness section.  
Why: This is your implementation claim, not external literature.

---

## Chapter 4: Insertions

1. Section `4.1 Datasets`: add one citation per dataset source line.  
Why: Dataset provenance should be citable in Methods/Experimental Design.

2. Sentence cue: "Airline ... significantly imbalanced ... F1 and balanced accuracy"  
Add: metric-definition citations (new refs needed; see "Missing related-work citations" below).  
Why: Metric choice rationale needs canonical backing.

3. Sentence cue: "California Housing ... MAE"  
Add: MAE definition/rationale citation (new ref).  
Why: Regression metric justification.

4. Sentence cue: "DPD and EOD" formula definitions  
Add: fairness-definition citations (new refs).  
Why: These definitions should cite canonical fairness literature.

5. Sentence cue: "chronological order ... decision step only had past observations"  
Add: `[1], [3]`  
Why: Temporal evaluation discipline.

6. Sentence cue: "label_delay_batches ∈ {0,2,5,10}" and delayed-label realism statement  
Add: `[1], [2]`  
Why: Delayed-label grounding.

7. Section `4.3 Policies evaluated`: "contextual bandit (LinUCB)"  
Add: foundational LinUCB/contextual-bandit citation (new ref) + optionally `[7]`.  
Why: Algorithm origin should be cited where introduced.

8. Section `4.4 Reward configuration`: cost penalty term  
Add: `[5]` (+ `[6]` if energy term discussed).  
Why: Cost-aware optimization literature.

9. Section `4.4`: fairness shaping term and fairness-trigger baseline language  
Add: `[8]` plus internal Chapter 5 ablation reference.  
Why: External motivation + internal evidence.

10. Section `4.5 Evaluation metrics`: accuracy/F1/balanced accuracy/precision/recall definitions  
Add: metric-definition citations (new refs).  
Why: Evaluation section should define and source metrics.

11. Section `4.5`: policy metrics (action rate, cost, update count)  
Add: no external citation required if these are your framework-defined diagnostics.  
Why: These are system-specific evaluation variables.

---

## Chapter 5: Insertions

1. All numeric comparison claims ("substantially reduced cost", "best F1", "lower action rate")  
Add: internal table references in same sentence (e.g., Table 5, Table 6).  
Why: Evidence should point to your results directly.

2. Sentence cue: "F1 more appropriate than raw accuracy on imbalanced Airline"  
Add: metric-definition citation (new ref) + optionally `[2]`.  
Why: Methodological justification.

3. Sentence cue: "rolling reward improved mean reward and reduced variance"  
Add: internal ablation table reference.  
Why: This is your empirical result.

4. Sentence cue: "conservative RL operating points robust to delayed supervision"  
Add: internal delayed-label table reference + `[1], [2]`.  
Why: Internal result + external realism context.

5. Sentence cue: "retraining should be framed as sequential decision problem"  
Add: `[7]` + `[1], [5]`.  
Why: External conceptual support.

6. Extension task paragraph (Regression/Text/Fairness)  
Add: internal experiment artifact/table references and Chapter 4 dataset references.  
Why: Keeps claims tied to evidence and scope.

7. Fairness statement (evaluation-first + optional shaping + no reliable action shift yet)  
Add: internal fairness ablation table callout and fairness subsection callout.  
Why: This is your thesis-specific observed behavior.

---

## Chapter 6: Insertions

1. "Fairness limited implementation"  
Add: `[8]` + internal fairness ablation result reference.  
Why: External need + current evidence boundary.

2. "No deep RL methods" / future RL expansion  
Add: `[7]` (+ foundational RL method refs if added).  
Why: Future direction should be anchored.

3. "No production deployment" / monitoring gap  
Add: `[2], [4]`  
Why: Deployment monitoring context.

4. "Scaling considerations / cost and energy"  
Add: `[5], [6]`  
Why: Scalability trade-offs are operational.

---

## Missing Related-Work Citations To Add (Recommended)

Your current bibliography `[1]`–`[8]` is strong for Chapter 2 positioning, but Chapters 3–5 would be more defensible with these added:

1. LinUCB / contextual bandits foundational paper (for Ch1, Ch4 policy definition).
2. Demographic Parity and Equal Opportunity canonical definitions (for Ch4 metric definitions).
3. Balanced Accuracy definition source (for Ch4/Ch5 metric rationale).
4. MAE metric definition/rationale source (for Ch4 regression metric grounding).
5. Dataset origin citations (Electricity, Airline/BTS, California Housing, AG News, Adult) in Ch4.

If you want, I can generate the exact IEEE-formatted bibliography entries for these missing references next.

---

## Cleanup Needed

1. Remove or complete dangling reference `[9]` in `REFERENCES`.
2. Ensure every new in-text citation has a matching bibliography entry (and vice versa).
3. In Chapter 5, replace generic comparative statements with explicit Table/Figure callouts where possible.
