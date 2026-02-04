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