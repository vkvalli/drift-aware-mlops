"""
Generate fairness evaluation plots for the Adult OpenML 3-seed benchmark.

Outputs saved to the same directory as this script:
  - fairness_vs_performance.svg : DPD vs Mean Balanced Accuracy
  - fairness_vs_cost.svg        : DPD vs Mean Cost (USD)
"""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Data — mean across 3 seeds (111 / 222 / 333)
# Adult OpenML, window_size=200, max_rows=40000
# ---------------------------------------------------------------------------
policies   = ["no_retrain", "fixed_interval\n(N=10)", "threshold_persistence\n(T=0.01, K=3)"]
bal_acc    = [0.5000, 0.5215, 0.5395]
cost       = [0.0,    30.0,   292.5]
dpd        = [0.0000, 0.0097, 0.0185]

colors     = ["#4C72B0", "#DD8452", "#55A868"]
out_dir    = pathlib.Path(__file__).parent
out_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Plot 1: Fairness vs Performance (Balanced Accuracy)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
for label, x, y, c in zip(policies, bal_acc, dpd, colors):
    ax.scatter(x, y, color=c, s=140, zorder=3)
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(8, 4),
        textcoords="offset points",
        fontsize=9,
    )

ax.set_xlabel("Mean Balanced Accuracy", fontsize=11)
ax.set_ylabel("Mean DPD (lower = fairer)", fontsize=11)
ax.set_title("Fairness vs Performance (Adult OpenML, 3 seeds)", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.5)
ax.set_xlim(0.485, 0.555)
ax.set_ylim(-0.002, 0.024)
fig.tight_layout()
out1 = out_dir / "fairness_vs_performance.svg"
fig.savefig(out1, format="svg")
plt.close(fig)
print(f"Saved: {out1}")

# ---------------------------------------------------------------------------
# Plot 2: Fairness vs Cost
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
for label, x, y, c in zip(policies, cost, dpd, colors):
    ax.scatter(x, y, color=c, s=140, zorder=3)
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(8, 4),
        textcoords="offset points",
        fontsize=9,
    )

ax.set_xlabel("Mean Cost (USD)", fontsize=11)
ax.set_ylabel("Mean DPD (lower = fairer)", fontsize=11)
ax.set_title("Fairness vs Cost (Adult OpenML, 3 seeds)", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.5)
ax.set_xlim(-20, 340)
ax.set_ylim(-0.002, 0.024)
fig.tight_layout()
out2 = out_dir / "fairness_vs_cost.svg"
fig.savefig(out2, format="svg")
plt.close(fig)
print(f"Saved: {out2}")
