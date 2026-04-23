#!/usr/bin/env python3
"""Generate Chapter 5 thesis plots (sections 5.3-5.6)."""

from __future__ import annotations

from pathlib import Path
import os
import tempfile
import math

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache")
)
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
FIGS = ROOT / "docs" / "figures"


PALETTE = {
    "fixed_interval": "#2563eb",
    "threshold_persistence": "#0f766e",
    "linucb": "#dc2626",
    "linucb_locked": "#dc2626",
    "no_retrain": "#6b7280",
    "wfair": "#7c3aed",
}


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": "#1f2937",
            "axes.labelcolor": "#111827",
            "xtick.color": "#111827",
            "ytick.color": "#111827",
            "text.color": "#111827",
            "axes.grid": True,
            "grid.color": "#d1d5db",
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGS / f"{stem}.svg", format="svg", bbox_inches="tight")
    fig.savefig(FIGS / f"{stem}.png", format="png", dpi=320, bbox_inches="tight")
    fig.savefig(FIGS / f"{stem}.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


def _pareto_frontier(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    work = df.sort_values(x).reset_index(drop=True)
    keep = []
    best_y = -math.inf
    for _, row in work.iterrows():
        yv = float(row[y])
        if yv >= best_y:
            keep.append(True)
            best_y = yv
        else:
            keep.append(False)
    return work[pd.Series(keep)]


def plot_53_electricity_pareto() -> None:
    src = RUNS / "matrix_20260211_063634" / "summary.csv"
    df = pd.read_csv(src)
    df["accuracy_mean"] = pd.to_numeric(df["accuracy_mean"], errors="coerce")
    df["total_cost_usd"] = pd.to_numeric(df["total_cost_usd"], errors="coerce")
    agg = (
        df.groupby("policy", as_index=False)
        .agg(
            mean_cost_usd=("total_cost_usd", "mean"),
            mean_accuracy=("accuracy_mean", "mean"),
            mean_action_rate=("action_rate", "mean"),
        )
        .sort_values("mean_cost_usd")
    )
    order = ["no_retrain", "threshold_persistence", "fixed_interval"]
    agg["policy"] = pd.Categorical(agg["policy"], categories=order, ordered=True)
    agg = agg.sort_values("policy")

    valid = agg.dropna(subset=["mean_accuracy"]).copy()
    no_retrain = agg[agg["policy"] == "no_retrain"].copy()
    y_floor = float(valid["mean_accuracy"].min()) - 0.01

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for _, row in valid.iterrows():
        name = str(row["policy"])
        ax.scatter(
            row["mean_cost_usd"],
            row["mean_accuracy"],
            s=220,
            color=PALETTE.get(name, "#374151"),
            edgecolor="#111827",
            linewidth=1.2,
            zorder=3,
        )
        ax.annotate(
            name,
            (row["mean_cost_usd"], row["mean_accuracy"]),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=10,
        )

    frontier = _pareto_frontier(valid, "mean_cost_usd", "mean_accuracy")
    ax.plot(
        frontier["mean_cost_usd"],
        frontier["mean_accuracy"],
        color="#111827",
        linestyle="-",
        linewidth=1.6,
        label="Pareto frontier (maximize accuracy, minimize cost)",
        zorder=2,
    )

    if not no_retrain.empty:
        x_nr = float(no_retrain.iloc[0]["mean_cost_usd"])
        ax.scatter(
            [x_nr],
            [y_floor],
            marker="X",
            s=180,
            color=PALETTE["no_retrain"],
            edgecolor="#111827",
            linewidth=1.0,
            zorder=3,
        )
        ax.annotate(
            "no_retrain (accuracy N/A)",
            (x_nr, y_floor),
            xytext=(8, -14),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_title("Electricity: Cost-Accuracy Trade-off (Section 5.3)")
    ax.set_xlabel("Mean Cost (USD)")
    ax.set_ylabel("Mean Accuracy")
    ax.legend(loc="lower right", frameon=True)
    _save(fig, "ch5_3_electricity_pareto")


def plot_53_airline_pareto() -> None:
    src = RUNS / "airline_120k_policy_comparison.csv"
    df = pd.read_csv(src)
    df = df.rename(
        columns={
            "mean_total_cost_usd": "mean_cost_usd",
            "mean_f1": "mean_f1",
            "mean_action_rate": "mean_action_rate",
        }
    )
    df["policy_key"] = (
        df["policy"]
        .str.lower()
        .map(
            lambda x: (
                "fixed_interval"
                if "fixed_interval" in x
                else (
                    "threshold_persistence"
                    if "threshold" in x
                    else "linucb_locked" if "linucb" in x else x
                )
            )
        )
    )
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for _, row in df.iterrows():
        k = row["policy_key"]
        size = 120 + 2200 * float(row["mean_action_rate"])
        ax.scatter(
            row["mean_cost_usd"],
            row["mean_f1"],
            s=size,
            color=PALETTE.get(k, "#374151"),
            edgecolor="#111827",
            linewidth=1.0,
            alpha=0.92,
            zorder=3,
        )
        ax.annotate(
            str(row["policy"]),
            (row["mean_cost_usd"], row["mean_f1"]),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=9,
        )

    frontier = _pareto_frontier(df, "mean_cost_usd", "mean_f1")
    ax.plot(
        frontier["mean_cost_usd"],
        frontier["mean_f1"],
        color="#111827",
        linewidth=1.6,
        label="Pareto frontier",
        zorder=2,
    )
    ax.set_title("Airline: Pareto Trade-off (Cost vs. F1, bubble=Action Rate)")
    ax.set_xlabel("Mean Cost (USD)")
    ax.set_ylabel("Mean F1")
    ax.legend(loc="lower right", frameon=True)
    _save(fig, "ch5_3_airline_pareto")


def plot_54_delay_stress() -> None:
    src = RUNS / "delay_stress_summary.csv"
    df = pd.read_csv(src).sort_values(["dataset", "label_delay_batches"])
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    metrics = [
        ("perf_mean", "Performance Mean"),
        ("reward_mean", "Reward Mean"),
        ("action_rate_mean", "Action Rate"),
        ("update_count_mean", "Mean Updates"),
    ]
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        for dataset, g in df.groupby("dataset"):
            ax.plot(
                g["label_delay_batches"],
                g[metric],
                marker="o",
                linewidth=2.0,
                label=dataset,
            )
        ax.set_title(title)
        ax.set_xlabel("Label Delay (batches)")
    axes[0, 0].set_ylabel("Value")
    axes[1, 0].set_ylabel("Value")
    axes[0, 1].legend(frameon=True)
    fig.suptitle(
        "Delayed-Label Stress: Stability Across Delay Levels (Section 5.4)", y=1.02
    )
    _save(fig, "ch5_4_delay_stress_panels")

    fig2, ax2 = plt.subplots(figsize=(8.2, 5.0))
    for dataset, g in df.groupby("dataset"):
        ax2.plot(
            g["label_delay_batches"],
            g["proxy_groundtruth_gap"],
            marker="o",
            linewidth=2.0,
            label=dataset,
        )
    ax2.set_title("Proxy vs. Ground-Truth Gap vs. Label Delay (Section 5.4)")
    ax2.set_xlabel("Label Delay (batches)")
    ax2.set_ylabel("Proxy-GT Gap")
    ax2.legend(frameon=True)
    _save(fig2, "ch5_4_proxy_gap_vs_delay")


def _aggregate_policy_means(
    path: Path, policy_filter: str | None = None
) -> pd.DataFrame:
    df = pd.read_csv(path)
    if policy_filter:
        df = df[df["policy"] == policy_filter].copy()
    grouped = (
        df.groupby("policy", as_index=False)
        .agg(
            mean_cost_usd=("total_cost_usd", "mean"),
            mean_f1=("f1_mean", "mean"),
            mean_action_rate=("action_rate", "mean"),
        )
        .sort_values("mean_cost_usd")
    )
    return grouped


def plot_55_extensions() -> None:
    fair = pd.read_csv(
        RUNS / "adult_rl_wfair_sweep_20260410_161019" / "wfair_sweep_summary.csv"
    ).sort_values("w_fair")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.8))
    ax1.plot(
        fair["w_fair"],
        fair["reward_mean"],
        marker="o",
        linewidth=2.0,
        color=PALETTE["wfair"],
    )
    ax1.set_title("Fairness Weight Sweep: Reward Sensitivity")
    ax1.set_xlabel("w_fair")
    ax1.set_ylabel("Reward Mean")

    ax2.plot(fair["w_fair"], fair["dpd_mean"], marker="o", linewidth=2.0, label="DPD")
    ax2.plot(fair["w_fair"], fair["eod_mean"], marker="s", linewidth=2.0, label="EOD")
    ax2.set_title("Fairness Weight Sweep: Fairness Metrics")
    ax2.set_xlabel("w_fair")
    ax2.set_ylabel("Metric Value")
    ax2.legend(frameon=True)
    fig.suptitle("Fairness Extension Diagnostics (Section 5.5)", y=1.03)
    _save(fig, "ch5_5_fairness_weight_sweep")

    baseline = _aggregate_policy_means(RUNS / "matrix_20260412_233030" / "summary.csv")
    e22 = _aggregate_policy_means(
        RUNS / "matrix_20260412_233503" / "summary.csv", "linucb"
    )
    e26 = _aggregate_policy_means(
        RUNS / "matrix_20260413_000038" / "summary.csv", "linucb"
    )
    e22["policy"] = "linucb (epsilon2=0.22)"
    e26["policy"] = "linucb (epsilon2=0.26)"
    baseline = baseline.rename(columns={"policy": "policy"})
    ag = pd.concat([baseline, e22, e26], ignore_index=True)

    fig2, ax = plt.subplots(figsize=(8.6, 5.2))
    for _, row in ag.iterrows():
        policy = str(row["policy"])
        base_key = (
            "fixed_interval"
            if "fixed_interval" in policy
            else (
                "threshold_persistence"
                if "threshold" in policy
                else "no_retrain" if "no_retrain" in policy else "linucb"
            )
        )
        size = 100 + 1800 * float(row["mean_action_rate"])
        ax.scatter(
            row["mean_cost_usd"],
            row["mean_f1"],
            s=size,
            color=PALETTE.get(base_key, "#374151"),
            edgecolor="#111827",
            linewidth=1.0,
            alpha=0.92,
        )
        ax.annotate(
            policy,
            (row["mean_cost_usd"], row["mean_f1"]),
            xytext=(5, 7),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_title("AG News Extension: Cost vs. F1 (bubble=Action Rate)")
    ax.set_xlabel("Mean Cost (USD)")
    ax.set_ylabel("Mean F1")
    _save(fig2, "ch5_5_agnews_tradeoff")


def plot_56_policy_tradeoffs() -> None:
    df = pd.read_csv(RUNS / "airline_120k_policy_comparison.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.8))

    for _, row in df.iterrows():
        p = str(row["policy"])
        key = (
            "fixed_interval"
            if "fixed_interval" in p
            else "threshold_persistence" if "threshold" in p else "linucb_locked"
        )
        size = 90 + 7.5 * float(row["mean_total_cost_usd"])
        ax1.scatter(
            row["mean_action_rate"],
            row["mean_f1"],
            s=size,
            color=PALETTE.get(key, "#374151"),
            edgecolor="#111827",
            linewidth=1.0,
        )
        ax1.annotate(
            p,
            (row["mean_action_rate"], row["mean_f1"]),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=9,
        )
    ax1.set_title("Action Rate vs F1 (bubble=Cost)")
    ax1.set_xlabel("Mean Action Rate")
    ax1.set_ylabel("Mean F1")

    x = range(len(df))
    ax2.bar(
        x,
        df["mean_total_cost_usd"],
        color=["#2563eb", "#0f766e", "#dc2626"],
        alpha=0.85,
        label="Cost (USD)",
    )
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(df["policy"], rotation=20, ha="right")
    ax2.set_ylabel("Mean Cost (USD)")
    ax2.set_title("Cost and Updates by Policy")
    ax2b = ax2.twinx()
    ax2b.plot(
        x,
        df["mean_updates"],
        color="#111827",
        marker="o",
        linewidth=2.0,
        label="Mean Updates",
    )
    ax2b.set_ylabel("Mean Updates")
    fig.suptitle("Policy Trade-offs Summary (Section 5.6)", y=1.03)
    _save(fig, "ch5_6_policy_tradeoffs")


def main() -> None:
    _apply_style()
    plot_53_electricity_pareto()
    plot_53_airline_pareto()
    plot_54_delay_stress()
    plot_55_extensions()
    plot_56_policy_tradeoffs()
    print(f"Generated Chapter 5 plot set under {FIGS}")


if __name__ == "__main__":
    main()
