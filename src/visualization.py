"""Visualization and comparison tables for evaluation results."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"


def plot_bias_comparison(
    results: dict[str, pd.DataFrame],
    title: str = "CrowS-Pairs Bias Metric by Translation Method",
) -> plt.Figure:
    """Bar chart of bias metric (mean ± std) per method × model.

    Args:
        results: {"method_name": DataFrame with columns [model, metric_score]}
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    methods = list(results.keys())
    all_models = sorted(
        set(m for df in results.values() for m in df["model"].unique())
    )
    x = range(len(all_models))
    width = 0.8 / len(methods)

    for i, method in enumerate(methods):
        df = results[method]
        means, stds = [], []
        for model in all_models:
            sub = df[df["model"] == model]["metric_score"]
            means.append(sub.mean())
            stds.append(sub.std())
        offset = (i - len(methods) / 2 + 0.5) * width
        bars = ax.bar(
            [xi + offset for xi in x], means, width,
            yerr=stds, label=method, capsize=3,
        )

    ax.set_ylabel("Bias Metric Score")
    ax.set_title(title)
    ax.set_xticks(list(x))
    ax.set_xticklabels([m.split("/")[-1] for m in all_models], rotation=15)
    ax.legend()
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    fig.tight_layout()
    return fig


def plot_slot_comparison(
    results: dict[str, pd.DataFrame],
    title: str = "SNIPS Slot Filling & Intent Classification",
) -> plt.Figure:
    """Grouped bar chart for intent accuracy and slot F1."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    methods = list(results.keys())
    all_models = sorted(
        set(m for df in results.values() for m in df["model"].unique())
    )

    for ax, metric, label in zip(
        axes,
        ["intent_accuracy", "slot_f1"],
        ["Intent Accuracy", "Slot F1"],
    ):
        x = range(len(all_models))
        width = 0.8 / len(methods)
        for i, method in enumerate(methods):
            df = results[method]
            means, stds = [], []
            for model in all_models:
                sub = df[df["model"] == model][metric]
                means.append(sub.mean())
                stds.append(sub.std())
            offset = (i - len(methods) / 2 + 0.5) * width
            ax.bar(
                [xi + offset for xi in x], means, width,
                yerr=stds, label=method, capsize=3,
            )
        ax.set_ylabel(label)
        ax.set_xticks(list(x))
        ax.set_xticklabels([m.split("/")[-1] for m in all_models], rotation=15)
        ax.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_cultural_shift(
    results: pd.DataFrame,
    title: str = "Cultural Shift Quality by Method",
) -> plt.Figure:
    """Bar chart of mean cultural shift score per method."""
    fig, ax = plt.subplots(figsize=(10, 5))
    agg = results.groupby("method").agg(
        mean=("mean_score", "mean"),
        std=("mean_score", "std"),
    ).reset_index()
    ax.bar(agg["method"], agg["mean"], yerr=agg["std"], capsize=5)
    ax.set_ylabel("Mean Cultural Shift Score (1-5)")
    ax.set_title(title)
    ax.set_ylim(0, 5.5)
    fig.tight_layout()
    return fig


def build_summary_table(
    bias_results: dict[str, pd.DataFrame],
    slot_results: dict[str, pd.DataFrame],
    shift_results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a single summary table with all metrics across methods."""
    rows = []
    methods = set(bias_results.keys()) | set(slot_results.keys())

    for method in sorted(methods):
        row = {"method": method}
        if method in bias_results:
            bdf = bias_results[method]
            row["bias_mean"] = bdf["metric_score"].mean()
            row["bias_std"] = bdf["metric_score"].std()
        if method in slot_results:
            sdf = slot_results[method]
            row["intent_acc_mean"] = sdf["intent_accuracy"].mean()
            row["intent_acc_std"] = sdf["intent_accuracy"].std()
            row["slot_f1_mean"] = sdf["slot_f1"].mean()
            row["slot_f1_std"] = sdf["slot_f1"].std()
        if shift_results is not None:
            sub = shift_results[shift_results["method"] == method]
            if len(sub):
                row["shift_mean"] = sub["mean_score"].mean()
                row["shift_std"] = sub["mean_score"].std()
        rows.append(row)
    return pd.DataFrame(rows)


def format_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Format summary table with mean ± std strings."""
    out = summary[["method"]].copy()
    for col_mean, col_std, label in [
        ("bias_mean", "bias_std", "Bias Score"),
        ("intent_acc_mean", "intent_acc_std", "Intent Acc"),
        ("slot_f1_mean", "slot_f1_std", "Slot F1"),
        ("shift_mean", "shift_std", "Cultural Shift"),
    ]:
        if col_mean in summary.columns:
            out[label] = summary.apply(
                lambda r: f"{r[col_mean]:.3f} ± {r[col_std]:.3f}"
                if pd.notna(r.get(col_mean)) else "—",
                axis=1,
            )
    return out
