from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def regenerate(metric_csv: Path, out_dir: Path) -> None:
    metrics = pd.read_csv(metric_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Chart 1: Executive scorecard (normalized KPI heatmap + weighted ranking)
    metric_cols = [
        "ndcg_at_k",
        "recall_at_k",
        "coverage_at_k",
        "intra_list_diversity",
        "novelty",
        "latency_ms",
    ]
    metric_names = ["NDCG@10", "Recall@10", "Coverage@10", "Diversity", "Novelty", "Latency"]

    score_df = metrics[["model", *metric_cols]].copy()
    norm_df = score_df.copy()

    for col in metric_cols:
        vals = score_df[col].to_numpy(dtype=float)
        lo = float(np.min(vals))
        hi = float(np.max(vals))
        if hi - lo < 1e-12:
            norm = np.ones_like(vals) * 0.5
        else:
            norm = (vals - lo) / (hi - lo)
        if col == "latency_ms":
            norm = 1.0 - norm
        norm_df[col] = norm

    weights = {
        "ndcg_at_k": 0.35,
        "recall_at_k": 0.20,
        "coverage_at_k": 0.15,
        "intra_list_diversity": 0.10,
        "novelty": 0.05,
        "latency_ms": 0.15,
    }
    norm_df["overall_score"] = sum(norm_df[c] * w for c, w in weights.items())

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.6, 5.0), gridspec_kw={"width_ratios": [1.55, 1.0]})

    heat = norm_df[metric_cols].to_numpy(dtype=float).T
    im = ax0.imshow(heat, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax0.set_xticks(np.arange(len(norm_df)))
    ax0.set_xticklabels(norm_df["model"].tolist(), rotation=18)
    ax0.set_yticks(np.arange(len(metric_names)))
    ax0.set_yticklabels(metric_names)
    ax0.set_title("Executive KPI Scorecard (Normalized)")

    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax0.text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)

    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cbar.set_label("KPI score (0-1)")

    ranked = norm_df.sort_values("overall_score", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(ranked), dtype=float)
    ax1.barh(y_pos, ranked["overall_score"].to_numpy(dtype=float), color=["#FCA311", "#6BAED6", "#2A9D8F"])
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(ranked["model"].tolist())
    for i, row in enumerate(ranked.itertuples(index=False)):
        ax1.text(float(row.overall_score) + 0.01, float(i), f"{float(row.overall_score):.3f}", va="center", fontsize=8.5)
    ax1.set_xlim(0, 1.0)
    ax1.set_xlabel("Weighted score")
    ax1.set_title("Overall Model Ranking")
    ax1.grid(axis="x", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_dir / "metric_overview.png", dpi=180)
    plt.close(fig)

    # Chart 2: Pareto-style view
    fig2, ax2 = plt.subplots(figsize=(7.0, 5.2))
    x = metrics["intra_list_diversity"].to_numpy()
    y = metrics["ndcg_at_k"].to_numpy()
    coverage = metrics["coverage_at_k"].to_numpy()
    latency = metrics["latency_ms"].to_numpy()

    size = 120 + 2600 * coverage
    sc = ax2.scatter(
        x,
        y,
        s=size,
        c=latency,
        cmap="RdYlGn_r",
        edgecolors="white",
        linewidths=1.1,
        alpha=0.92,
    )

    frontier = metrics.sort_values(["intra_list_diversity", "ndcg_at_k"], ascending=[True, False])
    ax2.plot(
        frontier["intra_list_diversity"],
        frontier["ndcg_at_k"],
        color="#4F9DFF",
        linewidth=1.6,
        linestyle="--",
        alpha=0.85,
        label="Observed frontier",
    )

    for row in metrics.itertuples(index=False):
        label = f"{row.model}\nCov:{row.coverage_at_k:.3f}"
        ax2.annotate(
            label,
            (row.intra_list_diversity, row.ndcg_at_k),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=8.2,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.72, ec="none"),
        )

    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    ax2.axvline(x_mean, color="#777777", linestyle=":", linewidth=1.0, alpha=0.7)
    ax2.axhline(y_mean, color="#777777", linestyle=":", linewidth=1.0, alpha=0.7)
    ax2.text(x_mean + 0.0003, y_mean + 0.00025, "High accuracy + high diversity zone", fontsize=8.0, color="#E63946")

    cbar2 = fig2.colorbar(sc, ax=ax2)
    cbar2.set_label("Latency (ms, lower is better)")

    ax2.set_xlabel("Intra-list Diversity (higher is better)")
    ax2.set_ylabel("NDCG@10 (higher is better)")
    ax2.set_title("Accuracy-Diversity-Latency Pareto View")
    ax2.grid(alpha=0.32, linestyle="--")
    ax2.legend(loc="lower right", frameon=True)

    fig2.tight_layout()
    fig2.savefig(out_dir / "accuracy_diversity_tradeoff.png", dpi=180)
    plt.close(fig2)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    regenerate(root / "outputs" / "metrics.csv", root / "outputs")
    print("Charts regenerated in outputs/")
