"""
step4_visualize.py
===================
Task 3 — Component 4: Visualize ablation results.

Generates three publication-quality figures from the 9-config result data:

  1. cider_heatmap.png     — 3×3 heatmap of CIDEr by (beam_size × length_penalty)
  2. latency_barchart.png  — grouped bar chart of latency (s/100 images) per config
  3. metrics_scatter.png   — BLEU-4 vs CIDEr scatter, coloured by beam size

All figures are saved to `save_dir` (default: task/task_03/results/).

Public API
----------
    plot_cider_heatmap(results, save_dir="task/task_03/results") -> str  (path)
    plot_latency_barchart(results, save_dir)                     -> str
    plot_metrics_scatter(results, save_dir)                      -> str
    visualize_all(results, save_dir)                             -> dict[str, str]

Standalone usage
----------------
    export PYTHONPATH=.
    venv/bin/python task/task_03/step4_visualize.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BEAM_SIZES       = [1, 3, 5]
LENGTH_PENALTIES = [0.8, 1.0, 1.2]

PALETTE = {1: "#4C72B0", 3: "#DD8452", 5: "#55A868"}   # blue, orange, green


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _lookup(results: list, beam: int, lp: float, metric: str) -> float:
    for r in results:
        if r["beam_size"] == beam and abs(r["length_penalty"] - lp) < 1e-6:
            return r[metric]
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — CIDEr heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_cider_heatmap(results: list, save_dir: str = "task/task_03/results") -> str:
    """
    3×3 heatmap: rows = beam_size {1,3,5}, cols = length_penalty {0.8,1.0,1.2}.
    Cell value = CIDEr score.  Warmer = higher quality.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Build 2-D grid
    grid = np.array([[_lookup(results, bs, lp, "cider")
                       for lp in LENGTH_PENALTIES]
                      for bs in BEAM_SIZES])

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, cmap="YlOrRd", aspect="auto",
                   vmin=grid.min() - 0.02, vmax=grid.max() + 0.01)

    # Axis labels
    ax.set_xticks(range(len(LENGTH_PENALTIES)))
    ax.set_xticklabels([f"{lp:.1f}" for lp in LENGTH_PENALTIES], fontsize=12)
    ax.set_yticks(range(len(BEAM_SIZES)))
    ax.set_yticklabels([str(b) for b in BEAM_SIZES], fontsize=12)
    ax.set_xlabel("Length Penalty", fontsize=13, labelpad=8)
    ax.set_ylabel("Beam Size",      fontsize=13, labelpad=8)
    ax.set_title("CIDEr Score Heatmap\nBeam Size × Length Penalty", fontsize=14, fontweight="bold", pad=12)

    # Annotate each cell
    best_val = grid.max()
    for i, bs in enumerate(BEAM_SIZES):
        for j, lp in enumerate(LENGTH_PENALTIES):
            val = grid[i, j]
            colour = "white" if val < best_val - 0.04 else "black"
            marker = "★" if abs(val - best_val) < 1e-4 else ""
            ax.text(j, i, f"{val:.4f}{marker}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=colour)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("CIDEr Score", fontsize=11)
    fig.tight_layout()

    path = os.path.join(save_dir, "cider_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅  Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Latency bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_latency_barchart(results: list, save_dir: str = "task/task_03/results") -> str:
    """
    Grouped bar chart: x = length_penalty groups, bars = beam sizes.
    y-axis = seconds per 100 images.
    """
    os.makedirs(save_dir, exist_ok=True)

    x       = np.arange(len(LENGTH_PENALTIES))
    width   = 0.22
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(8, 5))

    for k, (bs, off) in enumerate(zip(BEAM_SIZES, offsets)):
        vals  = [_lookup(results, bs, lp, "latency_per_100") for lp in LENGTH_PENALTIES]
        cider = [_lookup(results, bs, lp, "cider")           for lp in LENGTH_PENALTIES]
        bars  = ax.bar(x + off, vals, width, label=f"beam={bs}",
                       color=PALETTE[bs], alpha=0.85, edgecolor="white", linewidth=0.5)
        # Annotate with CIDEr
        for bar, ci in zip(bars, cider):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    f"C={ci:.3f}", ha="center", va="bottom",
                    fontsize=7.5, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([f"lp={lp:.1f}" for lp in LENGTH_PENALTIES], fontsize=11)
    ax.set_xlabel("Length Penalty Config", fontsize=12)
    ax.set_ylabel("Latency (s / 100 images)", fontsize=12)
    ax.set_title("Generation Latency per Config\n(annotated with CIDEr score)", fontsize=13, fontweight="bold")
    ax.legend(title="Beam Size", fontsize=10, title_fontsize=10)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    path = os.path.join(save_dir, "latency_barchart.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅  Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Quality trade-off scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_metrics_scatter(results: list, save_dir: str = "task/task_03/results") -> str:
    """
    Scatter: x = latency (s/100), y = CIDEr.
    Each point = one config, coloured by beam size.
    Annotated with (beam, lp).
    Pareto-optimal frontier is drawn.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for r in results:
        col = PALETTE[r["beam_size"]]
        ax.scatter(r["latency_per_100"], r["cider"],
                   color=col, s=120, zorder=3, edgecolors="white", linewidth=0.8)
        ax.annotate(f"b={r['beam_size']}\nlp={r['length_penalty']}",
                    xy=(r["latency_per_100"], r["cider"]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=7.5, color="#333")

    # Pareto frontier (max CIDEr for each latency bucket)
    sorted_r = sorted(results, key=lambda r: r["latency_per_100"])
    pareto_x, pareto_y = [], []
    best_cider = -1.0
    for r in sorted_r:
        if r["cider"] > best_cider:
            best_cider = r["cider"]
            pareto_x.append(r["latency_per_100"])
            pareto_y.append(r["cider"])
    ax.step(pareto_x, pareto_y, where="post", color="#e83e3e",
            linewidth=1.5, linestyle="--", label="Pareto Frontier", zorder=2)

    # Legend patches
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=PALETTE[b], label=f"beam={b}") for b in BEAM_SIZES]
    legend_els.append(plt.Line2D([0], [0], color="#e83e3e", linestyle="--",
                                 label="Pareto Frontier"))
    ax.legend(handles=legend_els, fontsize=10)

    ax.set_xlabel("Latency (s / 100 images) ←  faster", fontsize=12)
    ax.set_ylabel("CIDEr Score  →  better quality", fontsize=12)
    ax.set_title("Quality vs. Speed Trade-off\n(each point = one beam × lp config)",
                 fontsize=13, fontweight="bold")
    ax.grid(linestyle="--", alpha=0.35)
    fig.tight_layout()

    path = os.path.join(save_dir, "quality_speed_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅  Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Master: run all three figures
# ─────────────────────────────────────────────────────────────────────────────

def visualize_all(results: list, save_dir: str = "task/task_03/results") -> dict:
    """
    Generate all three figures.

    Returns:
        dict with keys: 'heatmap', 'latency', 'scatter'  → absolute paths.
    """
    print("=" * 60)
    print("  Task 3 — Step 4: Generate Visualizations")
    print("=" * 60)
    paths = {
        "heatmap": plot_cider_heatmap(results, save_dir),
        "latency": plot_latency_barchart(results, save_dir),
        "scatter": plot_metrics_scatter(results, save_dir),
    }
    print(f"\n  3 figures saved to: {save_dir}")
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SAVE_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    CACHE_FILE  = os.path.join(SAVE_DIR, "ablation_results.json")

    # Load pre-computed or cached results
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            results = json.load(f)
        print(f"  Loaded results from {CACHE_FILE}")
    else:
        from step3_run_ablation import PRECOMPUTED_RESULTS
        results = PRECOMPUTED_RESULTS

    paths = visualize_all(results, SAVE_DIR)
    print("\n✅  All done. Open the PNG files in the results/ folder.")
    for name, p in paths.items():
        print(f"   {name:10}: {p}")
