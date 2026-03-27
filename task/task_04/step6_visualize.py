"""
step6_visualize.py
===================
Task 4 — Component 6: Generate publication-quality figures.

Produces three figures from the Task 4 results:

  Figure 1: diversity_histogram.png
    Histogram of per-image diversity scores (200 images).
    "Diverse" (>0.75) and "Repetitive" (<0.40) zones are shaded.

  Figure 2: diverse_vs_repetitive.png
    3-row grid: actual image thumbnail (or coloured placeholder) on the
    left, 5 generated captions + diversity score badge on the right.
    Two sides: top-3 most diverse (left half) vs top-3 most repetitive
    (right half).  Thumbnails loaded from results/images/img_{id}.jpg.

  Figure 3: steering_lambda_sweep.png
    Dual-axis line chart: λ on x-axis, mean caption length (left y-axis)
    and mean unique word count (right y-axis).  λ=0 baseline annotated.

Public API
----------
    plot_diversity_histogram(records, save_dir) -> str (path)
    plot_diverse_vs_repetitive(records, save_dir) -> str
    plot_steering_lambda_sweep(results, save_dir)  -> str
    visualize_all(records, steering_results, save_dir) -> dict[str, str]

Standalone usage
----------------
    export PYTHONPATH=.
    venv/bin/python task/task_04/step6_visualize.py
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
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────

C_DIVERSE    = "#4C72B0"   # blue
C_MEDIUM     = "#55A868"   # green
C_REPETITIVE = "#C44E52"   # red
C_LAMBDA     = "#DD8452"   # orange
C_UNIQ       = "#8172B2"   # purple


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Diversity histogram
# ─────────────────────────────────────────────────────────────────────────────

def plot_diversity_histogram(records: list,
                              save_dir: str = "task/task_04/results") -> str:
    """
    Histogram of per-image diversity scores.  Zones:
      "Repetitive"  < 0.40  ->  shaded red
      "Diverse"     > 0.75  ->  shaded blue
    """
    os.makedirs(save_dir, exist_ok=True)

    scores = [r["diversity_score"] for r in records]

    fig, ax = plt.subplots(figsize=(9, 5))

    # Shaded zones
    ax.axvspan(0.0, 0.40, alpha=0.12, color=C_REPETITIVE, label="Repetitive zone (<0.40)")
    ax.axvspan(0.75, 1.00, alpha=0.12, color=C_DIVERSE,   label="Diverse zone (>0.75)")

    # Histogram
    n_bins = 25
    ax.hist(scores, bins=n_bins, color=C_MEDIUM, edgecolor="white",
            linewidth=0.6, alpha=0.9, label="Image count")

    # Threshold lines
    ax.axvline(0.40, color=C_REPETITIVE, linewidth=1.8, linestyle="--")
    ax.axvline(0.75, color=C_DIVERSE,    linewidth=1.8, linestyle="--")

    # Mean line
    mean_score = np.mean(scores)
    ax.axvline(mean_score, color="#333", linewidth=1.4, linestyle=":",
               label=f"Mean = {mean_score:.3f}")

    ax.set_xlabel("Diversity Score  (unique n-grams / total n-grams)", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_title("Caption Diversity Distribution Across COCO Images\n"
                 "(5 nucleus-sampled captions per image, p=0.9)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=10)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_xlim(0, 1)

    # Annotations
    n_rep = sum(1 for s in scores if s < 0.40)
    n_div = sum(1 for s in scores if s > 0.75)
    ax.text(0.20, ax.get_ylim()[1] * 0.85, f"{n_rep} images\n(repetitive)",
            ha="center", va="top", color=C_REPETITIVE, fontsize=9)
    ax.text(0.875, ax.get_ylim()[1] * 0.85, f"{n_div} images\n(diverse)",
            ha="center", va="top", color=C_DIVERSE, fontsize=9)

    fig.tight_layout()
    path = os.path.join(save_dir, "diversity_histogram.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  OK  Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Diverse vs Repetitive  (image thumbnail + captions)
# ─────────────────────────────────────────────────────────────────────────────

def _load_thumb_array(img_id: int, save_dir: str):
    """Load thumbnail JPEG from results/images/. Returns numpy array or None."""
    from PIL import Image as PILImage
    path = os.path.join(save_dir, "images", f"img_{img_id}.jpg")
    if os.path.exists(path):
        try:
            return np.array(PILImage.open(path).convert("RGB"))
        except Exception:
            pass
    return None


def plot_diverse_vs_repetitive(records: list,
                                save_dir: str = "task/task_04/results") -> str:
    """
    3-row image-caption grid.

    Each row: [thumbnail | 5 captions] for one image.
    Left half = top-3 diverse,  right half = top-3 repetitive.
    Thumbnails come from results/images/img_{id}.jpg (generated by step3).
    Falls back to a coloured score-labelled placeholder if file missing.
    """
    os.makedirs(save_dir, exist_ok=True)

    def _get_top_unique(recs, reverse=True, n=3):
        sorted_recs = sorted(recs, key=lambda r: r["diversity_score"], reverse=reverse)
        unique_recs = []
        seen = set()
        for r in sorted_recs:
            cap_hash = tuple(r["captions"])
            if cap_hash not in seen:
                seen.add(cap_hash)
                unique_recs.append(r)
                if len(unique_recs) == n:
                    break
        return unique_recs

    diverse    = _get_top_unique(records, reverse=True, n=3)
    repetitive = _get_top_unique(records, reverse=False, n=3)

    N   = 3
    fig = plt.figure(figsize=(17, 12), facecolor="#F4F4F4")
    # 4 columns: [img_div | cap_div | img_rep | cap_rep]
    gs  = GridSpec(N, 4, figure=fig,
                   hspace=0.60, wspace=0.10,
                   left=0.03, right=0.97,
                   top=0.90, bottom=0.03,
                   width_ratios=[1, 2.2, 1, 2.2])

    def _render_row(row, rec, img_col, cap_col, badge_color):
        arr   = _load_thumb_array(rec["image_id"], save_dir)
        score = rec["diversity_score"]
        cat   = rec.get("category", "")

        # Image cell
        ax_img = fig.add_subplot(gs[row, img_col])
        ax_img.axis("off")
        if arr is not None:
            ax_img.imshow(arr, aspect="auto", interpolation="bilinear")
        else:
            ax_img.set_facecolor(badge_color)
            ax_img.text(0.5, 0.55, f"Image #{rec['image_id']}",
                        ha="center", va="center", color="white",
                        fontsize=9, fontweight="bold",
                        transform=ax_img.transAxes)
            ax_img.text(0.5, 0.35, f"{score:.3f}",
                        ha="center", va="center", color="white",
                        fontsize=9, transform=ax_img.transAxes)
        ax_img.set_title(f"Score: {score:.3f}  [{cat}]",
                         fontsize=8, color=badge_color,
                         pad=3, fontweight="bold")

        # Caption cell
        ax_cap = fig.add_subplot(gs[row, cap_col])
        ax_cap.set_facecolor("#FAFAFA")
        ax_cap.axis("off")
        y = 0.97
        for ci, cap in enumerate(rec["captions"][:5], 1):
            words, line, lines = cap.split(), "", []
            for w in words:
                if len(line) + len(w) + 1 > 52:
                    lines.append(line)
                    line = w
                else:
                    line = (line + " " + w).strip()
            if line:
                lines.append(line)
            first = True
            for ln in lines:
                prefix = f"{ci}. " if first else "   "
                ax_cap.text(0.03, y, prefix + ln,
                            transform=ax_cap.transAxes,
                            fontsize=7.8, color="#333", va="top",
                            fontfamily="monospace")
                y -= 0.13
                first = False
            y -= 0.04

    for row, rec in enumerate(diverse):
        _render_row(row, rec, img_col=0, cap_col=1, badge_color=C_DIVERSE)
    for row, rec in enumerate(repetitive):
        _render_row(row, rec, img_col=2, cap_col=3, badge_color=C_REPETITIVE)

    # Column headers
    fig.text(0.28, 0.945, "Top-3 Most DIVERSE Images",
             ha="center", va="bottom", fontsize=13, fontweight="bold",
             color=C_DIVERSE)
    fig.text(0.75, 0.945, "Top-3 Most REPETITIVE Images",
             ha="center", va="bottom", fontsize=13, fontweight="bold",
             color=C_REPETITIVE)

    # Centre divider
    sep = Line2D([0.505, 0.505], [0.02, 0.94],
                 transform=fig.transFigure,
                 color="#BBBBBB", linewidth=1.5, linestyle="--")
    fig.add_artist(sep)

    fig.suptitle("Caption Style Extremes — COCO Validation Set\n"
                 "(5 nucleus-sampled captions per image, top_p=0.9)",
                 fontsize=13, fontweight="bold", y=0.995)

    path = os.path.join(save_dir, "diverse_vs_repetitive.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  OK  Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Lambda sweep chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_steering_lambda_sweep(steering_results: list,
                                save_dir: str = "task/task_04/results") -> str:
    """
    Dual-axis line chart: lambda (x) vs mean caption length (left y) and
    mean unique word count (right y).  lambda=0 baseline marked.
    """
    os.makedirs(save_dir, exist_ok=True)

    lambdas = [r["lambda"]            for r in steering_results]
    lengths = [r["mean_length"]       for r in steering_results]
    uniq    = [r["mean_unique_words"] for r in steering_results]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    # Left axis: mean length
    ax1.plot(lambdas, lengths, "-o", color=C_LAMBDA, linewidth=2,
             markersize=7, label="Mean Caption Length (words)")
    ax1.set_xlabel("Steering Strength  (lambda)", fontsize=12)
    ax1.set_ylabel("Mean Caption Length (words)", color=C_LAMBDA, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=C_LAMBDA)

    # Right axis: unique words
    ax2 = ax1.twinx()
    ax2.plot(lambdas, uniq, "-s", color=C_UNIQ, linewidth=2,
             markersize=7, label="Mean Unique Words")
    ax2.set_ylabel("Mean Unique Word Count", color=C_UNIQ, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=C_UNIQ)

    # Baseline
    ax1.axvline(0.0, color="#999", linewidth=1.4, linestyle="--")
    ax1.text(0.02, max(lengths) * 0.97, "lambda=0\nbaseline",
             color="#777", fontsize=8.5)

    # Double-headed arrow annotation
    ax1.annotate("", xy=(lambdas[-1], lengths[-1] + 0.4),
                 xytext=(lambdas[0], lengths[0] + 0.4),
                 arrowprops=dict(arrowstyle="<->", color="#555", lw=1.2))
    mid_x = (lambdas[0] + lambdas[-1]) / 2
    ax1.text(mid_x, lengths[-1] + 0.6, "steering effect on length",
             ha="center", fontsize=8.5, color="#555")

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper left")

    ax1.set_title("Concept Steering Effect: lambda x d_short2detail\n"
                  "(BLIP decoder hidden-state injection, beam=3)",
                  fontsize=12, fontweight="bold", pad=10)
    ax1.grid(linestyle="--", alpha=0.35)
    fig.tight_layout()

    path = os.path.join(save_dir, "steering_lambda_sweep.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  OK  Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Master
# ─────────────────────────────────────────────────────────────────────────────

def visualize_all(records: list, steering_results: list,
                  save_dir: str = "task/task_04/results") -> dict:
    """
    Generate all three figures.

    Returns:
        dict with keys 'histogram', 'extremes', 'lambda_sweep' -> absolute paths
    """
    print("=" * 62)
    print("  Task 4 -- Step 6: Generate Visualizations")
    print("=" * 62)
    paths = {
        "histogram":    plot_diversity_histogram(records, save_dir),
        "extremes":     plot_diverse_vs_repetitive(records, save_dir),
        "lambda_sweep": plot_steering_lambda_sweep(steering_results, save_dir),
    }
    print(f"\n  3 figures saved to: {save_dir}")
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    div_cache = os.path.join(SAVE_DIR, "diversity_results.json")
    if os.path.exists(div_cache):
        with open(div_cache) as f:
            records = json.load(f)
        print(f"  Loaded diversity results from {div_cache}")
    else:
        from step3_diversity_analysis import _make_precomputed
        records = _make_precomputed()

    steer_cache = os.path.join(SAVE_DIR, "steering_results.json")
    if os.path.exists(steer_cache):
        with open(steer_cache) as f:
            steering_results = json.load(f)
        print(f"  Loaded steering results from {steer_cache}")
    else:
        from step5_steer_and_eval import PRECOMPUTED_STEERING
        steering_results = PRECOMPUTED_STEERING

    paths = visualize_all(records, steering_results, SAVE_DIR)
    print("\n  All figures generated.")
    for name, p in paths.items():
        print(f"   {name:14}: {p}")
