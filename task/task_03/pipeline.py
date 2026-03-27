"""
pipeline.py
============
Task 3 — Master Orchestrator

Chains all 5 steps in sequence with progress banners and timing:

    Step 1: Load BLIP model + fine-tuned weights
    Step 2: Prepare 500 COCO validation images
    Step 3: Run 9-config beam × length-penalty ablation
    Step 4: Generate visualizations (heatmap, latency, scatter)
    Step 5: Analyze results + print key findings

Usage
-----
    # Full pipeline with live GPU inference:
    export PYTHONPATH=.
    venv/bin/python task/task_03/pipeline.py

    # Demo mode (no GPU needed — uses pre-computed results):
    venv/bin/python task/task_03/pipeline.py --demo

Outputs (all written to task/task_03/results/)
-----------------------------------------------
    ablation_results.json   — 9-config metric table
    findings.md             — written findings report
    cider_heatmap.png       — 3×3 CIDEr quality heatmap
    latency_barchart.png    — grouped latency bars per config
    quality_speed_scatter.png — Pareto frontier scatter plot
"""

import os
import sys
import json
import time
import argparse

# Allow running from the project root or the task folder
_TASK_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_TASK_DIR))
sys.path.insert(0, _PROJECT_DIR)

RESULTS_DIR = os.path.join(_TASK_DIR, "results")


def _banner(step: int, title: str):
    line = "─" * 68
    print(f"\n{line}")
    print(f"  TASK 3  |  Step {step}/5  |  {title}")
    print(f"{line}")


def run_pipeline(live: bool = False):
    """
    Run the complete Task 3 pipeline.

    Args:
        live: If True, performs live GPU inference for the ablation.
              If False (default), loads pre-computed results for all
              steps requiring inference.
    """
    t_total = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 1 — Load Model
    # ──────────────────────────────────────────────────────────────────────────
    _banner(1, "Load BLIP Model")
    t0 = time.time()

    from step1_load_model import load_model
    model, processor, device = load_model()

    print(f"  ⏱  Step 1 complete in {time.time() - t0:.1f}s")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 2 — Prepare Data (only needed for live mode)
    # ──────────────────────────────────────────────────────────────────────────
    _banner(2, "Prepare 500 COCO Validation Images")
    t0 = time.time()

    dataloader = None
    if live:
        from step2_prepare_data import load_val_data
        dataloader = load_val_data(processor, n=500, batch_size=8)
    else:
        print("  ⚡  DEMO mode — skipping data download for ablation step.")
        print("      (DataLoader would normally load 500 COCO val images here)")

    print(f"  ⏱  Step 2 complete in {time.time() - t0:.1f}s")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 3 — Run Ablation
    # ──────────────────────────────────────────────────────────────────────────
    _banner(3, "Run 9-Config Beam × Length-Penalty Ablation")
    t0 = time.time()

    from step3_run_ablation import (
        run_ablation, PRECOMPUTED_RESULTS, _load_or_use_precomputed, _print_summary
    )

    cache_path = os.path.join(RESULTS_DIR, "ablation_results.json")

    if live and dataloader is not None:
        print("  🔴  LIVE — running GPU inference on all 9 configs …")
        results = run_ablation(model, processor, dataloader, device,
                               save_dir=RESULTS_DIR)
    else:
        print("  ⚡  DEMO — loading/saving pre-computed ablation results …")
        results = _load_or_use_precomputed(RESULTS_DIR)
        results_sorted = sorted(results, key=lambda r: -r["cider"])
        _print_summary(results_sorted)

    print(f"  ⏱  Step 3 complete in {time.time() - t0:.1f}s")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 4 — Visualize
    # ──────────────────────────────────────────────────────────────────────────
    _banner(4, "Generate Visualizations")
    t0 = time.time()

    from step4_visualize import visualize_all
    figure_paths = visualize_all(results, save_dir=RESULTS_DIR)

    print(f"  ⏱  Step 4 complete in {time.time() - t0:.1f}s")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 5 — Analyze
    # ──────────────────────────────────────────────────────────────────────────
    _banner(5, "Analyze Results & Key Findings")
    t0 = time.time()

    from step5_analyze import analyze_results
    findings = analyze_results(results, save_dir=RESULTS_DIR)

    print(f"  ⏱  Step 5 complete in {time.time() - t0:.1f}s")

    # ──────────────────────────────────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    best    = findings["best_cider_config"]

    print("\n" + "═" * 68)
    print("  TASK 3 PIPELINE — COMPLETE")
    print("═" * 68)
    print(f"  Total time       : {elapsed:.1f}s")
    print(f"  Mode             : {'LIVE inference' if live else 'DEMO (pre-computed)'}")
    print(f"  Results dir      : {RESULTS_DIR}")
    print()
    print(f"  🏆 Best Config   : beam_size={best['beam_size']}, "
          f"length_penalty={best['length_penalty']}")
    print(f"     CIDEr         : {best['cider']:.4f}")
    print(f"     BLEU-4        : {best['bleu4']:.4f}")
    print(f"     METEOR        : {best['meteor']:.4f}")
    print(f"     ROUGE-L       : {best['rougeL']:.4f}")
    print(f"     Mean length   : {best['mean_length']:.1f} tokens")
    print(f"     Latency/100   : {best['latency_per_100']:.1f}s")
    print()
    print("  📁 Output files:")
    print(f"     ablation_results.json    — full 9-config metric table")
    print(f"     findings.md              — written analysis report")
    for name, path in figure_paths.items():
        print(f"     {os.path.basename(path):<28} — {name} figure")
    print("═" * 68)

    return findings


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Make step imports work from inside the task folder
    sys.path.insert(0, _TASK_DIR)

    parser = argparse.ArgumentParser(
        description="Task 3 Master Pipeline — Beam Search × Length Penalty Ablation"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Use pre-computed results (no GPU / data download required)"
    )
    args = parser.parse_args()

    run_pipeline(live=not args.demo)
