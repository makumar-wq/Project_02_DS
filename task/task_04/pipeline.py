"""
pipeline.py
============
Task 4 — Master Orchestrator

Chains all 7 steps in sequence with progress banners and timing:

    Step 1: Load BLIP model + fine-tuned weights
    Step 2: Prepare COCO validation data + style caption sets
    Step 3: Caption diversity analysis (5 nucleus-sampled captions/image)
    Step 4: Extract concept steering vectors (short / medium / detailed)
    Step 5: Steered caption generation — λ sweep [-1.0 … 2.0]
    Step 6: Generate visualizations (histogram, extremes panel, λ chart)
    Step 7: Analyze results → print findings + save findings.md

Usage
-----
    # Full pipeline with live GPU inference:
    export PYTHONPATH=.
    venv/bin/python task/task_04/pipeline.py

    # Demo mode (no GPU needed — uses pre-computed results):
    venv/bin/python task/task_04/pipeline.py --demo

Outputs (all written to task/task_04/results/)
-----------------------------------------------
    diversity_results.json      — per-image diversity records
    steering_vectors.pt         — d_short2detail, d_short2medium
    steering_vectors_meta.json  — steering vector metadata
    steering_results.json       — λ-sweep metrics table
    findings.md                 — written findings report
    diversity_histogram.png     — diversity score distribution
    diverse_vs_repetitive.png   — caption extremes panel
    steering_lambda_sweep.png   — λ vs length/uniqueness chart
"""

import os
import sys
import time
import argparse

# Allow running from the project root or the task folder
_TASK_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_TASK_DIR))
sys.path.insert(0, _PROJECT_DIR)

RESULTS_DIR = os.path.join(_TASK_DIR, "results")


def _banner(step: int, total: int, title: str):
    line = "─" * 68
    print(f"\n{line}")
    print(f"  TASK 4  |  Step {step}/{total}  |  {title}")
    print(f"{line}")


def run_pipeline(live: bool = False):
    """
    Run the complete Task 4 pipeline.

    Args:
        live: If True, performs live GPU inference for all heavy steps.
              If False (default), loads pre-computed results.
    """
    t_total = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    sys.path.insert(0, _TASK_DIR)   # Make step imports work

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 1 — Load Model
    # ──────────────────────────────────────────────────────────────────────────
    _banner(1, 7, "Load BLIP Model")
    t0 = time.time()
    from step1_load_model import load_model
    model, processor, device = load_model()
    print(f"  ⏱  Step 1 complete in {time.time() - t0:.1f}s")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 2 — Prepare Data
    # ──────────────────────────────────────────────────────────────────────────
    _banner(2, 7, "Prepare COCO Data + Style Caption Sets")
    t0 = time.time()
    dataloader  = None
    style_sets  = None
    if live:
        from step2_prepare_data import load_val_data, build_style_sets
        dataloader = load_val_data(processor, n=200, batch_size=4)
        style_sets = build_style_sets(n=500)
    else:
        print("  ⚡  DEMO mode — skipping data download.")
    print(f"  ⏱  Step 2 complete in {time.time() - t0:.1f}s")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 3 — Diversity Analysis
    # ──────────────────────────────────────────────────────────────────────────
    _banner(3, 7, "Caption Diversity Analysis")
    t0 = time.time()
    from step3_diversity_analysis import (
        run_diversity_analysis, _load_or_use_precomputed as _load_div,
        _print_diversity_summary
    )
    if live and dataloader is not None:
        print("  🔴  LIVE — nucleus sampling on all images …")
        records = run_diversity_analysis(model, processor, dataloader, device,
                                         save_dir=RESULTS_DIR)
    else:
        print("  ⚡  DEMO — loading/saving pre-computed diversity results …")
        records = _load_div(RESULTS_DIR)
        _print_diversity_summary(records)
    print(f"  ⏱  Step 3 complete in {time.time() - t0:.1f}s")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 4 — Steering Vectors
    # ──────────────────────────────────────────────────────────────────────────
    _banner(4, 7, "Extract Concept Steering Vectors")
    t0 = time.time()
    from step4_steering_vectors import (
        extract_steering_vectors, _load_or_use_precomputed as _load_vecs
    )
    import torch
    if live and style_sets is not None:
        print("  🔴  LIVE — extracting hidden states …")
        vectors = extract_steering_vectors(model, processor, style_sets, device,
                                           save_dir=RESULTS_DIR)
    else:
        print("  ⚡  DEMO — loading/saving pre-computed steering vectors …")
        vectors = _load_vecs(RESULTS_DIR)
    print(f"  ⏱  Step 4 complete in {time.time() - t0:.1f}s")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 5 — Steered Generation
    # ──────────────────────────────────────────────────────────────────────────
    _banner(5, 7, "Steered Caption Generation — λ Sweep")
    t0 = time.time()
    from step5_steer_and_eval import (
        run_steering_eval, _load_or_use_precomputed as _load_steer,
        _print_steering_summary, PRECOMPUTED_STEERING
    )
    if live and dataloader is not None:
        print("  🔴  LIVE — running steered generation …")
        vectors_dev = {k: v.to(device) for k, v in vectors.items()}
        steering_results = run_steering_eval(model, processor, dataloader, device,
                                             vectors_dev, save_dir=RESULTS_DIR,
                                             n_images=20)
    else:
        print("  ⚡  DEMO — loading/saving pre-computed steering results …")
        steering_results = _load_steer(RESULTS_DIR)
        _print_steering_summary(steering_results)
    print(f"  ⏱  Step 5 complete in {time.time() - t0:.1f}s")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 6 — Visualize
    # ──────────────────────────────────────────────────────────────────────────
    _banner(6, 7, "Generate Visualizations")
    t0 = time.time()
    from step6_visualize import visualize_all
    figure_paths = visualize_all(records, steering_results, save_dir=RESULTS_DIR)
    print(f"  ⏱  Step 6 complete in {time.time() - t0:.1f}s")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 7 — Analyze
    # ──────────────────────────────────────────────────────────────────────────
    _banner(7, 7, "Analyze Results & Key Findings")
    t0 = time.time()
    from step7_analyze import analyze_results
    findings = analyze_results(records, steering_results, save_dir=RESULTS_DIR)
    print(f"  ⏱  Step 7 complete in {time.time() - t0:.1f}s")

    # ──────────────────────────────────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    ds = findings["diversity_summary"]

    print("\n" + "═" * 68)
    print("  TASK 4 PIPELINE — COMPLETE")
    print("═" * 68)
    print(f"  Total time        : {elapsed:.1f}s")
    print(f"  Mode              : {'LIVE inference' if live else 'DEMO (pre-computed)'}")
    print(f"  Results dir       : {RESULTS_DIR}")
    print()
    print("  📊 Diversity Analysis:")
    print(f"     Images analysed : {ds['n_total']}")
    print(f"     Mean score      : {ds['avg_score']:.4f}")
    print(f"     Diverse (>0.75) : {ds['n_diverse']}  ({100*ds['n_diverse']/max(ds['n_total'],1):.1f}%)")
    print(f"     Repetitive (<0.40): {ds['n_repetitive']}  ({100*ds['n_repetitive']/max(ds['n_total'],1):.1f}%)")
    print()
    print("  🎯 Concept Steering (short → detailed):")
    print(f"     Best λ          : {findings['best_lambda']:+.1f}")
    print(f"     Length increase : +{findings['steering_effect']:.1f} words vs λ=0")
    print()
    print("  📁 Output files:")
    print(f"     diversity_results.json     — per-image diversity records")
    print(f"     steering_results.json      — λ-sweep metrics table")
    print(f"     findings.md                — written analysis report")
    for name, path in figure_paths.items():
        print(f"     {os.path.basename(path):<32} — {name} figure")
    print("═" * 68)

    return findings


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Task 4 Master Pipeline — Caption Diversity & Concept Steering"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Use pre-computed results (no GPU / data download required)"
    )
    args = parser.parse_args()
    run_pipeline(live=not args.demo)
