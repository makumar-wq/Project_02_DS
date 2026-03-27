"""
pipeline.py
============
Task 5 — Master Orchestrator: Toxicity & Bias Detection Pipeline

Chains all 7 steps in order. Supports --demo mode (pre-computed fallbacks)
or live GPU inference mode.

Usage
-----
    # Demo mode — uses precomputed results (~8 seconds, no GPU needed)
    export PYTHONPATH=.
    venv/bin/python task/task_05/pipeline.py --demo

    # Live mode — generates 1000 captions then runs full analysis
    venv/bin/python task/task_05/pipeline.py
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR  = os.path.join(TASK_DIR, "results")


def _banner(step: int, title: str, t0: float):
    elapsed = time.time() - t0
    print(f"\n{'═'*68}")
    print(f"  TASK 5  |  Step {step}/7  |  {title}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'═'*68}")


def run_pipeline(demo: bool = True):
    import time
    t0 = time.time()

    print("═" * 68)
    print("  TASK 5 PIPELINE — Toxicity & Bias Detection with Mitigation")
    print(f"  Mode: {'DEMO (pre-computed)' if demo else 'LIVE (GPU inference)'}")
    print(f"  Results dir: {SAVE_DIR}")
    print("═" * 68)

    # ─── Step 1: Load models ──────────────────────────────────────────────────
    _banner(1, "Load Models", t0)
    from step1_load_model import load_model, load_toxicity_model
    if not demo:
        model, processor, device = load_model()
        tox_tok, tox_mdl         = load_toxicity_model()
    else:
        model = processor = device = None
        tox_tok = tox_mdl = None
    t1 = time.time()
    print(f"  ⏱  Step 1 complete in {t1-t0:.1f}s")

    # ─── Step 2: Prepare captions ─────────────────────────────────────────────
    _banner(2, "Caption Generation / Load (1000 captions)", t0)
    from step2_prepare_data import generate_captions, _load_or_use_precomputed as load_caps
    if demo:
        caption_records = load_caps(SAVE_DIR)
    else:
        caption_records = generate_captions(model, processor, device,
                                            n=1000, save_dir=SAVE_DIR)
    t2 = time.time()
    print(f"  ⏱  Step 2 complete in {t2-t1:.1f}s")

    # ─── Step 3: Toxicity scoring ─────────────────────────────────────────────
    _banner(3, "Toxicity Scoring (unitary/toxic-bert)", t0)
    from step3_toxicity_score import (run_toxicity_scoring,
                                       _load_or_use_precomputed as load_tox,
                                       _print_toxicity_summary)
    if demo:
        tox_scores = load_tox(SAVE_DIR, caption_records)
    else:
        tox_scores = run_toxicity_scoring(caption_records, tox_tok, tox_mdl,
                                          save_dir=SAVE_DIR)
    _print_toxicity_summary(tox_scores)
    t3 = time.time()
    print(f"  ⏱  Step 3 complete in {t3-t2:.1f}s")

    # ─── Step 4: Bias audit ───────────────────────────────────────────────────
    _banner(4, "Bias / Stereotype Audit", t0)
    from step4_bias_audit import run_bias_audit, _load_or_use_precomputed as load_bias
    if demo:
        bias_records, freq_table = load_bias(SAVE_DIR)
    else:
        bias_records, freq_table = run_bias_audit(caption_records, save_dir=SAVE_DIR)
    t4 = time.time()
    print(f"  ⏱  Step 4 complete in {t4-t3:.1f}s")

    # ─── Step 5: Mitigation ───────────────────────────────────────────────────
    _banner(5, "Toxicity Mitigation (BadWords Logit Penalty)", t0)
    from step5_mitigate import run_mitigation, _load_or_use_precomputed as load_mit
    if demo:
        mitigation_results = load_mit(SAVE_DIR)
    else:
        mitigation_results = run_mitigation(model, processor, device,
                                            caption_records, tox_scores,
                                            save_dir=SAVE_DIR)
    t5 = time.time()
    print(f"  ⏱  Step 5 complete in {t5-t4:.1f}s")

    # ─── Step 6: Visualize ────────────────────────────────────────────────────
    _banner(6, "Generate Fairness Visualizations", t0)
    from step6_visualize import visualize_all
    figure_paths = visualize_all(tox_scores, freq_table, mitigation_results, SAVE_DIR)
    t6 = time.time()
    print(f"  ⏱  Step 6 complete in {t6-t5:.1f}s")

    # ─── Step 7: Fairness report ──────────────────────────────────────────────
    _banner(7, "Generate Fairness Report", t0)
    from step7_fairness_report import generate_report
    report_path = generate_report(tox_scores, bias_records, freq_table,
                                  mitigation_results, save_dir=SAVE_DIR)
    t7 = time.time()
    print(f"  ⏱  Step 7 complete in {t7-t6:.1f}s")

    # ─── Pipeline summary ─────────────────────────────────────────────────────
    total_captions   = len(tox_scores)
    n_tox_flagged    = sum(1 for r in tox_scores if r["flagged"])
    n_bias_flagged   = sum(1 for r in bias_records if r["flagged"])
    n_mitigated      = sum(1 for r in mitigation_results if r["mitigated"])

    print(f"\n{'═'*68}")
    print(f"  TASK 5 PIPELINE — COMPLETE")
    print(f"{'═'*68}")
    print(f"  Total time        : {t7-t0:.1f}s")
    print(f"  Mode              : {'DEMO (pre-computed)' if demo else 'LIVE'}")
    print(f"  Results dir       : {SAVE_DIR}")
    print()
    print(f"  ☣️  Toxicity Analysis:")
    print(f"     Captions scored : {total_captions}")
    print(f"     Flagged         : {n_tox_flagged} ({100*n_tox_flagged/max(total_captions,1):.1f}%)")
    print()
    print(f"  🏥 Bias Audit:")
    print(f"     Captions with stereotype : {n_bias_flagged} ({100*n_bias_flagged/max(total_captions,1):.1f}%)")
    print()
    print(f"  🛡️  Mitigation:")
    print(f"     Tested / cleaned : {len(mitigation_results)} / {n_mitigated}")
    print()
    print(f"  📁 Output files:")
    print(f"     captions_1000.json       — 1000 generated captions")
    print(f"     toxicity_scores.json     — per-caption 6-label toxicity scores")
    print(f"     bias_audit.json          — stereotype flags + frequency table")
    print(f"     mitigation_results.json  — before/after caption pairs")
    print(f"     fairness_report.md       — full written report")
    for name, p in figure_paths.items():
        fname = os.path.basename(p)
        print(f"     {fname:40s}— {name} figure")
    print(f"{'═'*68}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 5 — Toxicity & Bias Pipeline")
    parser.add_argument("--demo", action="store_true",
                        help="Use pre-computed results (no GPU required)")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(TASK_DIR)))   # project root
    sys.path.insert(0, TASK_DIR)

    run_pipeline(demo=args.demo)
