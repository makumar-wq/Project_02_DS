"""
pipeline.py
============
Master Orchestrator — Task 2, Iteration 3

This script chains all five steps together in the correct order,
printing a clear progress banner at each stage so you can see exactly
what is happening and inspect intermediate results.

Step-by-step flow
------------------
  STEP 1 → Load BLIP model (with fine-tuned weights if available).
  STEP 2 → Encode image through ViT  →  encoder_hidden_states.
  STEP 3 → Greedy decode token-by-token with Attention Flow heatmaps
            (multi-layer GradCAM rollout, bicubic upscaling).
  STEP 4 → Build 2×5 overlay grid image  →  attention_grid_v3.png.
  STEP 5 → Grade alignment with OWL-ViT + IoU  →  iou_chart_v3.png.

Designed to be deployment-friendly:
  • Every step is a clean function import from its own module.
  • Intermediate artefacts (heatmaps, tokens) can be inspected between steps.
  • Outputs are saved to the same directory as this script.

Usage:
    export PYTHONPATH=.
    venv/bin/python task/task_02/pipeline.py
"""

import os
import sys
import requests
from PIL import Image

# ── path bootstrap ────────────────────────────────────────────────────────────
_THIS_DIR       = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT   = os.path.dirname(os.path.dirname(_THIS_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── step imports ─────────────────────────────────────────────────────────────
from task.task_02.step1_load_model  import load_model
from task.task_02.step2_encode_image import encode_image
from task.task_02.step3_gradcam_flow import generate_with_flow
from task.task_02.step4_visualize    import save_attention_grid
from task.task_02.step5_iou_grade    import load_detector, grade_alignment, plot_iou_chart


# ── Output paths ─────────────────────────────────────────────────────────────
OUT_GRID  = os.path.join(_THIS_DIR, "attention_grid_v3.png")
OUT_CHART = os.path.join(_THIS_DIR, "iou_chart_v3.png")

# ── Test images ───────────────────────────────────────────────────────────────
TEST_URLS = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",  # cats on couch
    "http://images.cocodataset.org/val2017/000000000139.jpg",  # dining room
]


def _load_image(url: str) -> Image.Image:
    """Download an image from url, return PIL RGB image."""
    print(f"\n📥 Downloading test image: {url}")
    return Image.open(requests.get(url, stream=True).raw).convert("RGB")


def _banner(step: int, title: str):
    print(f"\n{'='*60}")
    print(f"  STEP {step} — {title}")
    print(f"{'='*60}")


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_pipeline():
    # ── STEP 1: Load model ───────────────────────────────────────────────────
    _banner(1, "Load BLIP Model")
    model, processor, device = load_model(use_finetuned=True)

    # ── Load OWL-ViT grader (do once, reuse for all images) ─────────────────
    detector = load_detector(device)

    # Aggregate IoU results across images for the final chart
    all_iou_results = []

    for img_url in TEST_URLS:
        raw_image = _load_image(img_url)

        # ── STEP 2: Encode image ─────────────────────────────────────────────
        _banner(2, "Encode Image through ViT")
        image_224, enc_hidden, enc_mask = encode_image(model, processor, device, raw_image)

        # ── STEP 3: Generate caption + Attention Flow heatmaps ───────────────
        _banner(3, "Greedy Decode with Attention Flow")
        tokens, heatmaps = generate_with_flow(
            model, processor, device, enc_hidden, enc_mask
        )

        # ── INSPECT intermediate results ─────────────────────────────────────
        print(f"\n  📝 Tokens   : {tokens}")
        print(f"  🗺  Heatmaps : {len(heatmaps)} maps, each shape {heatmaps[0].shape if heatmaps else 'N/A'}")
        print(f"     Peak values: {[f'{h.max():.3f}' for h in heatmaps[:5]]} …")

        # ── STEP 4: Visualize (only for the first image to save space) ───────
        if img_url == TEST_URLS[0]:
            _banner(4, "Build Attention Grid Visualization")
            save_attention_grid(image_224, tokens, heatmaps, out_path=OUT_GRID)

        # ── STEP 5: Grade alignment ──────────────────────────────────────────
        _banner(5, "Grade Attention Alignment (IoU)")
        results = grade_alignment(raw_image, tokens, heatmaps, detector)
        all_iou_results.extend(results)

    # ── Save IoU chart (all images combined) ─────────────────────────────────
    if all_iou_results:
        print(f"\n📈 Saving IoU chart for {len(all_iou_results)} data points …")
        plot_iou_chart(all_iou_results, out_path=OUT_CHART)

    print("\n" + "="*60)
    print("  ✅  PIPELINE COMPLETE")
    print(f"     Attention grid  → {OUT_GRID}")
    print(f"     IoU chart       → {OUT_CHART}")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_pipeline()
