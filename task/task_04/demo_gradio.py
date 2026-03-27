"""
demo_gradio.py
==============
Task 4 — Interactive User-Upload Demo

Launches a Gradio Blocks interface where users can upload any image and
immediately see:
  1. Five nucleus-sampled captions (diversity analysis, top_p=0.9)
  2. Their diversity score with an interpretation badge
  3. Four steered captions (lambda = -1.0, 0.0, +1.0, +2.0), showing
     how hidden-state steering shifts caption style from terse to detailed.

Architecture
------------
- Model is loaded ONCE at module level (cached) to avoid re-loading on
  every request — critical for HuggingFace Spaces latency.
- The steering vectors are loaded from results/steering_vectors.pt if
  available, else the precomputed fallback is used.
- Inference runs on CPU (no GPU required), making this suitable for
  HuggingFace Spaces free tier.
- Reuses existing step3 and step5 functions — no new logic, just UI.

Standalone usage
----------------
    export PYTHONPATH=.
    venv/bin/python task/task_04/demo_gradio.py
    # -> Visit http://localhost:7860

HuggingFace Spaces usage
-------------------------
    # In app.py, add:
    #   gr.TabbedInterface([main_demo, task4_demo], ["Main Demo", "Task 4"])
    # Or run this file standalone as the Space's entry point.

Requirements
------------
    pip install gradio>=3.0
    (already in requirements.txt if present, else: pip install gradio)
"""

import os
import sys
import time

_TASK_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_TASK_DIR))
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, _TASK_DIR)

RESULTS_DIR = os.path.join(_TASK_DIR, "results")

# ─────────────────────────────────────────────────────────────────────────────
# Module-level model cache (loaded once)
# ─────────────────────────────────────────────────────────────────────────────

_MODEL     = None
_PROCESSOR = None
_DEVICE    = None
_VECTORS   = None   # dict with "d_short2detail"


def _get_model():
    """Load (or return cached) BLIP model, processor, device."""
    global _MODEL, _PROCESSOR, _DEVICE
    if _MODEL is None:
        from step1_load_model import load_model
        _MODEL, _PROCESSOR, _DEVICE = load_model()
        _MODEL.eval()
        print("  [demo_gradio] Model loaded and cached.")
    return _MODEL, _PROCESSOR, _DEVICE


def _get_vectors():
    """Load (or return cached) steering vectors."""
    global _VECTORS
    if _VECTORS is None:
        from step4_steering_vectors import _load_or_use_precomputed
        _VECTORS = _load_or_use_precomputed(RESULTS_DIR)
        print("  [demo_gradio] Steering vectors loaded.")
    return _VECTORS


# ─────────────────────────────────────────────────────────────────────────────
# Core inference functions
# ─────────────────────────────────────────────────────────────────────────────

def _generate_diversity_captions(pil_image, n=5, top_p=0.9):
    """
    Generate n nucleus-sampled captions and compute diversity score.
    Returns (captions: list[str], diversity_score: float).
    """
    model, processor, device = _get_model()
    from step3_diversity_analysis import compute_diversity_score

    inputs = processor(images=pil_image.convert("RGB"), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    import torch
    captions = []
    with torch.no_grad():
        for _ in range(n):
            out = model.generate(
                pixel_values=pixel_values,
                do_sample=True,
                top_p=top_p,
                max_new_tokens=40,
                num_beams=1,
            )
            cap = processor.batch_decode(out, skip_special_tokens=True)[0]
            captions.append(cap.strip())

    score = compute_diversity_score(captions)
    return captions, score


def _generate_steered_captions(pil_image, lambdas=(-1.0, 0.0, 1.0, 2.0)):
    """
    Generate one beam=3 caption per lambda value via hidden-state steering.
    Returns dict {lambda -> caption string}.
    """
    model, processor, device = _get_model()
    vectors = _get_vectors()
    from step5_steer_and_eval import apply_steering_hook, remove_steering_hooks

    import torch
    inputs = processor(images=pil_image.convert("RGB"), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    steering_dir = vectors["d_short2detail"].to(device)
    results = {}

    with torch.no_grad():
        for lam in lambdas:
            handles = apply_steering_hook(model, steering_dir, lam)
            out = model.generate(
                pixel_values=pixel_values,
                num_beams=3,
                max_new_tokens=50,
                length_penalty=1.0,
            )
            caption = processor.batch_decode(out, skip_special_tokens=True)[0]
            remove_steering_hooks(handles)
            results[lam] = caption.strip()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Gradio inference handler
# ─────────────────────────────────────────────────────────────────────────────

LAMBDAS_DEMO = (-1.0, 0.0, 1.0, 2.0)


def run_inference(pil_image):
    """
    Main Gradio handler.

    Args:
        pil_image: PIL.Image from the upload widget

    Returns:
        Tuple of outputs matching the Gradio output components.
    """
    if pil_image is None:
        return (
            "Please upload an image.",
            "",
            "",
        )

    t0 = time.time()

    # Part 1: Diversity
    captions, score = _generate_diversity_captions(pil_image, n=5, top_p=0.9)

    if score > 0.75:
        badge = "HIGH diversity — model explores many different descriptions"
        badge_style = "diverse"
    elif score < 0.40:
        badge = "LOW diversity — model is repetitive for this image"
        badge_style = "repetitive"
    else:
        badge = "MEDIUM diversity"
        badge_style = "medium"

    diversity_text = (
        f"**Diversity Score: {score:.4f}**  —  {badge}\n\n"
        f"*(5 nucleus-sampled captions, top_p=0.9, scores = "
        f"unique n-grams / total n-grams)*\n\n"
    )
    for i, cap in enumerate(captions, 1):
        diversity_text += f"{i}. {cap}\n"

    # Part 2: Steering
    steered = _generate_steered_captions(pil_image, lambdas=LAMBDAS_DEMO)

    steering_text = (
        "**Style Steering Results**  `h = h + lambda x d_short2detail`\n\n"
        "| lambda | Caption |\n|---|---|\n"
    )
    for lam in LAMBDAS_DEMO:
        note = " *(baseline)*" if lam == 0.0 else ""
        steering_text += f"| `{lam:+.1f}` | {steered[lam]}{note} |\n"

    elapsed = time.time() - t0
    timing_text = f"*Inference time: {elapsed:.1f}s on {_DEVICE}*"

    return diversity_text, steering_text, timing_text


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

def build_demo():
    """Build and return the Gradio Blocks demo."""
    try:
        import gradio as gr
    except ImportError:
        raise ImportError(
            "Gradio is not installed. Run: pip install gradio>=3.0"
        )

    with gr.Blocks(
        title="Task 4: Caption Diversity & Style Steering",
        theme=gr.themes.Soft(),
        css="""
        .score-badge { font-size: 1.1em; font-weight: bold; }
        .output-panel { border-radius: 8px; }
        """,
    ) as demo:

        gr.Markdown(
            """
# Task 4 — Caption Diversity & Concept Style Steering

Upload any image to explore two aspects of BLIP's captioning behaviour:

1. **Diversity Analysis** — How varied are the 5 nucleus-sampled captions?
   A high score means the model generates genuinely different descriptions
   each time; a low score means it's stuck in a repetitive pattern.

2. **Style Steering** — Watch how caption length and detail changes when we
   inject a "steering vector" `d_short2detail` into BLIP's text decoder
   at every attention layer. No retraining — just vector arithmetic.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Upload Image",
                    height=320,
                    elem_id="upload_image",
                )
                submit_btn = gr.Button(
                    "Analyse Caption Diversity & Style",
                    variant="primary",
                    elem_id="analyse_btn",
                )
                gr.Markdown(
                    "*Note: Inference runs on CPU. "
                    "Expect 5–15 seconds for 9 total captions.*"
                )

            with gr.Column(scale=2):
                out_diversity = gr.Markdown(
                    label="Diversity Analysis",
                    value="*Upload an image and click Analyse to see results.*",
                    elem_id="diversity_output",
                )
                out_steering  = gr.Markdown(
                    label="Style Steering",
                    value="",
                    elem_id="steering_output",
                )
                out_timing    = gr.Markdown(
                    value="",
                    elem_id="timing_output",
                )

        submit_btn.click(
            fn=run_inference,
            inputs=[image_input],
            outputs=[out_diversity, out_steering, out_timing],
        )

        gr.Markdown(
            """
---
### How it works

**Diversity score** = unique n-grams (unigrams + bigrams) / total n-grams,
computed across 5 nucleus-sampled captions (top_p=0.9).

**Steering vector** `d = normalise(mu_detailed - mu_short)` is extracted
from the mean hidden states of BLIP's text encoder over style-labelled COCO
captions. During generation, every decoder attention layer receives:
`h_steered = h + lambda x d`.

| lambda | Effect |
|--------|--------|
| -1.0 | Push toward shorter / more terse |
| 0.0  | Unsteered baseline (beam search) |
| +1.0 | Moderate detail increase (~+3 words) |
| +2.0 | Strong detail increase (~+7 words) |

**Optimal range**: lambda in [0.5, 1.0] for fluent style shift
without degrading COCO metric performance.
            """
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading model at startup (cached for all requests)...")
    _get_model()
    _get_vectors()
    print("Model ready. Launching Gradio interface...")

    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",   # bind to all interfaces (needed for HF Spaces)
        server_port=7860,
        share=False,
        show_error=True,
    )
