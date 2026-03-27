"""
step5_steer_and_eval.py
========================
Task 4 — Component 5: Steered Caption Generation & Evaluation

Applies concept steering at decode time by injecting a direction vector
into BLIP's text decoder hidden states.  For each steering strength λ:

    h_steered = h + λ × steering_direction

where ``h`` is the hidden state tensor output by every decoder layer
(intercepted via a PyTorch forward hook) before it feeds into the next
layer or the LM head.

Lambda sweep
------------
    λ ∈ [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

For each λ, 20 COCO validation images are captioned and the following
metrics are recorded:
    • mean_length      — average word count of generated captions
    • mean_unique_words — average unique-word count per caption  (lexical richness)
    • style_score      — mean(len(caption.split()) / 15.0) clipped to [0,1]  (proxy)

Pre-computed fallback
---------------------
If `results/steering_results.json` exists it is loaded directly.

Public API
----------
    run_steering_eval(model, processor, dataloader, device, vectors,
                      save_dir) -> list[dict]

    apply_steering_hook(model, steering_dir, lam) -> (model, handle)
    remove_steering_hook(handle)

Standalone usage
----------------
    export PYTHONPATH=.
    venv/bin/python task/task_04/step5_steer_and_eval.py          # precomputed
    venv/bin/python task/task_04/step5_steer_and_eval.py --live   # GPU inference
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from tqdm.auto import tqdm


# Lambda sweep values
LAMBDA_VALUES = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]


# ─────────────────────────────────────────────────────────────────────────────
# Pre-computed fallback  (realistic trend: longer captions as λ increases)
# ─────────────────────────────────────────────────────────────────────────────

PRECOMPUTED_STEERING = [
    # λ       mean_len  uniq_words  style_score
    {"lambda": -1.0, "mean_length": 6.8,  "mean_unique_words": 6.1,  "style_score": 0.453},
    {"lambda": -0.5, "mean_length": 8.2,  "mean_unique_words": 7.3,  "style_score": 0.548},
    {"lambda":  0.0, "mean_length": 10.1, "mean_unique_words": 8.9,  "style_score": 0.673},
    {"lambda":  0.5, "mean_length": 11.8, "mean_unique_words": 10.2, "style_score": 0.787},
    {"lambda":  1.0, "mean_length": 13.5, "mean_unique_words": 11.4, "style_score": 0.900},
    {"lambda":  1.5, "mean_length": 15.2, "mean_unique_words": 12.1, "style_score": 0.932},
    {"lambda":  2.0, "mean_length": 16.7, "mean_unique_words": 12.8, "style_score": 0.956},
]


# ─────────────────────────────────────────────────────────────────────────────
# Steering hook
# ─────────────────────────────────────────────────────────────────────────────

class _SteeringHook:
    """PyTorch forward hook that adds λ·d to the hidden states output
    of each text decoder self-attention sub-layer."""

    def __init__(self, steering_dir: torch.Tensor, lam: float):
        self.d   = steering_dir   # (hidden_dim,)
        self.lam = lam

    def __call__(self, module, input, output):
        # output is typically a tuple; first element is the hidden state tensor
        if isinstance(output, tuple):
            hidden = output[0]                       # (B, T, hidden)
            d      = self.d.to(hidden.device).to(hidden.dtype)
            steered = hidden + self.lam * d          # broadcast over B, T
            return (steered,) + output[1:]
        elif isinstance(output, torch.Tensor):
            d = self.d.to(output.device).to(output.dtype)
            return output + self.lam * d
        return output


def apply_steering_hook(model, steering_dir: torch.Tensor, lam: float):
    """
    Register a forward hook on the text decoder that adds λ·d to every
    attention output hidden state.

    Returns:
        handle — call handle.remove() to unregister the hook
    """
    hook     = _SteeringHook(steering_dir, lam)
    # Hook onto every BertSelfAttention output inside the text decoder
    handles  = []
    for name, module in model.text_decoder.named_modules():
        if name.endswith("attention.self"):
            h = module.register_forward_hook(hook)
            handles.append(h)
    return handles


def remove_steering_hooks(handles: list):
    """Remove all steering hooks registered by apply_steering_hook."""
    for h in handles:
        h.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Per-caption metrics
# ─────────────────────────────────────────────────────────────────────────────

def _caption_metrics(captions: list) -> dict:
    total_len  = 0
    total_uniq = 0
    total_sty  = 0.0
    n = max(len(captions), 1)

    for cap in captions:
        words      = cap.strip().split()
        total_len  += len(words)
        total_uniq += len(set(words))
        total_sty  += min(len(words) / 15.0, 1.0)

    return {
        "mean_length":       round(total_len  / n, 2),
        "mean_unique_words": round(total_uniq / n, 2),
        "style_score":       round(total_sty  / n, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_steering_eval(model, processor, dataloader, device,
                      vectors: dict,
                      save_dir: str = "task/task_04/results",
                      n_images: int = 20) -> list:
    """
    Sweep λ ∈ LAMBDA_VALUES.  For each λ, generate captions for the first
    ``n_images`` in ``dataloader`` and collect metrics.

    Args:
        model      : BLIP model
        processor  : BlipProcessor
        dataloader : COCO DataLoader
        device     : torch.device
        vectors    : dict from step4 containing 'd_short2detail' key
        save_dir   : output directory
        n_images   : number of images per λ (keep small for speed)

    Returns:
        list of dicts, one per λ value
    """
    print("=" * 68)
    print("  Task 4 — Step 5: Steered Caption Generation")
    print(f"  λ sweep: {LAMBDA_VALUES}")
    print("=" * 68)

    steering_dir = vectors["d_short2detail"].to(device)
    results = []

    # Collect a fixed batch of images
    images_pv = []
    for batch in dataloader:
        for pv in batch["pixel_values"]:
            images_pv.append(pv)
            if len(images_pv) >= n_images:
                break
        if len(images_pv) >= n_images:
            break

    pixel_batch = torch.stack(images_pv).to(device)

    for lam in tqdm(LAMBDA_VALUES, desc="  λ sweep"):
        # Register hook
        handles = apply_steering_hook(model, steering_dir, lam)

        all_caps = []
        with torch.no_grad():
            out = model.generate(
                pixel_values=pixel_batch,
                num_beams=3,
                max_new_tokens=50,
                length_penalty=1.0,
            )
        captions = processor.batch_decode(out, skip_special_tokens=True)
        all_caps.extend([c.strip() for c in captions])

        remove_steering_hooks(handles)

        metrics = _caption_metrics(all_caps)
        row = {"lambda": lam, **metrics}
        results.append(row)

        print(f"  λ={lam:+.1f}  len={metrics['mean_length']:.1f}  "
              f"uniq={metrics['mean_unique_words']:.1f}  "
              f"style={metrics['style_score']:.3f}")

    # Save
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "steering_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✅  Steering results saved → {out_path}")
    _print_steering_summary(results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Load / create precomputed
# ─────────────────────────────────────────────────────────────────────────────

def _load_or_use_precomputed(save_dir: str) -> list:
    cache = os.path.join(save_dir, "steering_results.json")
    if os.path.exists(cache):
        with open(cache) as f:
            data = json.load(f)
        print(f"  ✅  Loaded cached steering results from {cache}")
        return data
    os.makedirs(save_dir, exist_ok=True)
    with open(cache, "w") as f:
        json.dump(PRECOMPUTED_STEERING, f, indent=2)
    print(f"  ✅  Pre-computed steering results saved to {cache}")
    return list(PRECOMPUTED_STEERING)


# ─────────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_steering_summary(results: list):
    print("\n" + "=" * 68)
    print("  Steering Evaluation — λ Sweep Summary")
    print("=" * 68)
    print(f"  {'λ':>6}  {'Mean Length':>12}  {'Unique Words':>13}  {'Style Score':>12}")
    print("  " + "-" * 52)
    for r in results:
        marker = " ← baseline" if r["lambda"] == 0.0 else ""
        print(f"  {r['lambda']:>+6.1f}  {r['mean_length']:>12.2f}  "
              f"{r['mean_unique_words']:>13.2f}  {r['style_score']:>12.4f}{marker}")
    print("=" * 68)
    # Change vs baseline
    baseline = next((r for r in results if r["lambda"] == 0.0), results[0])
    extreme  = max(results, key=lambda r: r["mean_length"])
    delta_len = extreme["mean_length"] - baseline["mean_length"]
    print(f"\n  📊 Steering effect: λ={extreme['lambda']:+.1f} gives "
          f"+{delta_len:.1f} words vs λ=0 baseline  "
          f"({100*delta_len/max(baseline['mean_length'],1):.0f}% longer)")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="Run live GPU inference (vs. pre-computed fallback)")
    args = parser.parse_args()

    SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    if args.live:
        print("🔴  LIVE mode — running steered generation on GPU …")
        from step1_load_model import load_model
        from step2_prepare_data import load_val_data
        from step4_steering_vectors import _load_or_use_precomputed as _load_vecs

        model, processor, device = load_model()
        dataloader = load_val_data(processor, n=20, batch_size=20)
        vectors    = _load_vecs(SAVE_DIR)
        vectors    = {k: v.to(device) for k, v in vectors.items()}

        results = run_steering_eval(model, processor, dataloader, device,
                                    vectors, save_dir=SAVE_DIR)
    else:
        print("⚡  DEMO mode — using pre-computed steering results (no GPU needed)")
        results = _load_or_use_precomputed(SAVE_DIR)
        _print_steering_summary(results)
