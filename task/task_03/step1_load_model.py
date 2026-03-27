"""
step1_load_model.py
====================
Task 3 — Component 1: Load BLIP model with fine-tuned weights.

This module loads the BLIP image-captioning model and attempts to restore
the best fine-tuned checkpoint from `outputs/blip/best/`. If no checkpoint
is found it falls back gracefully to the pretrained HuggingFace weights.

Public API
----------
    load_model(weights_dir="outputs/blip/best") -> (model, processor, device)

Standalone usage
----------------
    export PYTHONPATH=.
    venv/bin/python task/task_03/step1_load_model.py
"""

import os
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor


# ─────────────────────────────────────────────────────────────────────────────
# Device helper
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return the best available device: MPS → CUDA → CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Main loader
# ─────────────────────────────────────────────────────────────────────────────

BLIP_BASE_ID = "Salesforce/blip-image-captioning-base"


def load_model(weights_dir: str = "outputs/blip/best"):
    """
    Load BLIP for conditional generation.

    1. Downloads/caches base weights from HuggingFace (first run only).
    2. Loads fine-tuned checkpoint from `weights_dir` if it exists.

    Args:
        weights_dir: Path to a directory containing a BLIP checkpoint saved
                     by `train.py` (e.g. ``outputs/blip/best``).  Can be
                     relative to the *project root*.

    Returns:
        (model, processor, device)
            model     : BlipForConditionalGeneration (eval mode)
            processor : BlipProcessor
            device    : torch.device
    """
    device = get_device()
    print("=" * 60)
    print("  Task 3 — Step 1: Load BLIP Model")
    print("=" * 60)
    print(f"  Device  : {device}")

    # ── Load processor ────────────────────────────────────────────────────────
    processor = BlipProcessor.from_pretrained(BLIP_BASE_ID)
    print(f"  ✅ Processor loaded  ({BLIP_BASE_ID})")

    # ── Try fine-tuned checkpoint first ───────────────────────────────────────
    abs_weights = os.path.abspath(weights_dir)
    if os.path.isdir(abs_weights) and os.listdir(abs_weights):
        print(f"  Loading fine-tuned weights from: {abs_weights}")
        model = BlipForConditionalGeneration.from_pretrained(abs_weights)
        print("  ✅ Fine-tuned checkpoint loaded")
        weights_source = f"fine-tuned ({weights_dir})"
    else:
        print(f"  ⚠️  No checkpoint at {abs_weights}. Using base HuggingFace weights.")
        model = BlipForConditionalGeneration.from_pretrained(BLIP_BASE_ID)
        print("  ✅ Base pretrained weights loaded")
        weights_source = "base (pretrained)"

    model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}  |  Weights: {weights_source}")
    print("=" * 60)

    return model, processor, device


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    # Allow running from the task folder directly
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    model, processor, device = load_model()
    print(f"\n✅  load_model() returned successfully.")
    print(f"   model type : {type(model).__name__}")
    print(f"   device     : {device}")
    print(f"\nYou can now import this in any notebook:")
    print("  from task.task_03.step1_load_model import load_model")
    print("  model, processor, device = load_model()")
