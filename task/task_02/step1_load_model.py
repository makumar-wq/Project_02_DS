"""
step1_load_model.py
====================
STEP 1 — Load the BLIP model and processor.

Responsibilities:
- Detect the best available device (MPS / CUDA / CPU).
- Load base BLIP weights via project's get_blip_model().
- Optionally patch in fine-tuned weights from outputs/blip/best/.
- Disable gradient checkpointing (required for backward hooks).
- Return a ready-to-use (model, processor, device) triplet.

This module is intentionally tiny and self-contained so it can be
called independently from a notebook, a Streamlit app, or a HuggingFace Space.
"""

import os
import sys
import torch

# ── project path resolution ─────────────────────────────────────────────────
_THIS_DIR       = os.path.dirname(os.path.abspath(__file__))
_TASK_DIR       = os.path.dirname(_THIS_DIR)
_PROJECT_ROOT   = os.path.dirname(_TASK_DIR)

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import CFG
from models.blip_tuner import get_blip_model

FINETUNED_PATH = os.path.join(_PROJECT_ROOT, "outputs", "blip", "best")


# ────────────────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    """Return the best available compute device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(use_finetuned: bool = True, verbose: bool = True):
    """
    Load the BLIP model and processor.

    Args:
        use_finetuned: If True, patches base weights with our fine-tuned
                       COCO checkpoint from outputs/blip/best/.
        verbose:       Print loading progress.

    Returns:
        model     – BlipForConditionalGeneration, eval mode.
        processor – BlipProcessor.
        device    – torch.device.
    """
    from transformers import BlipForConditionalGeneration

    device = get_device()
    cfg = CFG.load_for_model("blip")
    model, processor = get_blip_model(cfg, device)

    # Optionally load fine-tuned weights
    if use_finetuned and os.path.isdir(FINETUNED_PATH):
        if verbose:
            print(f"🔄 Loading fine-tuned weights from {FINETUNED_PATH} …")
        ft = BlipForConditionalGeneration.from_pretrained(FINETUNED_PATH)
        model.load_state_dict(ft.state_dict(), strict=False)
        model.to(device)
        if verbose:
            print("✅ Fine-tuned weights loaded")
    else:
        if verbose:
            print("⚠️  Fine-tuned weights not found (or disabled), using base model")

    # Disable gradient checkpointing — incompatible with backward hooks
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass
    model.config.use_cache = False
    model.eval()

    if verbose:
        print(f"✅ Model ready on device: {device}")

    return model, processor, device
