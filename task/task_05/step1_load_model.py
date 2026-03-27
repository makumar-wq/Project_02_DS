"""
step1_load_model.py
====================
Task 5 — Component 1: Load BLIP model + toxicity classifier.

Loads:
  - BLIP (Salesforce/blip-image-captioning-base) with fine-tuned checkpoint
    fallback to the base HuggingFace weights.
  - unitary/toxic-bert for toxicity scoring (reusing the exact same model
    used in app.py — single source of truth).

Public API
----------
    load_model(checkpoint_dir=None) -> (model, processor, device)
    load_toxicity_model()           -> (tokenizer, model)

Standalone usage
----------------
    export PYTHONPATH=.
    venv/bin/python task/task_05/step1_load_model.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ─────────────────────────────────────────────────────────────────────────────
# BLIP caption model
# ─────────────────────────────────────────────────────────────────────────────

_BLIP_BASE   = "Salesforce/blip-image-captioning-base"
_CKPT_SEARCH = [
    "outputs/blip/best",
    "outputs/blip/latest",
]


def load_model(checkpoint_dir: str = None):
    """
    Load BLIP for caption generation.

    Args:
        checkpoint_dir: path to fine-tuned checkpoint directory.
                        If None, searches _CKPT_SEARCH; falls back to base.

    Returns:
        (model, processor, device)
    """
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"  [step1] Device: {device}")

    processor = BlipProcessor.from_pretrained(_BLIP_BASE, use_fast=True)
    model     = BlipForConditionalGeneration.from_pretrained(_BLIP_BASE)

    # Try to load fine-tuned weights
    ckpt = checkpoint_dir
    if ckpt is None:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        for rel in _CKPT_SEARCH:
            cand = os.path.join(root, rel)
            if os.path.isdir(cand) and os.listdir(cand):
                ckpt = cand
                break

    if ckpt and os.path.isdir(ckpt):
        try:
            loaded = BlipForConditionalGeneration.from_pretrained(ckpt)
            model.load_state_dict(loaded.state_dict())
            del loaded
            print(f"  [step1] Loaded fine-tuned weights from: {ckpt}")
        except Exception as e:
            print(f"  [step1] Warning: could not load checkpoint ({e}). Using base weights.")
    else:
        print(f"  [step1] Using base HuggingFace weights ({_BLIP_BASE})")

    model.to(device).eval()
    return model, processor, device


# ─────────────────────────────────────────────────────────────────────────────
# Toxicity classifier  (reuses app.py's load_toxicity_filter pattern)
# ─────────────────────────────────────────────────────────────────────────────

_TOX_MODEL_ID = "unitary/toxic-bert"


def load_toxicity_model():
    """
    Load unitary/toxic-bert for multi-label toxicity scoring.

    Returns:
        (tokenizer, model)   [both on CPU; eval mode]

    Note: This is the same model used in app.py's load_toxicity_filter().
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print(f"  [step1] Loading toxicity model: {_TOX_MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(_TOX_MODEL_ID)
    mdl = AutoModelForSequenceClassification.from_pretrained(_TOX_MODEL_ID)
    mdl.eval()
    return tok, mdl


# ─────────────────────────────────────────────────────────────────────────────
# Standalone
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Task 5 — Step 1: Load Models")
    print("=" * 60)

    model, processor, device = load_model()
    print(f"  BLIP loaded on {device}")

    tox_tok, tox_mdl = load_toxicity_model()
    labels = tox_mdl.config.id2label
    print(f"  Toxicity labels: {list(labels.values())}")
    print("  Step 1 OK.")
