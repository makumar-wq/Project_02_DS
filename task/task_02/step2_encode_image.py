"""
step2_encode_image.py
======================
STEP 2 — Encode an image through BLIP's Vision Transformer (ViT).

Responsibilities:
- Accept a PIL image.
- Run it through the ViT image encoder.
- Return encoder_hidden_states (197 patch tokens × 768 dim).
- Also return the encoder_attention_mask.

This is kept separate so the expensive ViT encode is run ONCE for
however many words we later generate — zero redundant computation.

Shape of the output:
  encoder_hidden_states : (1, 197, 768)
  encoder_mask          : (1, 197)   — all-ones (no padding)
"""

import os
import sys
import torch
from PIL import Image

_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ────────────────────────────────────────────────────────────────────────────
_ENCODE_SIZE = 224     # ViT expects 224×224 → 14×14 patches of size 16


def encode_image(model, processor, device, image_pil: Image.Image, verbose: bool = True):
    """
    Encode a PIL image through BLIP's ViT backbone.

    Args:
        model     – BlipForConditionalGeneration.
        processor – BlipProcessor.
        device    – torch.device.
        image_pil – Any PIL image (will be resized to 224×224 internally).
        verbose   – Print progress.

    Returns:
        image_224         – PIL image resized to 224×224.
        encoder_hidden    – Tensor (1, 197, 768), detached, no grad.
        encoder_mask      – Tensor (1, 197), all-ones.
    """
    image_224 = image_pil.resize((_ENCODE_SIZE, _ENCODE_SIZE), Image.LANCZOS)
    inputs = processor(images=image_224, return_tensors="pt").to(device)

    if verbose:
        print(f"📷 Image resized to {_ENCODE_SIZE}×{_ENCODE_SIZE} and encoded through ViT …")

    with torch.no_grad():
        vision_out = model.vision_model(pixel_values=inputs["pixel_values"])

    # Shape: (1, 197, 768) — 1 [CLS] token + 196 patch tokens
    encoder_hidden = vision_out[0].detach().requires_grad_(False)
    encoder_mask   = torch.ones(encoder_hidden.size()[:-1], dtype=torch.long, device=device)

    if verbose:
        print(f"✅ Encoder output shape: {encoder_hidden.shape}  "
              f"(1 CLS + {encoder_hidden.shape[1]-1} patches)")

    return image_224, encoder_hidden, encoder_mask
