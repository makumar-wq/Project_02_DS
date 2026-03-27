"""
step3_gradcam_flow.py
======================
STEP 3 — Multi-Layer Gradient-Weighted Attention Flow.

What this implements (Iteration 3 upgrade over single-layer GradCAM):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
In Iteration 2, we only looked at the last cross-attention layer.
Here, we hook into ALL cross-attention layers simultaneously, run
GradCAM on each one, and then combine them using "Attention Flow"
(Abnar & Zuidema, 2020):

  rollout[0] = gradcam(layer 0)
  rollout[i] = normalize(rollout[i-1]) × gradcam(layer i)
             + 0.5 × I               ← residual identity keeps a
                                       "floor" of global context so
                                       the rollout never collapses to
                                       zero for deep layers.

This multi-layer aggregation captures:
  • Early layers  → edges, textures, spatial structure.
  • Middle layers → part-level features (ears, wheels, legs).
  • Last layer    → high-level semantic concepts.

The combined result is a 14×14 heatmap with dramatically tighter
object contours and less background bleed.

Additionally, instead of plain OpenCV nearest-neighbour upsampling,
we use PyTorch bicubic interpolation to upscale 14×14 → 224×224,
producing pixel-smooth, non-blocky heatmap edges.
"""

import os
import sys
import math
import torch
import torch.nn.functional as F
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ────────────────────────────────────────────────────────────────────────────
class FlowExtractor:
    """
    Registers forward hooks on every cross-attention layer of BLIP's
    text decoder.  Stores per-layer attention_probs AND their gradients
    so we can compute per-layer GradCAM and then roll them up.
    """

    def __init__(self, model):
        self.model   = model
        self._hooks  = []
        self.layers  = []               # list of (fwd_map, grad_map) per layer, in order

        # Collect ALL cross-attention layers in order (layer 0 … 11)
        for layer in model.text_decoder.bert.encoder.layer:
            if hasattr(layer, "crossattention"):
                holder = {"fwd": None, "grad": None}
                self.layers.append(holder)

                # Closure to avoid late-binding bug
                def _make_hooks(h):
                    def _fwd(module, inp, out):
                        if len(out) > 1 and out[1] is not None:
                            h["fwd"] = out[1]
                            if h["fwd"].requires_grad:
                                h["fwd"].register_hook(lambda g, _h=h: _h.update({"grad": g.detach()}))
                    return _fwd

                target = layer.crossattention.self
                self._hooks.append(target.register_forward_hook(_make_hooks(holder)))

    def clear(self):
        for h in self.layers:
            h["fwd"] = None
            h["grad"] = None

    def remove(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []


def _single_layer_gradcam(holder, token_idx: int = -1) -> torch.Tensor:
    """
    Compute GradCAM for one layer.

    holder  : dict with 'fwd' and 'grad' tensors, shape (1, heads, seq, 197)
    Returns : 1D tensor of length 197, values ≥ 0.
    """
    attn = holder["fwd"][:, :, token_idx, :]      # (1, heads, 197)
    grad = holder["grad"][:, :, token_idx, :]      # (1, heads, 197)
    cam  = (attn * grad).mean(dim=1).squeeze()     # (197,)
    return torch.clamp(cam, min=0.0)


def _normalize1d(t: torch.Tensor) -> torch.Tensor:
    """L1-normalize a 1D tensor so it sums to 1."""
    s = t.sum()
    if s > 0:
        return t / s
    return t


def compute_attention_flow(
    extractor: FlowExtractor,
    num_image_tokens: int = 197,
    residual_weight: float = 0.05,
    out_resolution: int = 224,
) -> np.ndarray:
    """
    Compute Attention Flow across all layers.

    Algorithm
    ---------
    1.  For each layer i that has valid fwd AND grad maps:
        - Compute single-layer GradCAM (shape 197,).
        - Apply ReLU.
    2.  Roll up using recursive multiplication with residual identity:
        rollout[0] = cam[0]
        rollout[i] = norm(rollout[i-1]) ⊙ norm(cam[i])
                     + residual_weight × uniform

    ⚠️  Why residual_weight MUST be small (0.05, not 0.5):
        The uniform term provides a "background floor" so the rollout
        never collapses to zero.  BUT if it is too large, the uniform
        baseline dominates after 12 layers and ALL words produce the
        same flat heatmap (everything lights up equally, losing
        word-specificity).  0.05 keeps a tiny floor while letting the
        gradient signal dominate.

    3.  Drop the [CLS] token → 196 spatial patches → 14×14.
    4.  Bicubic upsample to (out_resolution × out_resolution) for
        pixel-smooth overlay.
    5.  Min-max normalise to [0, 1].

    Args:
        extractor       : FlowExtractor after a backward pass.
        num_image_tokens: Total image tokens (197 for BLIP ViT-base).
        residual_weight : Small regularisation weight (keep ≤ 0.1).
        out_resolution  : Target spatial size for the output heatmap.

    Returns:
        heatmap_np : (out_resolution, out_resolution) numpy float32 array.
    """
    valid_cams = []

    for holder in extractor.layers:
        if holder["fwd"] is None or holder["grad"] is None:
            continue
        cam = _single_layer_gradcam(holder)      # (197,)
        valid_cams.append(cam.detach())

    if not valid_cams:
        # Fallback: uniform heatmap
        grid = int(math.sqrt(num_image_tokens - 1))
        return np.zeros((out_resolution, out_resolution), dtype=np.float32)

    # --- Attention Flow rollout ---
    # Uniform baseline (Abnar & Zuidema eq. 6 identity term)
    uniform = torch.ones(num_image_tokens, device=valid_cams[0].device) / num_image_tokens

    rollout = _normalize1d(valid_cams[0])
    for cam in valid_cams[1:]:
        rollout = _normalize1d(rollout) * _normalize1d(cam) + residual_weight * uniform
        rollout = torch.clamp(rollout, min=0.0)

    # Drop [CLS] token (index 0) → 196 patch tokens
    spatial = rollout[1:]                   # (196,)
    grid_sz = int(math.sqrt(spatial.numel()))

    # Reshape → (1, 1, 14, 14) for F.interpolate
    hm_tensor = spatial.detach().cpu().reshape(1, 1, grid_sz, grid_sz).float()

    # Bicubic upsampling → (1, 1, out_res, out_res)
    hm_up = F.interpolate(
        hm_tensor,
        size=(out_resolution, out_resolution),
        mode="bicubic",
        align_corners=False,
    ).squeeze()                             # (out_res, out_res)

    hm_np = hm_up.numpy()

    # Min-max normalise to [0, 1]
    lo, hi = hm_np.min(), hm_np.max()
    if hi > lo:
        hm_np = (hm_np - lo) / (hi - lo)
    else:
        hm_np = np.zeros_like(hm_np)

    return hm_np.astype(np.float32)


# ── Main decoding loop ────────────────────────────────────────────────────────
def generate_with_flow(
    model,
    processor,
    device,
    encoder_hidden,
    encoder_mask,
    max_tokens: int = 20,
    verbose: bool = True,
) -> tuple[list[str], list[np.ndarray]]:
    """
    Token-by-token greedy decode with full Attention Flow heatmap per step.

    Args:
        model           – BlipForConditionalGeneration (eval, grad-ckpt disabled).
        processor       – BlipProcessor.
        device          – torch.device.
        encoder_hidden  – (1, 197, 768) from step2_encode_image.
        encoder_mask    – (1, 197) all-ones mask.
        max_tokens      – Maximum tokens to generate.
        verbose         – Print per-token progress.

    Returns:
        tokens   – List of decoded word strings.
        heatmaps – Parallel list of (224, 224) numpy heatmaps.
    """
    extractor = FlowExtractor(model)

    input_ids = torch.LongTensor([[model.config.text_config.bos_token_id]]).to(device)
    tokens    = []
    heatmaps  = []

    if verbose:
        print("🔄 Generating caption with Attention Flow heatmaps …")

    for step in range(max_tokens):
        model.zero_grad()
        extractor.clear()

        outputs = model.text_decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=encoder_mask,
            output_attentions=True,
            return_dict=True,
        )

        logits     = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1)

        if next_token.item() == model.config.text_config.sep_token_id:
            break

        # Backward from the chosen token's logit
        logits[0, next_token.item()].backward(retain_graph=False)

        # Compute Attention Flow across all layers
        hm = compute_attention_flow(extractor)
        heatmaps.append(hm)

        word = processor.tokenizer.decode([next_token.item()]).strip()
        tokens.append(word)

        if verbose:
            print(f"  [{step+1:02d}] '{word}'  heatmap peak={hm.max():.3f}")

        input_ids = torch.cat([input_ids, next_token.reshape(1, 1)], dim=-1)

    extractor.remove()

    if verbose:
        caption = " ".join(tokens)
        print(f"\n✅ Caption: {caption}")

    return tokens, heatmaps
