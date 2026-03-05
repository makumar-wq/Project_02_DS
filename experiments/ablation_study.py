"""
experiments/ablation_study.py
==============================
Cross-Attention Masking Ablation Study for BLIP and ViT-GPT2.

Four encoder_attention_mask ablation modes:

  Mode 1 — Baseline (Full Attention)
    Mask  : all 1s → text decoder sees all 197 patches (1 CLS + 196 spatial)
    Intent: Upper-bound reference; no information is hidden.

  Mode 2 — Random Patch Dropout (Sparse Attention)
    Mask  : 50% of 196 spatial patches randomly zeroed; CLS always kept at idx 0
    Intent: Tests redundancy — how much spatial information is truly needed?

  Mode 3 — Center-Focus Spatial Cropping
    Mask  : Only the inner 8×8 grid of the 14×14 spatial patch grid kept
    Intent: Tests whether the image periphery (background clutter) hurts captions.

  Mode 4 — "The Squint" (Global Pooling Proxy)
    Mask  : 196 spatial patches averaged → 1 token appended after CLS
            The mask then has shape (1, 2): [CLS=1, global_pool=1]
    Intent: Tests whether granular local patch details are necessary, or a
            global compressed summary suffices.

Note: GIT does not support encoder_attention_mask (no cross-attention).
      GIT ablations are noted as N/A in the results table.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm.auto import tqdm
from pycocoevalcap.cider.cider import Cider
from models.blip_tuner import generate_with_mask


# ─────────────────────────────────────────────────────────────────────────────
# Available Modes
# ─────────────────────────────────────────────────────────────────────────────

ABLATION_MODES = ["baseline", "random_dropout", "center_focus", "squint"]


# ─────────────────────────────────────────────────────────────────────────────
# Ablation Mask Builders
# ─────────────────────────────────────────────────────────────────────────────

def build_ablation_mask(mode: str, batch_size: int, num_patches: int,
                        device: torch.device, cfg=None):
    """
    Build an encoder_attention_mask tensor for a given ablation mode.

    Args:
        mode        : 'baseline' | 'random_dropout' | 'center_focus' | 'squint'
        batch_size  : number of images in the batch
        num_patches : total patches including CLS (usually 197 = 1 + 196)
        device      : target torch device
        cfg         : config object for dropout_ratio (default 0.5 if None)

    Returns:
        mask : LongTensor of shape (batch_size, num_patches)
               Squint returns shape (batch_size, 2) — handled separately.
    """
    B = batch_size
    N = num_patches
    spatial = N - 1       # 196 spatial patches (excluding CLS at index 0)
    dropout_ratio = cfg.dropout_ratio if cfg else 0.5

    if mode == "baseline":
        # ── Mode 1: Full attention — all 197 patches visible ─────────────────
        return torch.ones(B, N, dtype=torch.long, device=device)

    elif mode == "random_dropout":
        # ── Mode 2: Randomly zero 50% of spatial patches; keep CLS ──────────
        mask = torch.ones(B, N, dtype=torch.long, device=device)
        n_drop = int(spatial * dropout_ratio)
        for b in range(B):
            drop_indices = torch.randperm(spatial, device=device)[:n_drop] + 1
            mask[b, drop_indices] = 0
        return mask

    elif mode == "center_focus":
        # ── Mode 3: Keep only the inner 8×8 of the 14×14 spatial grid ────────
        GRID = 14
        INNER = 8
        offset = (GRID - INNER) // 2  # 3

        keep_indices = set()
        for row in range(offset, offset + INNER):
            for col in range(offset, offset + INNER):
                keep_indices.add(row * GRID + col + 1)  # +1 for CLS offset

        mask = torch.zeros(B, N, dtype=torch.long, device=device)
        mask[:, 0] = 1  # Always keep CLS
        for idx in keep_indices:
            if idx < N:
                mask[:, idx] = 1
        return mask

    elif mode == "squint":
        # ── Mode 4: Global Pooling Proxy ──────────────────────────────────────
        # Returns a 2-token mask: [CLS=1, global_pool=1]
        # The actual global pooling is handled in evaluate_blip_ablation().
        return torch.ones(B, 2, dtype=torch.long, device=device)

    else:
        raise ValueError(
            f"Unknown ablation mode: {mode!r}. "
            "Choose from: baseline, random_dropout, center_focus, squint"
        )


# ─────────────────────────────────────────────────────────────────────────────
# BLIP CIDEr Evaluation (single mode)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_blip_ablation(model, processor, dataloader, device,
                            mode="baseline", cfg=None,
                            num_beams=4, max_new_tokens=32,
                            length_penalty=1.0, eval_batches=25):
    """
    Evaluate BLIP CIDEr score for a specific ablation mode.

    For 'squint' mode, we manually extract the visual encoder embeddings,
    pool the spatial patches, and pass them as encoder_hidden_states directly.
    For all other modes, we use generate_with_mask() with encoder_attention_mask.

    Args:
        eval_batches  : max number of batches to evaluate (keep small for speed)
        length_penalty: passed to beam search (1.0 = neutral, >1 favors longer)

    Returns:
        cider_score: float
    """
    model.eval()
    gts = {}
    res = {}

    print(f"\n{'='*60}")
    print(f"  Ablation Mode : {mode.upper()}")
    print(f"  Beams={num_beams}  MaxTokens={max_new_tokens}  LenPenalty={length_penalty}")
    print(f"{'='*60}")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc=f"Eval [{mode}]")):
            if i >= eval_batches:
                break

            pixel_values = batch["pixel_values"].to(device)
            B = pixel_values.shape[0]

            if mode == "squint":
                vision_outputs = model.vision_model(pixel_values=pixel_values)
                hidden_states = vision_outputs.last_hidden_state  # (B, 197, 768)
                cls_token = hidden_states[:, :1, :]
                spatial = hidden_states[:, 1:, :]
                global_pool = spatial.mean(dim=1, keepdim=True)
                pooled_hidden = torch.cat([cls_token, global_pool], dim=1)

                decoded = generate_with_mask(
                    model, processor, device=device,
                    encoder_hidden_states=pooled_hidden,
                    encoder_attention_mask=torch.ones(B, 2, dtype=torch.long, device=device),
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                )
            else:
                num_patches = 197
                mask = build_ablation_mask(mode, B, num_patches, device, cfg)
                decoded = generate_with_mask(
                    model, processor, device=device,
                    pixel_values=pixel_values,
                    encoder_attention_mask=mask,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                )

            preds = decoded  # generate_with_mask returns decoded strings

            labels = batch["labels"].clone()
            gts_batch = processor.batch_decode(labels, skip_special_tokens=True)

            for j in range(len(preds)):
                idx_key = str(i * len(preds) + j)
                res[idx_key] = [preds[j]]
                gts[idx_key] = [gts_batch[j]]

    if not gts:
        print("⚠️  No predictions gathered. Returning 0.")
        return 0.0

    cider_scorer = Cider()
    score, _ = cider_scorer.compute_score(gts, res)
    print(f"  ✅ CIDEr [{mode}]: {score:.4f}")
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Full Ablation Study
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation_study(model, processor, dataloader, device, cfg,
                       num_beams=4, max_new_tokens=32, length_penalty=1.0,
                       eval_batches=25):
    """
    Run all 4 ablation modes and print a CIDEr comparison table.

    Returns:
        results: dict mapping mode → CIDEr score
    """
    results = {}
    for mode in ABLATION_MODES:
        score = evaluate_blip_ablation(
            model, processor, dataloader, device,
            mode=mode, cfg=cfg,
            num_beams=num_beams, max_new_tokens=max_new_tokens,
            length_penalty=length_penalty,
            eval_batches=eval_batches,
        )
        results[mode] = score

    print("\n")
    print("=" * 60)
    print("  Cross-Attention Ablation Results (CIDEr)")
    print(f"  Beams={num_beams}  MaxTokens={max_new_tokens}  LenPenalty={length_penalty}")
    print("=" * 60)
    print(f"  {'Mode':<25} {'CIDEr':>10}  {'Δ Baseline':>12}")
    print("-" * 60)
    baseline_score = results.get("baseline", 0.0)
    for mode, score in results.items():
        delta = score - baseline_score
        sign = "+" if delta >= 0 else ""
        print(f"  {mode:<25} {score:>10.4f}  {sign}{delta:>11.4f}")
    print("=" * 60)
    print("=" * 60)

    return results

if __name__ == "__main__":
    import argparse
    from config import CFG
    from models.blip_tuner import get_blip_model
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    import aiohttp

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_batches", type=int, default=25)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cfg = CFG.load_for_model("blip")
    model, processor = get_blip_model(cfg, device)

    ds = load_dataset(
        cfg.dataset_id,
        storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}}
    )
    val_split = "validation" if "validation" in ds else "train"
    val_hf = ds[val_split].shuffle(seed=43).select(range(min(2000, len(ds[val_split]))))

    def _collate(examples):
        images = [ex["image"].convert("RGB") for ex in examples]
        captions = [ex["captions"][0] for ex in examples]
        enc = processor(images=images, text=captions, padding="max_length", truncation=True, max_length=cfg.max_target_len, return_tensors="pt")
        enc["labels"] = enc["input_ids"].clone()
        return enc

    val_loader = DataLoader(val_hf, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=_collate)

    run_ablation_study(model, processor, val_loader, device, cfg, eval_batches=args.eval_batches)
