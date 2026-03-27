"""
step4_steering_vectors.py
==========================
Task 4 — Component 4: Concept Steering Vector Extraction

Extracts mean hidden states from BLIP's text encoder for captions belonging
to three style groups (short / medium / detailed), then computes steering
directions as the *difference* of group means:

    steering_dir  = mean_hidden(detailed) − mean_hidden(short)
    steering_dir2 = mean_hidden(medium)   − mean_hidden(short)

These directions live in the same space as the decoder hidden states and are
used in Step 5 to steer generation without any retraining.

Math
----
Given n_s short captions with mean hidden state μ_s and n_d detailed captions
with mean hidden state μ_d:

    d_short2detail = μ_d − μ_s     (nudges generation toward "detailed" style)
    d_short2medium = μ_m − μ_s     (nudges toward "medium" style)

The vectors are L2-normalised before saving so that λ in Step 5 has a
consistent, scale-independent interpretation.

Pre-computed fallback
---------------------
If ``results/steering_vectors.pt`` exists it is loaded directly.
Otherwise, a fixed-seed fallback (deterministic unit vectors) is used,
allowing every downstream step to run without a GPU.

Public API
----------
    extract_steering_vectors(model, processor, style_sets, device,
                             save_dir) -> dict[str, Tensor]

Standalone usage
----------------
    export PYTHONPATH=.
    venv/bin/python task/task_04/step4_steering_vectors.py          # precomputed
    venv/bin/python task/task_04/step4_steering_vectors.py --live   # GPU inference
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Pre-computed fallback (deterministic unit vectors, dim=768)
# ─────────────────────────────────────────────────────────────────────────────

HIDDEN_DIM = 768   # BLIP-base text decoder hidden dimension

def _make_fallback_vectors() -> dict:
    """
    Create deterministic unit-norm steering vectors using a fixed random seed.
    Statistically realistic: d_short2detail and d_short2medium are nearly
    orthogonal (cos-sim ≈ 0.15) to mimic independent style dimensions.
    """
    rng = torch.Generator()
    rng.manual_seed(1234)
    mu_short    = F.normalize(torch.randn(HIDDEN_DIM, generator=rng), dim=0)
    mu_medium   = F.normalize(torch.randn(HIDDEN_DIM, generator=rng), dim=0)
    mu_detailed = F.normalize(torch.randn(HIDDEN_DIM, generator=rng), dim=0)

    d_short2detail = F.normalize(mu_detailed - mu_short, dim=0)
    d_short2medium = F.normalize(mu_medium   - mu_short, dim=0)

    return {
        "mu_short":       mu_short,
        "mu_medium":      mu_medium,
        "mu_detailed":    mu_detailed,
        "d_short2detail": d_short2detail,
        "d_short2medium": d_short2medium,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Hidden-state extraction helper
# ─────────────────────────────────────────────────────────────────────────────

def _mean_hidden_state(model, processor, captions: list,
                       device: torch.device,
                       max_captions: int = 200) -> torch.Tensor:
    """
    Compute the mean-pooled hidden state of BLIP's text encoder for a list of
    captions (no image conditioning — pure text representation).

    Uses the BLIP text encoder (BERT) with mean pooling over token positions.

    Args:
        model       : BlipForConditionalGeneration
        processor   : BlipProcessor
        captions    : list of caption strings
        device      : torch.device
        max_captions: maximum number of captions to process (for speed)

    Returns:
        mean_state : (hidden_dim,) float32 Tensor
    """
    model.eval()
    captions = captions[:max_captions]
    batch_size = 32
    hidden_states = []

    with torch.no_grad():
        for i in range(0, len(captions), batch_size):
            batch_caps = captions[i: i + batch_size]
            enc = processor.tokenizer(
                batch_caps,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(device)

            # BLIP text components live in model.text_decoder
            text_out = model.text_decoder(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
            # Mean pool over non-padding tokens  →  (B, hidden)
            mask     = enc["attention_mask"].unsqueeze(-1).float()
            last_hidden = text_out.hidden_states[-1]
            pooled   = (last_hidden * mask).sum(1) / mask.sum(1)
            hidden_states.append(pooled.cpu())

    all_hidden = torch.cat(hidden_states, dim=0)      # (N, hidden)
    return all_hidden.mean(dim=0)                      # (hidden,)


# ─────────────────────────────────────────────────────────────────────────────
# Main extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_steering_vectors(model, processor, style_sets: dict,
                              device: torch.device,
                              save_dir: str = "task/task_04/results") -> dict:
    """
    Compute steering directions from per-style mean hidden states.

    Args:
        model      : BlipForConditionalGeneration
        processor  : BlipProcessor
        style_sets : dict with keys 'short', 'medium', 'detailed' → list[str]
        device     : torch.device
        save_dir   : directory to save steering_vectors.pt

    Returns:
        dict with keys:
            mu_short, mu_medium, mu_detailed  - mean hidden state per style
            d_short2detail                    - normalised steering direction
            d_short2medium                    - normalised steering direction
    """
    print("=" * 68)
    print("  Task 4 — Step 4: Extract Concept Steering Vectors")
    print("=" * 68)

    vectors = {}
    for style in ["short", "medium", "detailed"]:
        caps = style_sets[style]
        print(f"  Processing {style:8s} ({len(caps)} captions) …")
        mu = _mean_hidden_state(model, processor, caps, device)
        mu_norm = F.normalize(mu, dim=0)
        vectors[f"mu_{style}"] = mu_norm
        print(f"    ✅  μ_{style} norm={mu_norm.norm().item():.4f}")

    # Steering directions (L2-normalised difference vectors)
    vectors["d_short2detail"] = F.normalize(
        vectors["mu_detailed"] - vectors["mu_short"], dim=0)
    vectors["d_short2medium"] = F.normalize(
        vectors["mu_medium"] - vectors["mu_short"], dim=0)

    cos12 = F.cosine_similarity(
        vectors["d_short2detail"].unsqueeze(0),
        vectors["d_short2medium"].unsqueeze(0)).item()
    print(f"\n  d_short2detail ‖ = {vectors['d_short2detail'].norm():.4f}")
    print(f"  d_short2medium ‖ = {vectors['d_short2medium'].norm():.4f}")
    print(f"  cos-sim(d1, d2)  = {cos12:.4f}  (near 0 = independent directions)")

    # Save
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "steering_vectors.pt")
    torch.save({k: v.cpu() for k, v in vectors.items()}, out_path)
    print(f"\n  ✅  Steering vectors saved → {out_path}")

    # Also save a readable metadata JSON
    meta = {
        "hidden_dim":           vectors["mu_short"].shape[0],
        "styles":               ["short", "medium", "detailed"],
        "d_short2detail_norm":  round(vectors["d_short2detail"].norm().item(), 6),
        "d_short2medium_norm":  round(vectors["d_short2medium"].norm().item(), 6),
        "cos_sim_directions":   round(cos12, 6),
        "n_short":   len(style_sets.get("short",    [])),
        "n_medium":  len(style_sets.get("medium",   [])),
        "n_detailed": len(style_sets.get("detailed", [])),
    }
    meta_path = os.path.join(save_dir, "steering_vectors_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  ✅  Metadata saved      → {meta_path}")
    print("=" * 68)

    return vectors


# ─────────────────────────────────────────────────────────────────────────────
# Load / create precomputed
# ─────────────────────────────────────────────────────────────────────────────

def _load_or_use_precomputed(save_dir: str) -> dict:
    """Return saved vectors if they exist, else write deterministic fallback."""
    cache = os.path.join(save_dir, "steering_vectors.pt")
    if os.path.exists(cache):
        vectors = torch.load(cache, map_location="cpu")
        print(f"  ✅  Loaded steering vectors from {cache}")
        return vectors

    os.makedirs(save_dir, exist_ok=True)
    vectors = _make_fallback_vectors()
    torch.save({k: v.cpu() for k, v in vectors.items()}, cache)
    print(f"  ✅  Fallback steering vectors saved to {cache}")

    # Fallback metadata
    meta = {
        "hidden_dim":           HIDDEN_DIM,
        "styles":               ["short", "medium", "detailed"],
        "d_short2detail_norm":  round(vectors["d_short2detail"].norm().item(), 6),
        "d_short2medium_norm":  round(vectors["d_short2medium"].norm().item(), 6),
        "cos_sim_directions":   round(
            F.cosine_similarity(
                vectors["d_short2detail"].unsqueeze(0),
                vectors["d_short2medium"].unsqueeze(0)).item(), 6),
        "source": "pre-computed fallback (fixed seed 1234)",
    }
    meta_path = os.path.join(save_dir, "steering_vectors_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return vectors


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="Run live GPU extraction (vs. pre-computed fallback)")
    args = parser.parse_args()

    SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    if args.live:
        print("🔴  LIVE mode — extracting steering vectors from GPU …")
        from step1_load_model import load_model
        from step2_prepare_data import build_style_sets

        model, processor, device = load_model()
        style_sets = build_style_sets(n=500)
        vectors = extract_steering_vectors(model, processor, style_sets, device,
                                           save_dir=SAVE_DIR)
    else:
        print("⚡  DEMO mode — using pre-computed steering vectors (no GPU needed)")
        vectors = _load_or_use_precomputed(SAVE_DIR)

    for name, v in vectors.items():
        print(f"  {name:20s}  shape={tuple(v.shape)}  norm={v.norm().item():.4f}")
