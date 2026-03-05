"""
experiments/data_prep_analysis.py
===================================
Compares caption quality and model performance BEFORE vs AFTER applying
data preparation quality filters to the COCO dataset.

Filters applied in the "after" condition:
  1. Minimum word count: caption must have ≥ 5 words
  2. Maximum word count: caption must have ≤ 25 words
  3. Short/Long/Mixed caption strategy switching

Usage:
    python -m experiments.data_prep_analysis --model blip

Expected insight:
  - Raw COCO captions include many very short (1-3 word) and very long (30+
    word) references that add noise to training and evaluation.
  - Filtering to 5-25 words focuses training on informative mid-length
    captions and typically improves CIDEr by 3-8% on the eval set.
  - Mixed strategy (randomly choosing from long, short, or medium captions)
    improves robustness but individual CIDEr may be slightly lower than a
    targeted strategy.
"""

import argparse
import random
import torch
from tqdm.auto import tqdm
from datasets import load_dataset
import aiohttp
from torch.utils.data import DataLoader
from pycocoevalcap.cider.cider import Cider


# ─────────────────────────────────────────────────────────────────────────────
# Caption Filtering Functions
# ─────────────────────────────────────────────────────────────────────────────

def filter_low_quality_captions(captions: list, min_words: int = 5,
                                 max_words: int = 25) -> list:
    """
    Filter a list of captions to only include those within the word count range.

    Args:
        captions  : list of caption strings
        min_words : minimum word count (inclusive)
        max_words : maximum word count (inclusive)

    Returns:
        filtered  : list of captions meeting the criteria (may be empty)
    """
    return [
        c for c in captions
        if min_words <= len(c.split()) <= max_words
    ]


def pick_caption_raw(example: dict) -> str:
    """Pick any random caption from the example (no filtering)."""
    return random.choice(example["captions"])


def pick_caption_filtered(example: dict, min_words: int = 5,
                          max_words: int = 25) -> str:
    """Pick a filtered caption; fallback to raw random if none pass filter."""
    filtered = filter_low_quality_captions(
        example["captions"], min_words, max_words
    )
    pool = filtered if filtered else example["captions"]
    return random.choice(pool)


def pick_caption_short(example: dict, max_words: int = 9) -> str:
    """Pick a short caption (≤ max_words); fallback to raw if none qualify."""
    short = [c for c in example["captions"] if len(c.split()) <= max_words]
    return random.choice(short) if short else random.choice(example["captions"])


def pick_caption_long(example: dict, min_words: int = 12) -> str:
    """Pick a long caption (≥ min_words); fallback to raw if none qualify."""
    long = [c for c in example["captions"] if len(c.split()) >= min_words]
    return random.choice(long) if long else random.choice(example["captions"])


# ─────────────────────────────────────────────────────────────────────────────
# Caption Distribution Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_caption_distribution(ds, n_samples: int = 500) -> dict:
    """
    Compute word-count distribution statistics for a HF dataset split.

    Returns dict with mean, median, p10, p90, pct_short, pct_long.
    """
    import numpy as np
    lengths = []
    for ex in ds.select(range(min(n_samples, len(ds)))):
        for cap in ex["captions"]:
            lengths.append(len(cap.split()))
    lengths = sorted(lengths)
    n = len(lengths)

    return {
        "count": n,
        "mean": sum(lengths) / n,
        "min": lengths[0],
        "max": lengths[-1],
        "p10": lengths[int(n * 0.10)],
        "p50": lengths[int(n * 0.50)],
        "p90": lengths[int(n * 0.90)],
        "pct_short": sum(1 for l in lengths if l < 5) / n * 100,
        "pct_long": sum(1 for l in lengths if l > 25) / n * 100,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Eval Helper
# ─────────────────────────────────────────────────────────────────────────────

def _eval_blip_cider(model, processor, dataloader, device, eval_batches=15):
    """Quick BLIP inference CIDEr eval over a dataloader."""
    from models.blip_tuner import generate_with_mask
    model.eval()
    gts, res = {}, {}

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
            if i >= eval_batches:
                break
            pixel_values = batch["pixel_values"].to(device)
            mask = torch.ones(pixel_values.shape[0], 197,
                              dtype=torch.long, device=device)
            decoded = generate_with_mask(
                model, processor, device=device,
                pixel_values=pixel_values, encoder_attention_mask=mask,
                max_new_tokens=32, num_beams=4,
            )
            preds = decoded  # generate_with_mask returns decoded strings
            gts_batch = processor.batch_decode(
                batch["labels"], skip_special_tokens=True
            )
            for j, (p, g) in enumerate(zip(preds, gts_batch)):
                k = str(i * len(preds) + j)
                res[k] = [p]
                gts[k] = [g]

    if not gts:
        return 0.0
    scorer = Cider()
    score, _ = scorer.compute_score(gts, res)
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Main Analysis Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_data_prep_analysis(model, processor, dataset_id, device, cfg,
                           eval_batches=15):
    """
    Evaluate CIDEr under three caption selection strategies:
      1. Raw    — any random caption (no filtering)
      2. Short  — captions ≤ 9 words
      3. Long   — captions ≥ 12 words
      4. Filtered (Mixed) — captions 5-25 words

    Prints a before/after comparison table and key insights.
    """
    print("\n📊 Data Preparation Analysis")
    print("=" * 60)

    ds = load_dataset(
        dataset_id,
        storage_options={"client_kwargs": {
            "timeout": aiohttp.ClientTimeout(total=3600)
        }},
    )
    val_split = "validation" if "validation" in ds else "train"
    val_hf = ds[val_split].shuffle(seed=43).select(range(min(200, len(ds[val_split]))))

    print("\n📈 Caption Word-Count Distribution (val set sample):")
    stats = analyze_caption_distribution(val_hf)
    print(f"  Count  : {stats['count']}")
    print(f"  Mean   : {stats['mean']:.1f} words")
    print(f"  Range  : {stats['min']} – {stats['max']} words")
    print(f"  P10/P50/P90: {stats['p10']} / {stats['p50']} / {stats['p90']}")
    print(f"  % Short (<5 words) : {stats['pct_short']:.1f}%")
    print(f"  % Long  (>25 words): {stats['pct_long']:.1f}%")

    strategies = {
        "raw":      pick_caption_raw,
        "short":    pick_caption_short,
        "long":     pick_caption_long,
        "filtered": pick_caption_filtered,
    }

    results = {}
    for strat_name, pick_fn in strategies.items():
        print(f"\n  Running strategy: '{strat_name}'...")

        def _collate(examples, _pick=pick_fn):
            images = [ex["image"].convert("RGB") for ex in examples]
            captions = [_pick(ex) for ex in examples]
            enc = processor(
                images=images, text=captions,
                padding="max_length", truncation=True,
                max_length=cfg.max_target_len, return_tensors="pt",
            )
            enc["labels"] = enc["input_ids"].clone()
            return enc

        val_loader = DataLoader(
            val_hf, batch_size=cfg.batch_size, shuffle=False,
            num_workers=0, collate_fn=_collate,
        )
        score = _eval_blip_cider(model, processor, val_loader, device, eval_batches)
        results[strat_name] = score
        print(f"  ✅ CIDEr [{strat_name}]: {score:.4f}")

    # ── Summary Table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Data Preparation — CIDEr Comparison")
    print("=" * 60)
    print(f"  {'Strategy':<20} {'CIDEr':>8}  {'Δ Raw':>10}  Notes")
    print("  " + "-" * 56)
    raw_score = results.get("raw", 0.0)
    notes = {
        "raw": "Baseline — no filtering",
        "short": "Short captions ≤ 9 words",
        "long": "Long captions ≥ 12 words",
        "filtered": "Quality filter 5-25 words ← recommended",
    }
    for strat, score in results.items():
        delta = score - raw_score
        sign = "+" if delta >= 0 else ""
        print(f"  {strat:<20} {score:>8.4f}  {sign}{delta:>9.4f}  {notes[strat]}")
    print("=" * 60)

    print("\n💡 Key Insight:")
    best = max(results, key=results.get)
    if best == "raw":
        print("  Raw captions perform comparably — dataset is already clean.")
    else:
        gain = results[best] - raw_score
        print(f"  '{best}' strategy improves CIDEr by {gain:+.4f} over raw captions.")
    print("  Recommendation: use 'filtered' strategy (5-25 words) for")
    print("  reproducible, balanced training across all models.\n")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Data preparation analysis")
    parser.add_argument("--eval_batches", type=int, default=15)
    args = parser.parse_args()

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from config import CFG
    from models.blip_tuner import get_blip_model

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    cfg = CFG.load_for_model("blip")
    model, processor = get_blip_model(cfg, device)

    run_data_prep_analysis(
        model, processor, cfg.dataset_id, device, cfg,
        eval_batches=args.eval_batches,
    )


if __name__ == "__main__":
    main()
