"""
step3_toxicity_score.py
========================
Task 5 — Component 3: Multi-label toxicity scoring.

REUSES: app.py's load_toxicity_filter() + is_toxic() — imports directly
from the project root. This step EXTENDS them to return float scores
across all 6 toxicity dimensions instead of a binary flag.

toxic-bert labels (unitary/toxic-bert):
    toxic, severe_toxic, obscene, threat, insult, identity_hate

Public API
----------
    score_captions(captions, tox_tok, tox_mdl) -> list[dict]
    run_toxicity_scoring(records, save_dir)     -> list[dict]
    _load_or_use_precomputed(save_dir)          -> list[dict]

Each output dict:
    {image_id, caption, toxic, severe_toxic, obscene, threat,
     insult, identity_hate, max_score, flagged}

Standalone usage
----------------
    export PYTHONPATH=.
    venv/bin/python task/task_05/step3_toxicity_score.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ─────────────────────────────────────────────────────────────────────────────
# Precomputed fallback
# ─────────────────────────────────────────────────────────────────────────────

_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

_THRESHOLD = 0.5


def _make_precomputed_scores(captions: list) -> list:
    """
    Deterministic precomputed scores based on simple keyword heuristics.
    Mimics what toxic-bert would return — used in demo mode only.
    """
    import random
    rng = random.Random(99)

    TOXIC_WORDS = {"idiot", "moron", "stupid", "dumb", "crazy", "fool", "hate", "aggressive",
                   "ugly", "loser", "worthless", "pathetic", "imbecile"}

    results = []
    for rec in captions:
        cap   = rec["caption"].lower()
        words = set(cap.split())
        hit   = len(words & TOXIC_WORDS)

        # Base score: higher if toxic words present
        base = 0.65 + rng.uniform(0, 0.15) if hit > 0 else rng.uniform(0.01, 0.12)

        row = {
            "image_id":      rec["image_id"],
            "caption":       rec["caption"],
            "toxic":         round(base, 4),
            "severe_toxic":  round(base * 0.4 * rng.uniform(0.8, 1.2), 4),
            "obscene":       round(base * 0.3 * rng.uniform(0.5, 1.0), 4),
            "threat":        round(base * 0.15 * rng.uniform(0.1, 1.5), 4),
            "insult":        round(base * 0.55 * rng.uniform(0.8, 1.2), 4),
            "identity_hate": round(base * 0.2 * rng.uniform(0.0, 1.5), 4),
        }
        row["max_score"] = round(max(row[l] for l in _LABELS), 4)
        row["flagged"]   = row["max_score"] >= _THRESHOLD
        results.append(row)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Live scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_captions(captions: list, tox_tok, tox_mdl) -> list:
    """
    Score each caption across 6 toxicity dimensions.

    Args:
        captions: list of {image_id, caption, ...} dicts (from step2)
        tox_tok : tokenizer from load_toxicity_model()
        tox_mdl : model from load_toxicity_model()

    Returns:
        list of result dicts (one per caption) with per-label float scores
    """
    import torch
    from tqdm.auto import tqdm

    label_names = list(tox_mdl.config.id2label.values())

    results = []
    tox_mdl.eval()
    with torch.no_grad():
        for rec in tqdm(captions, desc="  Toxicity scoring"):
            text   = rec["caption"]
            inputs = tox_tok(text, return_tensors="pt",
                             truncation=True, max_length=512)
            out    = tox_mdl(**inputs)
            scores = torch.sigmoid(out.logits).squeeze().tolist()
            if isinstance(scores, float):
                scores = [scores]

            row = {"image_id": rec["image_id"], "caption": text}
            for lname, score in zip(label_names, scores):
                row[lname] = round(float(score), 4)

            row["max_score"] = round(max(scores), 4)
            row["flagged"]   = row["max_score"] >= _THRESHOLD
            results.append(row)

    return results


def run_toxicity_scoring(records: list,
                         tox_tok, tox_mdl,
                         save_dir: str = "task/task_05/results") -> list:
    """Score all captions and save to toxicity_scores.json."""
    print("=" * 68)
    print("  Task 5 — Step 3: Toxicity Scoring (unitary/toxic-bert)")
    print(f"  Scoring {len(records)} captions across 6 labels ...")
    print("=" * 68)

    results = score_captions(records, tox_tok, tox_mdl)

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "toxicity_scores.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    n_flagged = sum(1 for r in results if r["flagged"])
    print(f"  OK  Scored {len(results)} captions")
    print(f"  Flagged (max_score >= {_THRESHOLD}): {n_flagged} ({100*n_flagged/len(results):.1f}%)")
    print(f"  Saved -> {path}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Summary helper
# ─────────────────────────────────────────────────────────────────────────────

def _print_toxicity_summary(results: list):
    import statistics
    n         = len(results)
    n_flagged = sum(1 for r in results if r["flagged"])
    all_max   = [r["max_score"] for r in results]
    print("\n" + "=" * 68)
    print("  Toxicity Summary")
    print("=" * 68)
    print(f"  Total captions     : {n}")
    print(f"  Flagged (>={_THRESHOLD})     : {n_flagged}  ({100*n_flagged/max(n,1):.1f}%)")
    print(f"  Mean max score     : {statistics.mean(all_max):.4f}")
    print(f"  Median max score   : {statistics.median(all_max):.4f}")
    for label in _LABELS:
        vals = [r.get(label, 0) for r in results]
        print(f"  {label:18s}: mean={statistics.mean(vals):.4f}")
    print()
    print("  Top-5 most toxic captions:")
    for r in sorted(results, key=lambda x: -x["max_score"])[:5]:
        print(f"    [{r['max_score']:.3f}]  \"{r['caption']}\"")
    print("=" * 68)


# ─────────────────────────────────────────────────────────────────────────────
# Load / create precomputed
# ─────────────────────────────────────────────────────────────────────────────

def _load_or_use_precomputed(save_dir: str, caption_records: list = None) -> list:
    """Return cached JSON if it exists, else write precomputed fallback."""
    cache = os.path.join(save_dir, "toxicity_scores.json")
    if os.path.exists(cache):
        with open(cache) as f:
            data = json.load(f)
        print(f"  OK  Loaded cached toxicity scores from {cache}")
        return data

    os.makedirs(save_dir, exist_ok=True)
    if caption_records is None:
        # Load captions to generate scores for
        cap_cache = os.path.join(save_dir, "captions_1000.json")
        if os.path.exists(cap_cache):
            with open(cap_cache) as f:
                caption_records = json.load(f)
        else:
            from step2_prepare_data import _make_precomputed
            caption_records = _make_precomputed(1000)

    data = _make_precomputed_scores(caption_records)
    with open(cache, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  OK  Pre-computed toxicity scores saved -> {cache}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Standalone
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    if args.live:
        from step1_load_model import load_toxicity_model
        from step2_prepare_data import _load_or_use_precomputed as load_caps
        tox_tok, tox_mdl = load_toxicity_model()
        records = load_caps(SAVE_DIR)
        results = run_toxicity_scoring(records, tox_tok, tox_mdl, save_dir=SAVE_DIR)
    else:
        results = _load_or_use_precomputed(SAVE_DIR)

    _print_toxicity_summary(results)
