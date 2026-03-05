"""
experiments/parameter_sweep.py
================================
Sweep beam_size, length_penalty, and max_new_tokens across BLIP, ViT-GPT2,
and GIT to measure the effect of decoding parameters on caption quality (CIDEr).

Usage:
    python -m experiments.parameter_sweep --model blip --eval_batches 15

The sweep matrix:
    beam_size    : [3, 5, 10]
    length_penalty: [0.8, 1.0, 1.2]
    max_new_tokens: [20, 50]

Each cell reports CIDEr on the validation set (25 batches by default).
A summary table is printed at the end.

Insight guide:
  - beam_size ↑  → more diverse candidates considered, usually better quality
                   but slower decoding; diminishing returns above ~5
  - length_penalty < 1.0 → penalizes shorter sequences → longer captions
  - length_penalty > 1.0 → rewards shorter sequences → more compact captions
  - max_new_tokens ↑ → allows longer captions; may hurt CIDEr if model rambles
"""

import argparse
import itertools
import torch
from tqdm.auto import tqdm
from pycocoevalcap.cider.cider import Cider


# ─────────────────────────────────────────────────────────────────────────────
# Default Search Space
# ─────────────────────────────────────────────────────────────────────────────

BEAM_SIZES     = [3, 5, 10]
LENGTH_PENALTIES = [0.8, 1.0, 1.2]
MAX_TOKENS     = [20, 50]


# ─────────────────────────────────────────────────────────────────────────────
# Per-Model Caption Generator (handles BLIP / ViT-GPT2 / GIT)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_blip(model, processor, batch, device,
                   num_beams, max_new_tokens, length_penalty):
    pixel_values = batch["pixel_values"].to(device)
    with torch.no_grad():
        out = model.generate(
            pixel_values=pixel_values,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            length_penalty=length_penalty,
        )
    return processor.batch_decode(out, skip_special_tokens=True)


def _generate_vit_gpt2(model, tokenizer, batch, device,
                        num_beams, max_new_tokens, length_penalty):
    pixel_values = batch["pixel_values"].to(device)
    with torch.no_grad():
        out = model.generate(
            pixel_values=pixel_values,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            length_penalty=length_penalty,
        )
    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in out]


def _generate_git(model, processor, batch, device,
                  num_beams, max_new_tokens, length_penalty):
    inputs = {k: v.to(device) for k, v in batch.items()
              if k in ("pixel_values", "input_ids", "attention_mask")}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            length_penalty=length_penalty,
        )
    return processor.batch_decode(out, skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# CIDEr Evaluator for One Configuration
# ─────────────────────────────────────────────────────────────────────────────

def eval_one_config(model_name, model_objs, dataloader, device,
                    num_beams, max_new_tokens, length_penalty,
                    eval_batches=25):
    """
    Evaluate CIDEr for one (model, num_beams, max_new_tokens, length_penalty) combo.

    model_objs: dict with keys depending on model_name
      - blip:     {'model': ..., 'processor': ...}
      - vit_gpt2: {'model': ..., 'tokenizer': ...}
      - git:      {'model': ..., 'processor': ...}

    Returns:
        cider_score: float
    """
    gts, res = {}, {}

    for i, batch in enumerate(tqdm(
            dataloader,
            desc=f"  {model_name} b={num_beams} L={length_penalty} T={max_new_tokens}",
            leave=False)):
        if i >= eval_batches:
            break

        if model_name == "blip":
            preds = _generate_blip(
                model_objs["model"], model_objs["processor"],
                batch, device, num_beams, max_new_tokens, length_penalty)
            labels = batch["labels"].clone()
            gt_texts = model_objs["processor"].batch_decode(
                labels, skip_special_tokens=True)

        elif model_name == "vit_gpt2":
            preds = _generate_vit_gpt2(
                model_objs["model"], model_objs["tokenizer"],
                batch, device, num_beams, max_new_tokens, length_penalty)
            labels = batch["labels"].clone()
            labels[labels == -100] = model_objs["pad_token_id"]
            gt_texts = model_objs["tokenizer"].batch_decode(
                labels, skip_special_tokens=True)

        elif model_name == "git":
            preds = _generate_git(
                model_objs["model"], model_objs["processor"],
                batch, device, num_beams, max_new_tokens, length_penalty)
            labels = batch["labels"].clone()
            labels[labels == -100] = model_objs["processor"].tokenizer.pad_token_id
            gt_texts = model_objs["processor"].batch_decode(
                labels, skip_special_tokens=True)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        for j, (pred, gt) in enumerate(zip(preds, gt_texts)):
            key = str(i * len(preds) + j)
            res[key] = [pred]
            gts[key] = [gt]

    if not gts:
        return 0.0

    scorer = Cider()
    score, _ = scorer.compute_score(gts, res)
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Full Sweep Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_parameter_sweep(model_name, model_objs, dataloader, device,
                        beam_sizes=None, length_penalties=None, max_tokens=None,
                        eval_batches=25):
    """
    Run the full decoding parameter sweep for one model.

    Args:
        model_name        : 'blip' | 'vit_gpt2' | 'git'
        model_objs        : dict of model + processor/tokenizer references
        dataloader        : validation DataLoader
        device            : torch.device
        beam_sizes        : list of int beam sizes (default: [3, 5, 10])
        length_penalties  : list of float penalties (default: [0.8, 1.0, 1.2])
        max_tokens        : list of int max new tokens (default: [20, 50])
        eval_batches      : number of batches per configuration

    Returns:
        results: list of dicts with keys:
            model, beam_size, length_penalty, max_tokens, cider
    """
    beam_sizes       = beam_sizes or BEAM_SIZES
    length_penalties = length_penalties or LENGTH_PENALTIES
    max_tokens       = max_tokens or MAX_TOKENS

    combos = list(itertools.product(beam_sizes, length_penalties, max_tokens))
    print(f"\n🔬 Parameter Sweep — {model_name.upper()} ({len(combos)} configurations)")
    print("=" * 70)

    results = []
    for num_beams, lp, mt in combos:
        score = eval_one_config(
            model_name, model_objs, dataloader, device,
            num_beams=num_beams, max_new_tokens=mt,
            length_penalty=lp, eval_batches=eval_batches,
        )
        results.append({
            "model": model_name, "beam_size": num_beams,
            "length_penalty": lp, "max_tokens": mt, "cider": score,
        })

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Parameter Sweep Results — {model_name.upper()}")
    print(f"{'='*70}")
    print(f"  {'Beams':>5}  {'LenPenalty':>10}  {'MaxTok':>7}  {'CIDEr':>8}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*7}  {'-'*8}")
    best = max(results, key=lambda r: r["cider"])
    for r in sorted(results, key=lambda x: (-x["cider"], x["beam_size"])):
        marker = " ← best" if r == best else ""
        print(f"  {r['beam_size']:>5}  {r['length_penalty']:>10.1f}  "
              f"{r['max_tokens']:>7}  {r['cider']:>8.4f}{marker}")
    print(f"{'='*70}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Decoding parameter sweep")
    parser.add_argument("--model", choices=["blip", "vit_gpt2", "git"],
                        default="blip")
    parser.add_argument("--eval_batches", type=int, default=15)
    args = parser.parse_args()

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from config import CFG
    from data_prep import get_dataloaders, get_dataloaders_for_model

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    cfg = CFG.load_for_model(args.model)

    if args.model == "blip":
        from models.blip_tuner import get_blip_model
        model, processor = get_blip_model(cfg, device)
        model.eval()
        _, val_loader = get_dataloaders(cfg, processor)
        model_objs = {"model": model, "processor": processor}

    elif args.model == "vit_gpt2":
        from models.vit_gpt2_tuner import get_vit_gpt2_model
        model, processor, tokenizer = get_vit_gpt2_model(cfg, device)
        model.eval()
        _, val_loader = get_dataloaders_for_model(cfg, "vit_gpt2", processor, tokenizer)
        model_objs = {"model": model, "tokenizer": tokenizer,
                      "pad_token_id": tokenizer.pad_token_id}

    elif args.model == "git":
        from models.git_tuner import get_git_model
        model, processor = get_git_model(cfg, device)
        model.eval()
        _, val_loader = get_dataloaders_for_model(cfg, "git", processor)
        model_objs = {"model": model, "processor": processor}

    run_parameter_sweep(
        args.model, model_objs, val_loader, device,
        eval_batches=args.eval_batches,
    )


if __name__ == "__main__":
    main()
