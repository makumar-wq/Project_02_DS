"""
eval.py
=======
Unified Evaluator — CIDEr across all four VLM architectures.

This module:
  1. Evaluates each model's baseline CIDEr on the COCO validation set
  2. Delegates ablation studies to experiments/ablation_study.py
  3. Provides a unified cross-model comparison table

Weight Selection (--weights flag):
  base       → Use pretrained HuggingFace weights (no fine-tuning)
  finetuned  → Load from outputs/{model}/latest/
  best       → Load from outputs/{model}/best/

Usage:
    python eval.py                              # BLIP base weights
    python eval.py --model blip --weights best  # BLIP best fine-tuned
    python eval.py --model all                  # All 4 models
    python eval.py --model all --weights best   # All 4 models, best weights
    python eval.py --ablation                   # BLIP 4-mode ablation
    python eval.py --sweep                      # Decoding parameter sweep
"""

import os
import argparse
import torch
from typing import Optional
from tqdm.auto import tqdm
from pycocoevalcap.cider.cider import Cider

from config import CFG
from data_prep import get_dataloaders, get_dataloaders_for_model
from models.blip_tuner import get_blip_model, load_ckpt, generate_with_mask
from experiments.ablation_study import run_ablation_study


# ─────────────────────────────────────────────────────────────────────────────
# Device Helper
# ─────────────────────────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Weight Loading Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_weights_dir(cfg, model_name: str, weights: str) -> Optional[str]:
    """
    Return the checkpoint directory for the given model and weight selection.

    Args:
        cfg         : CFG instance
        model_name  : 'blip', 'vit_gpt2', 'git', 'custom'
        weights     : 'base', 'finetuned', 'best'

    Returns:
        Absolute path to checkpoint dir, or None for base weights.
    """
    if weights == "base":
        return None

    subdir = "latest" if weights == "finetuned" else "best"
    path = os.path.join(cfg.output_root, model_name, subdir)

    if os.path.isdir(path) and os.listdir(path):
        return path

    print(f"⚠️  No {subdir} checkpoint found at {path}. Falling back to base weights.")
    return None


def print_weights_banner(model_name: str, weights: str, ckpt_dir: Optional[str]):
    """Print a clear banner showing which weights are being used."""
    print("=" * 60)
    print(f"  Model: {model_name}")
    if ckpt_dir:
        print(f"  Weights: {weights} → {ckpt_dir}")
    else:
        print(f"  Weights: base (pretrained, no fine-tuning)")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# BLIP Baseline CIDEr Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_blip(model, processor, dataloader, device,
                  num_beams=4, max_new_tokens=32, length_penalty=1.0,
                  eval_batches=25):
    """Evaluate BLIP CIDEr score (full attention — no ablation masking)."""
    model.eval()
    gts, res = {}, {}

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Eval [BLIP]")):
            if i >= eval_batches:
                break
            pixel_values = batch["pixel_values"].to(device)
            B = pixel_values.shape[0]
            mask = torch.ones(B, 197, dtype=torch.long, device=device)

            decoded = generate_with_mask(
                model, processor, device=device,
                pixel_values=pixel_values,
                encoder_attention_mask=mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            preds = decoded  # generate_with_mask already returns decoded strings
            labels = batch["labels"].clone()
            gt_texts = processor.batch_decode(labels, skip_special_tokens=True)

            for j, (p, g) in enumerate(zip(preds, gt_texts)):
                k = str(i * len(preds) + j)
                res[k] = [p]
                gts[k] = [g]

    if not gts:
        return 0.0
    scorer = Cider()
    score, _ = scorer.compute_score(gts, res)
    print(f"  ✅ CIDEr [BLIP]: {score:.4f}")
    return score


# ─────────────────────────────────────────────────────────────────────────────
# ViT-GPT2 CIDEr Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_vit_gpt2(model, tokenizer, dataloader, device,
                      num_beams=4, max_new_tokens=32, length_penalty=1.0,
                      eval_batches=25):
    """Evaluate ViT-GPT2 CIDEr score."""
    model.eval()
    gts, res = {}, {}

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Eval [ViT-GPT2]")):
            if i >= eval_batches:
                break
            pixel_values = batch["pixel_values"].to(device)
            out = model.generate(
                pixel_values=pixel_values,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                length_penalty=length_penalty,
            )
            preds = [tokenizer.decode(ids, skip_special_tokens=True) for ids in out]
            labels = batch["labels"].clone()
            labels[labels == -100] = tokenizer.pad_token_id
            gt_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

            for j, (p, g) in enumerate(zip(preds, gt_texts)):
                k = str(i * len(preds) + j)
                res[k] = [p]
                gts[k] = [g]

    if not gts:
        return 0.0
    scorer = Cider()
    score, _ = scorer.compute_score(gts, res)
    print(f"  ✅ CIDEr [ViT-GPT2]: {score:.4f}")
    return score


# ─────────────────────────────────────────────────────────────────────────────
# GIT CIDEr Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_git(model, processor, dataloader, device,
                 num_beams=4, max_new_tokens=32, length_penalty=1.0,
                 eval_batches=25):
    """Evaluate GIT CIDEr score."""
    model.eval()
    gts, res = {}, {}

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Eval [GIT]")):
            if i >= eval_batches:
                break
            inputs = {k: v.to(device) for k, v in batch.items()
                      if k in ("pixel_values", "input_ids", "attention_mask")}
            out = model.generate(
                **inputs,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                length_penalty=length_penalty,
            )
            preds = processor.batch_decode(out, skip_special_tokens=True)
            labels = batch["labels"].clone()
            labels[labels == -100] = processor.tokenizer.pad_token_id
            gt_texts = processor.batch_decode(labels, skip_special_tokens=True)

            for j, (p, g) in enumerate(zip(preds, gt_texts)):
                k = str(i * len(preds) + j)
                res[k] = [p]
                gts[k] = [g]

    if not gts:
        return 0.0
    scorer = Cider()
    score, _ = scorer.compute_score(gts, res)
    print(f"  ✅ CIDEr [GIT]: {score:.4f}")
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Custom VLM CIDEr Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_custom_vlm_cider(model, val_loader, device,
                               char_to_idx, idx_to_char,
                               max_new_tokens=80, num_beams=1,
                               length_penalty=1.0,
                               eval_batches=20):
    """Evaluate CIDEr score for the CustomVLM using autoregressive generation."""
    model.eval()
    gts, res = {}, {}

    print("\nEvaluating Custom VLM (Visual Prefix-Tuning)...")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Eval [CustomVLM]")):
            if i >= eval_batches:
                break
            pixel_values = batch["pixel_values"].to(device)
            B = pixel_values.shape[0]

            for b in range(B):
                pv_single = pixel_values[b:b+1]

                if num_beams > 1:
                    pred = model.generate_beam(
                        pv_single, char_to_idx, idx_to_char,
                        max_new_tokens=max_new_tokens,
                        num_beams=num_beams,
                        length_penalty=length_penalty,
                    )
                else:
                    pred = model.generate(
                        pv_single, char_to_idx, idx_to_char,
                        max_new_tokens=max_new_tokens,
                    )

                tgt_ids = batch["text_targets"][b].tolist()
                gt_text = "".join(idx_to_char.get(idx, "") for idx in tgt_ids if idx != 0)

                idx_key = str(i * B + b)
                res[idx_key] = [pred.strip()]
                gts[idx_key] = [gt_text.strip()]

    if not gts:
        return 0.0

    scorer = Cider()
    score, _ = scorer.compute_score(gts, res)
    print(f"  ✅ CIDEr [CustomVLM]: {score:.4f}")
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Custom VLM Loader (with weight selection)
# ─────────────────────────────────────────────────────────────────────────────

def load_custom_vlm_for_eval(cfg, device, weights="base"):
    """
    Load CustomVLM with the specified weight selection.

    Args:
        weights: 'base' (Shakespeare only), 'finetuned' (latest ckpt), 'best' (best ckpt)
    """
    from models.custom_vlm import CustomVLM, build_char_vocab
    from data_prep import get_custom_vlm_dataloader

    with open(cfg.shakespeare_file, "r") as f:
        text = f.read()
    _, c2i, i2c, vs = build_char_vocab(text)

    model = CustomVLM(
        vocab_size=vs,
        text_embed_dim=cfg.text_embed_dim,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        block_size=cfg.block_size,
        dropout=cfg.dropout,
    )

    # Always load Shakespeare weights first
    if os.path.exists(cfg.shakespeare_weights_path):
        model.load_shakespeare_weights(cfg.shakespeare_weights_path)

    # Then optionally load fine-tuned weights on top
    ckpt_dir = get_weights_dir(cfg, "custom_vlm", weights)
    if ckpt_dir:
        ckpt_path = os.path.join(ckpt_dir, "custom_vlm.pt")
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu")
            # Filter shape mismatches gracefully
            own_state = model.state_dict()
            filtered = {k: v for k, v in state["model_state"].items()
                        if k in own_state and own_state[k].shape == v.shape}
            model.load_state_dict(filtered, strict=False)
            print(f"  ✅ Loaded fine-tuned weights from {ckpt_path}")

    print_weights_banner("Custom VLM", weights, ckpt_dir)
    model.to(device).eval()

    _, val_loader = get_custom_vlm_dataloader(cfg, c2i)
    return model, c2i, i2c, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# All-Model Comparison Table
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all_models(cfg, device, weights="base",
                        num_beams=4, max_new_tokens=32,
                        length_penalty=1.0, eval_batches=25):
    """Run CIDEr evaluation for all four models and print a comparison table."""
    results = {}

    # ── BLIP ────────────────────────────────────────────────────────────────
    print("\n[1/4] Evaluating BLIP...")
    blip_cfg = CFG.load_for_model("blip")
    model_b, proc_b = get_blip_model(blip_cfg, device)
    ckpt = get_weights_dir(blip_cfg, "blip", weights)
    if ckpt:
        load_ckpt(model_b, None, None, ckpt)
    print_weights_banner("BLIP", weights, ckpt)
    _, val_b = get_dataloaders(blip_cfg, proc_b)
    results["BLIP"] = evaluate_blip(
        model_b, proc_b, val_b, device,
        num_beams=num_beams, max_new_tokens=max_new_tokens,
        length_penalty=length_penalty, eval_batches=eval_batches,
    )
    del model_b, proc_b

    # ── ViT-GPT2 ────────────────────────────────────────────────────────────
    print("\n[2/4] Evaluating ViT-GPT2...")
    from models.vit_gpt2_tuner import get_vit_gpt2_model
    vg2_cfg = CFG.load_for_model("vit_gpt2")
    model_v, proc_v, tok_v = get_vit_gpt2_model(vg2_cfg, device)
    ckpt = get_weights_dir(vg2_cfg, "vit_gpt2", weights)
    if ckpt:
        from transformers import VisionEncoderDecoderModel
        finetuned = VisionEncoderDecoderModel.from_pretrained(ckpt)
        model_v.load_state_dict(finetuned.state_dict())
        model_v.to(device)
    print_weights_banner("ViT-GPT2", weights, ckpt)
    _, val_v = get_dataloaders_for_model(vg2_cfg, "vit_gpt2", proc_v, tok_v)
    results["ViT-GPT2"] = evaluate_vit_gpt2(
        model_v, tok_v, val_v, device,
        num_beams=num_beams, max_new_tokens=max_new_tokens,
        length_penalty=length_penalty, eval_batches=eval_batches,
    )
    del model_v, proc_v, tok_v

    # ── GIT ─────────────────────────────────────────────────────────────────
    print("\n[3/4] Evaluating GIT...")
    from models.git_tuner import get_git_model
    git_cfg = CFG.load_for_model("git")
    model_g, proc_g = get_git_model(git_cfg, device)
    ckpt = get_weights_dir(git_cfg, "git", weights)
    if ckpt:
        from transformers import AutoModelForCausalLM
        finetuned = AutoModelForCausalLM.from_pretrained(ckpt)
        model_g.load_state_dict(finetuned.state_dict())
        model_g.to(device)
    print_weights_banner("GIT", weights, ckpt)
    _, val_g = get_dataloaders_for_model(git_cfg, "git", proc_g)
    results["GIT"] = evaluate_git(
        model_g, proc_g, val_g, device,
        num_beams=num_beams, max_new_tokens=max_new_tokens,
        length_penalty=length_penalty, eval_batches=eval_batches,
    )
    del model_g, proc_g

    # ── Custom VLM ──────────────────────────────────────────────────────────
    print("\n[4/4] Evaluating Custom VLM...")
    vlm_cfg = CFG.load_for_model("custom")
    model_c, c2i, i2c, val_c = load_custom_vlm_for_eval(vlm_cfg, device, weights)
    results["CustomVLM"] = evaluate_custom_vlm_cider(
        model_c, val_c, device, c2i, i2c,
        max_new_tokens=80, eval_batches=15,
    )
    del model_c

    # ── Summary Table ────────────────────────────────────────────────────────
    print("\n")
    print("=" * 65)
    print(f"  All-Model CIDEr Comparison  |  Weights: {weights}")
    print(f"  Beams={num_beams}  MaxTok={max_new_tokens}  LenPen={length_penalty}")
    print("=" * 65)
    print(f"  {'Architecture':<22}  {'CIDEr':>8}  {'CA Type'}")
    print("  " + "-" * 61)
    ca_types = {
        "BLIP": "Gated MED cross-attention",
        "ViT-GPT2": "Standard full cross-attention",
        "GIT": "Self-attention prefix (no CA)",
        "CustomVLM": "Linear bridge prefix (no CA)",
    }
    for name, score in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:<22}  {score:>8.4f}  {ca_types.get(name, '')}")
    print("=" * 65)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Unified VLM Evaluator")
    parser.add_argument(
        "--model", type=str, default="blip",
        choices=["blip", "vit_gpt2", "git", "custom", "all"],
        help="Which model(s) to evaluate",
    )
    parser.add_argument(
        "--weights", type=str, default="base",
        choices=["base", "finetuned", "best"],
        help="Which weights to use: base (pretrained), finetuned (latest/), best (best/)",
    )
    parser.add_argument("--ablation", action="store_true",
                        help="Run BLIP 4-mode cross-attention ablation study")
    parser.add_argument("--sweep", action="store_true",
                        help="Run decoding parameter sweep")
    parser.add_argument("--num_beams", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--length_penalty", type=float, default=1.2)
    parser.add_argument("--eval_batches", type=int, default=25)
    args = parser.parse_args()

    device = get_device()
    print(f"✅ Device: {device}")

    if args.model == "all":
        cfg = CFG.load_for_model("blip")
        evaluate_all_models(
            cfg, device,
            weights=args.weights,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            length_penalty=args.length_penalty,
            eval_batches=args.eval_batches,
        )
        return

    cfg = CFG.load_for_model(args.model)

    if args.model == "blip" or args.ablation:
        model, processor = get_blip_model(cfg, device)

        ckpt_dir = get_weights_dir(cfg, "blip", args.weights)
        if ckpt_dir:
            load_ckpt(model, None, None, ckpt_dir)
        print_weights_banner("BLIP", args.weights, ckpt_dir)

        _, val_loader = get_dataloaders(cfg, processor)

        if args.ablation:
            run_ablation_study(
                model, processor, val_loader, device, cfg,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                length_penalty=args.length_penalty,
                eval_batches=args.eval_batches,
            )
        elif args.sweep:
            from experiments.parameter_sweep import run_parameter_sweep
            run_parameter_sweep(
                "blip",
                {"model": model, "processor": processor},
                val_loader, device,
                eval_batches=args.eval_batches,
            )
        else:
            evaluate_blip(
                model, processor, val_loader, device,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                length_penalty=args.length_penalty,
                eval_batches=args.eval_batches,
            )

    elif args.model == "vit_gpt2":
        from models.vit_gpt2_tuner import get_vit_gpt2_model
        model, processor, tokenizer = get_vit_gpt2_model(cfg, device)
        ckpt_dir = get_weights_dir(cfg, "vit_gpt2", args.weights)
        if ckpt_dir:
            from transformers import VisionEncoderDecoderModel
            finetuned = VisionEncoderDecoderModel.from_pretrained(ckpt_dir)
            model.load_state_dict(finetuned.state_dict())
            model.to(device)
        print_weights_banner("ViT-GPT2", args.weights, ckpt_dir)
        _, val_loader = get_dataloaders_for_model(cfg, "vit_gpt2", processor, tokenizer)
        evaluate_vit_gpt2(
            model, tokenizer, val_loader, device,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            length_penalty=args.length_penalty,
            eval_batches=args.eval_batches,
        )

    elif args.model == "git":
        from models.git_tuner import get_git_model
        model, processor = get_git_model(cfg, device)
        ckpt_dir = get_weights_dir(cfg, "git", args.weights)
        if ckpt_dir:
            from transformers import AutoModelForCausalLM
            finetuned = AutoModelForCausalLM.from_pretrained(ckpt_dir)
            model.load_state_dict(finetuned.state_dict())
            model.to(device)
        print_weights_banner("GIT", args.weights, ckpt_dir)
        _, val_loader = get_dataloaders_for_model(cfg, "git", processor)
        evaluate_git(
            model, processor, val_loader, device,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            length_penalty=args.length_penalty,
            eval_batches=args.eval_batches,
        )

    elif args.model == "custom":
        vlm_cfg = CFG.load_for_model("custom")
        model, c2i, i2c, val_loader = load_custom_vlm_for_eval(
            vlm_cfg, device, args.weights)
        evaluate_custom_vlm_cider(
            model, val_loader, device, c2i, i2c,
            max_new_tokens=80,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            eval_batches=args.eval_batches,
        )


if __name__ == "__main__":
    main()
