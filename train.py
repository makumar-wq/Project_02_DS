"""
train.py
========
Unified training entrypoint for all VLM architectures:
  --model blip      → Fine-tune BLIP (Multimodal Mixture Attention)
  --model vit_gpt2  → Fine-tune ViT-GPT2 (Standard Cross-Attention)
  --model git       → Fine-tune GIT (Zero Cross-Attention / Self-Attention Prefix)
  --model custom    → Train visual_projection only (Visual Prefix-Tuning)

Checkpoint Strategy:
  All outputs are saved under outputs/{model_name}/:
    - latest/   — overwritten every epoch (always the most recent state)
    - best/     — overwritten only when validation loss improves

Optimized for Apple Silicon MPS backend with:
  - Gradient accumulation
  - Gradient checkpointing
  - Cosine LR scheduler with linear warmup
  - MPS-safe DataLoader settings (num_workers=0, pin_memory=False)
"""

import argparse
import math
import time
import os
import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

from config import CFG
from data_prep import get_dataloaders, get_dataloaders_for_model, get_custom_vlm_dataloader
from models.blip_tuner import get_blip_model, save_ckpt as blip_save, generate_with_mask
from models.vit_gpt2_tuner import get_vit_gpt2_model, save_ckpt as vit_gpt2_save
from models.git_tuner import get_git_model, save_ckpt as git_save
from models.custom_vlm import CustomVLM, build_char_vocab
from pycocoevalcap.cider.cider import Cider


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_output_paths(cfg, model_name: str):
    """
    Return (latest_dir, best_dir) for a given model.
    Creates directories if they don't exist.
    """
    base = os.path.join(cfg.output_root, model_name)
    latest = os.path.join(base, "latest")
    best = os.path.join(base, "best")
    os.makedirs(latest, exist_ok=True)
    os.makedirs(best, exist_ok=True)
    return latest, best


# ─────────────────────────────────────────────────────────────────────────────
# Shared Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def _generate_hf_captions(model, batch, model_name, device,
                          processor=None, tokenizer=None):
    """
    Generate captions for a batch of images using the appropriate HuggingFace model.
    Returns (predictions: list[str], ground_truths: list[str]).
    """
    pixel_values = batch["pixel_values"].to(device)

    if model_name == "BLIP":
        B = pixel_values.shape[0]
        mask = torch.ones(B, 197, dtype=torch.long, device=device)
        decoded = generate_with_mask(
            model, processor, device=device,
            pixel_values=pixel_values,
            encoder_attention_mask=mask,
            max_new_tokens=32, num_beams=4,
        )
        preds = decoded  # generate_with_mask already returns decoded strings
        labels = batch["labels"].clone()
        gt_texts = processor.batch_decode(labels, skip_special_tokens=True)

    elif model_name == "VIT_GPT2":
        out = model.generate(
            pixel_values=pixel_values, num_beams=4, max_new_tokens=32,
        )
        preds = [tokenizer.decode(ids, skip_special_tokens=True) for ids in out]
        labels = batch["labels"].clone()
        labels[labels == -100] = tokenizer.pad_token_id
        gt_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

    elif model_name == "GIT":
        inputs = {k: v.to(device) for k, v in batch.items()
                  if k in ("pixel_values", "input_ids", "attention_mask")}
        out = model.generate(**inputs, num_beams=4, max_new_tokens=32)
        preds = processor.batch_decode(out, skip_special_tokens=True)
        labels = batch["labels"].clone()
        labels[labels == -100] = processor.tokenizer.pad_token_id
        gt_texts = processor.batch_decode(labels, skip_special_tokens=True)
    else:
        return [], []

    return preds, gt_texts


def run_training_loop(model, optimizer, scheduler, train_loader, val_loader,
                      cfg, save_latest_fn, save_best_fn, model_name,
                      processor=None, tokenizer=None):
    """
    Shared gradient-accumulation training loop for all HuggingFace models.

    Now includes per-epoch:
      - Validation loss
      - CIDEr scoring via greedy generation
      - CIDEr-based checkpointing (saves best/ based on highest CIDEr)
    """
    device = get_device()
    model.train()
    global_step = 0
    best_cider = -1.0
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch}/{cfg.epochs}")
        running_loss = 0.0
        epoch_loss_sum = 0.0
        epoch_batches = 0
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(pbar, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            out = model(**batch)
            loss = out.loss / cfg.grad_accum
            loss.backward()
            running_loss += loss.item()
            epoch_loss_sum += out.loss.item()
            epoch_batches += 1

            if i % cfg.grad_accum == 0 or i == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % cfg.log_every == 0:
                    avg = running_loss / cfg.log_every
                    running_loss = 0.0
                    pbar.set_postfix({"loss": f"{avg:.4f}",
                                      "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        # End of epoch — training metrics
        epoch_avg_loss = epoch_loss_sum / max(epoch_batches, 1)
        print(f"\n📊 Epoch {epoch}/{cfg.epochs} avg loss (Train): {epoch_avg_loss:.4f}")

        # ── Validation Loop: Loss + CIDEr ────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        gts, res = {}, {}
        max_eval_batches = 10
        print("   🔍 Running Validation (Loss & CIDEr)...")

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= max_eval_batches:
                    break

                batch_d = {k: v.to(device) for k, v in batch.items()}

                # 1. Validation loss
                out = model(**batch_d)
                val_loss_sum += out.loss.item()
                val_batches += 1

                # 2. Generate captions for CIDEr
                preds, gt_texts = _generate_hf_captions(
                    model, batch, model_name, device,
                    processor=processor, tokenizer=tokenizer,
                )
                for j, (p, g) in enumerate(zip(preds, gt_texts)):
                    k = f"{epoch}_{i}_{j}"
                    res[k] = [p]
                    gts[k] = [g]

        val_avg_loss = val_loss_sum / max(val_batches, 1)
        print(f"   📉 Validation Loss: {val_avg_loss:.4f}")

        # Compute CIDEr
        cider_score = 0.0
        if gts:
            scorer = Cider()
            cider_score, _ = scorer.compute_score(gts, res)
        print(f"   🎯 Validation CIDEr: {cider_score:.4f}")

        # Save latest checkpoint
        save_latest_fn(step=global_step, epoch=epoch)
        print(f"   💾 Saved → latest/")

        # Save best based on CIDEr score
        if cider_score > best_cider:
            best_cider = cider_score
            save_best_fn(step=global_step, epoch=epoch)
            print(f"   🏆 New best CIDEr (score={best_cider:.4f}) → best/")

    elapsed = (time.time() - t0) / 60.0
    print(f"\n✅ {model_name} training complete in {elapsed:.2f} minutes")
    print(f"   Best validation CIDEr: {best_cider:.4f}")
    return global_step


# ─────────────────────────────────────────────────────────────────────────────
# Custom VLM Training (projection-only)
# ─────────────────────────────────────────────────────────────────────────────

def train_custom_vlm(cfg, device):
    print("📖 Loading Shakespeare corpus for character vocabulary...")
    with open(cfg.shakespeare_file, "r", encoding="utf-8") as f:
        text = f.read()
    _, char_to_idx, idx_to_char, vocab_size = build_char_vocab(text)
    print(f"✅ Vocabulary size: {vocab_size} characters")

    model = CustomVLM(
        vocab_size=vocab_size,
        text_embed_dim=cfg.text_embed_dim,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        block_size=cfg.block_size,
        dropout=cfg.dropout,
    )

    # ── Load pre-trained Shakespeare decoder weights (CRITICAL) ──────────────
    shakespeare_path = getattr(cfg, "shakespeare_weights_path",
                               "./shakespeare_transformer.pt")
    if os.path.exists(shakespeare_path):
        model.load_shakespeare_weights(shakespeare_path)
        print(f"✅ Shakespeare decoder weights loaded from {shakespeare_path}")
    else:
        print(f"⚠️  shakespeare_transformer.pt not found at {shakespeare_path}")
        print("    Training with randomly initialized decoder (significantly worse).")

    model.unfreeze_decoder()
    model.to(device)

    n_train = model.trainable_params()
    n_total = sum(p.numel() for p in model.parameters())
    print(f"✅ CustomVLM: {n_train:,} trainable / {n_total:,} total params")
    print(f"   (Projection + Decoder trainable — {n_train/n_total*100:.2f}%)")

    train_loader, val_loader = get_custom_vlm_dataloader(cfg, char_to_idx)

    # Discriminative learning rates: projection (higher) + decoder (gentler)
    param_groups = model.get_param_groups(
        projection_lr=cfg.lr,        # 1e-4
        decoder_lr=cfg.lr * 0.5,     # 5e-5
    )
    optimizer = AdamW(param_groups, weight_decay=cfg.weight_decay)
    total_steps = math.ceil(len(train_loader) / cfg.grad_accum) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    latest_dir, best_dir = get_output_paths(cfg, "custom_vlm")

    # Metrics history
    best_cider = -1.0
    cider_scorer = Cider()

    model.train()
    global_step = 0
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"[CustomVLM] Epoch {epoch}/{cfg.epochs}")
        running_loss = 0.0
        epoch_loss_sum = 0.0
        epoch_batches = 0
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(pbar, start=1):
            pixel_values = batch["pixel_values"].to(device)
            text_input_ids = batch["text_input_ids"].to(device)
            text_targets = batch["text_targets"].to(device)

            _, loss = model(pixel_values, text_input_ids, text_targets)
            (loss / cfg.grad_accum).backward()
            running_loss += loss.item()
            epoch_loss_sum += loss.item()
            epoch_batches += 1

            if i % cfg.grad_accum == 0 or i == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % cfg.log_every == 0:
                    avg = running_loss / cfg.log_every
                    running_loss = 0.0
                    pbar.set_postfix({"loss": f"{avg:.4f}",
                                      "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        # End of epoch metrics
        epoch_avg_loss = epoch_loss_sum / max(epoch_batches, 1)
        print(f"\n📊 Epoch {epoch}/{cfg.epochs} avg loss (Train): {epoch_avg_loss:.4f}")

        # --- Validation Loop ---
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        ref_dict = {}
        hyp_dict = {}

        # Use a small subset for quick CIDEr eval during training
        max_eval_batches = 10
        print("   🔍 Running Validation (Loss & CIDEr)...")

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= max_eval_batches:
                    break
                    
                pixel_values = batch["pixel_values"].to(device)
                text_input_ids = batch["text_input_ids"].to(device)
                text_targets = batch["text_targets"].to(device)
                
                # 1. Validation Loss
                _, loss = model(pixel_values, text_input_ids, text_targets)
                val_loss_sum += loss.item()
                val_batches += 1

                # 2. Generation for CIDEr — iterate per sample (generate expects single image)
                B = pixel_values.shape[0]
                for b in range(B):
                    pv_single = pixel_values[b:b+1]
                    gen_caption = model.generate(pv_single, char_to_idx, idx_to_char, max_new_tokens=40)
                    
                    tgt_cpu = text_targets[b].cpu().tolist()
                    true_str = "".join([idx_to_char.get(c, "") for c in tgt_cpu if c > 0])
                    
                    img_id = f"{epoch}_{i}_{b}"
                    ref_dict[img_id] = [true_str]
                    hyp_dict[img_id] = [gen_caption]

        val_avg_loss = val_loss_sum / max(val_batches, 1)
        print(f"   📉 Validation Loss: {val_avg_loss:.4f}")

        # Calculate CIDEr
        try:
            cider_score, _ = cider_scorer.compute_score(ref_dict, hyp_dict)
        except Exception:
            cider_score = 0.0
        
        print(f"   🎯 Validation CIDEr: {cider_score:.4f}")

        # Save latest (always)
        _save_custom(model, char_to_idx, idx_to_char, cfg,
                     global_step, epoch, latest_dir)
        print(f"   💾 Saved → {latest_dir}")

        # Save best (based on highest CIDEr score)
        if cider_score >= best_cider:
            best_cider = cider_score
            _save_custom(model, char_to_idx, idx_to_char, cfg,
                         global_step, epoch, best_dir)
            print(f"   🏆 New best CIDEr (score={best_cider:.4f}) → {best_dir}")

    elapsed = (time.time() - t0) / 60.0
    print(f"\n✅ CustomVLM training complete in {elapsed:.2f} minutes")
    print(f"   Best validation CIDEr: {best_cider:.4f}")


def _save_custom(model, char_to_idx, idx_to_char, cfg, step, epoch, save_dir):
    """Save CustomVLM checkpoint to the given directory (overwrites previous)."""
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
        "config": {
            "block_size": cfg.block_size,
            "text_embed_dim": cfg.text_embed_dim,
            "n_heads": cfg.n_heads,
            "n_layers": cfg.n_layers,
            "vocab_size": len(char_to_idx),
        },
        "step": step, "epoch": epoch,
    }, os.path.join(save_dir, "custom_vlm.pt"))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train VLM — BLIP | ViT-GPT2 | GIT | Custom")
    parser.add_argument(
        "--model", type=str, default="blip",
        choices=["blip", "vit_gpt2", "git", "custom"],
        help="Which architecture to train",
    )
    args = parser.parse_args()

    cfg = CFG.load_for_model(args.model)
    device = get_device()
    print(f"✅ Device: {device}")
    print(f"✅ Config: {args.model} | epochs={cfg.epochs} | lr={cfg.lr} | "
          f"batch_size={cfg.batch_size} | max_target_len={cfg.max_target_len}")
    print(f"✅ Output: {cfg.output_root}/{args.model}/")

    # ── Custom VLM has its own dedicated loop ──────────────────────────────
    if args.model == "custom":
        train_custom_vlm(cfg, device)
        return

    # ── HuggingFace Models ─────────────────────────────────────────────────
    latest_dir, best_dir = get_output_paths(cfg, args.model)

    processor = None
    tokenizer = None

    if args.model == "blip":
        model, processor = get_blip_model(cfg, device)
        train_loader, val_loader = get_dataloaders(cfg, processor)

        def save_latest_fn(step, epoch):
            blip_save(model, processor, None, None, step, epoch, cfg.__dict__, latest_dir)

        def save_best_fn(step, epoch):
            blip_save(model, processor, None, None, step, epoch, cfg.__dict__, best_dir)

    elif args.model == "vit_gpt2":
        model, processor, tokenizer = get_vit_gpt2_model(cfg, device)
        train_loader, val_loader = get_dataloaders_for_model(cfg, "vit_gpt2", processor, tokenizer)

        def save_latest_fn(step, epoch):
            vit_gpt2_save(model, processor, tokenizer, None, None, step, epoch, cfg.__dict__, latest_dir)

        def save_best_fn(step, epoch):
            vit_gpt2_save(model, processor, tokenizer, None, None, step, epoch, cfg.__dict__, best_dir)

    elif args.model == "git":
        model, processor = get_git_model(cfg, device)
        train_loader, val_loader = get_dataloaders_for_model(cfg, "git", processor)

        def save_latest_fn(step, epoch):
            git_save(model, processor, None, None, step, epoch, cfg.__dict__, latest_dir)

        def save_best_fn(step, epoch):
            git_save(model, processor, None, None, step, epoch, cfg.__dict__, best_dir)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = math.ceil(len(train_loader) / cfg.grad_accum) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"✅ Update steps: {total_steps} | Warmup: {warmup_steps}")

    run_training_loop(model, optimizer, scheduler, train_loader, val_loader, cfg,
                      save_latest_fn=save_latest_fn,
                      save_best_fn=save_best_fn,
                      model_name=args.model.upper(),
                      processor=processor, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
