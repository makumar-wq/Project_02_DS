"""
models/vit_gpt2_tuner.py
========================
Baseline 1 — Standard Cross-Attention (ViT-GPT2)

Architecture: Every generated text token in the GPT-2 decoder attends to ALL
197 ViT patch embeddings via explicit cross-attention blocks injected between
each GPT-2 self-attention layer.

This is the "brute-force" cross-attention baseline: no restrictions, no pooling.
"""

import os
import torch
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
)


def get_vit_gpt2_model(cfg, device):
    """
    Load the VisionEncoderDecoderModel (ViT-GPT2) with:
    - Gradient checkpointing enabled
    - use_cache=False (required with grad checkpointing)
    - Proper pad/bos/eos tokens set for GPT-2
    """
    model_id = cfg.vit_gpt2_model_id

    processor = ViTImageProcessor.from_pretrained(model_id, use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # GPT-2 has no pad token by default — use eos as pad
    tokenizer.pad_token = tokenizer.eos_token

    model = VisionEncoderDecoderModel.from_pretrained(model_id)
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # Memory optimizations
    try:
        model.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing enabled (ViT-GPT2)")
    except Exception as e:
        print(f"⚠️  Gradient checkpointing failed: {e}")

    model.config.use_cache = False

    # Resize images to cfg.image_size
    try:
        processor.size = {"height": cfg.image_size, "width": cfg.image_size}
        print(f"✅ Image size set to {cfg.image_size}px")
    except Exception as e:
        print(f"⚠️  Could not set image size: {e}")

    model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✅ ViT-GPT2 loaded on {device}: {model_id} ({n_params:.1f}M params)")

    return model, processor, tokenizer


def generate_caption(model, processor, tokenizer, image_pil, device,
                     max_new_tokens=32, num_beams=4,
                     encoder_attention_mask=None):
    """
    Generate a caption for a single PIL image.

    encoder_attention_mask: (1, num_patches) allows ablation-mode masking.
    If None, defaults to full attention (all 1s).
    """
    model.eval()
    inputs = processor(images=image_pil, return_tensors="pt").to(device)
    pixel_values = inputs["pixel_values"]

    gen_kwargs = dict(
        pixel_values=pixel_values,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )
    if encoder_attention_mask is not None:
        gen_kwargs["attention_mask"] = encoder_attention_mask.to(device)

    with torch.no_grad():
        output_ids = model.generate(**gen_kwargs)

    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption


def save_ckpt(model, processor, tokenizer, optimizer, scheduler,
              step, epoch, cfg_dict, path):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    processor.save_pretrained(path)
    tokenizer.save_pretrained(path)

    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler": scheduler.state_dict() if scheduler else None,
            "cfg": cfg_dict,
        },
        os.path.join(path, "train_state.pt"),
    )
    print(f"✅ ViT-GPT2 checkpoint saved: {path}")
