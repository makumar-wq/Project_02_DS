"""
models/git_tuner.py
===================
Baseline 2 — Zero Cross-Attention / Self-Attention Prefix (GIT)

Architecture: GIT (Generative Image-to-Text) abandons cross-attention entirely.
It concatenates image patch embeddings directly in front of the text tokens and
runs a single causal self-attention Transformer over the combined sequence.

There is NO cross-attention block. The model learns to fuse modalities purely
through self-attention across a unified image+text token sequence. This makes
the ablation masks work differently — we control which image tokens are
prepended to the sequence rather than using encoder_attention_mask.
"""

import os
import torch
from transformers import AutoProcessor, AutoModelForCausalLM


def get_git_model(cfg, device):
    """
    Load microsoft/git-base-coco with gradient checkpointing.
    GIT uses AutoModelForCausalLM interface.
    """
    model_id = cfg.git_model_id

    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    try:
        model.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing enabled (GIT)")
    except Exception as e:
        print(f"⚠️  Gradient checkpointing failed: {e}")

    model.config.use_cache = False
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✅ GIT loaded on {device}: {model_id} ({n_params:.1f}M params)")
    return model, processor


def generate_caption(model, processor, image_pil, device,
                     max_new_tokens=32, num_beams=4):
    """
    Generate a caption for a single PIL image using GIT.

    Note: GIT has no encoder_attention_mask concept (no cross-attention).
    Ablation for GIT is handled upstream by modifying the pixel_values
    (e.g., masking image regions) before passing to the model, OR by
    returning a note that GIT is not compatible with encoder-mask ablations.
    """
    model.eval()
    inputs = processor(images=image_pil, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )

    caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return caption


def save_ckpt(model, processor, optimizer, scheduler,
              step, epoch, cfg_dict, path):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    processor.save_pretrained(path)

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
    print(f"✅ GIT checkpoint saved: {path}")
