"""
models/blip_tuner.py
====================
Baseline 3 — Multimodal Mixture Attention (BLIP)

Architecture: BLIP's MED (Multimodal Encoder-Decoder) architecture injects
specialized gated cross-attention between self-attention and feed-forward layers.
The visual encoder output (image patch embeddings) is queried by the text decoder
via cross-attention that is applied carefully at each decoder layer.

This module also provides `generate_with_mask()` for inference-time ablation
experiments that manipulate the encoder_attention_mask to test spatial restrictions.
"""

import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration


def get_blip_model(cfg, device):
    """
    Loads BLIP model and processor with MPS and memory optimizations.
    """
    processor = BlipProcessor.from_pretrained(cfg.model_id, use_fast=True)
    model = BlipForConditionalGeneration.from_pretrained(cfg.model_id)

    # Force 224px images for efficiency (especially on Mac/MPS)
    try:
        processor.image_processor.size = {"height": cfg.image_size, "width": cfg.image_size}
        print(f"✅ Image size set to {cfg.image_size}px")
    except Exception as e:
        print(f"⚠️  Could not set image size: {e}")

    # Gradient checkpointing for VRAM efficiency
    try:
        model.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing enabled (BLIP)")
    except Exception as e:
        print(f"⚠️  Gradient checkpointing failed: {e}")

    model.config.use_cache = False  # Must be False with gradient checkpointing
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✅ BLIP loaded on {device}: {cfg.model_id} ({n_params:.1f}M params)")

    return model, processor


def generate_with_mask(model, processor, image_pil=None, device=None,
                       pixel_values=None,
                       encoder_hidden_states=None,
                       encoder_attention_mask=None,
                       max_new_tokens=32, num_beams=4):
    """
    Generate a caption for a single PIL image (or pre-computed tensors) with an ablation mask.

    Ablation modes supported:
      - Baseline:       197 patches visible
      - Random Dropout: 50% spatial patches masked
      - Center-Focus:   Inner 8x8 patches visible
      - Squint:         Requires passing pre-pooled `encoder_hidden_states` of shape (B, 2, C).
    """
    model.eval()
    
    # 1. Get pixel values
    if pixel_values is None and image_pil is not None:
        inputs = processor(images=image_pil, return_tensors="pt").to(device)
        pixel_values = inputs["pixel_values"]
        
    batch_size = pixel_values.shape[0] if pixel_values is not None else encoder_hidden_states.shape[0]
    dev = pixel_values.device if pixel_values is not None else encoder_hidden_states.device
    
    # 2. Extract visual features if not pre-provided (e.g., Squint mode provides them)
    if encoder_hidden_states is None:
        vision_outputs = model.vision_model(pixel_values=pixel_values)
        encoder_hidden_states = vision_outputs[0]
        
    # 3. Handle encoder_attention_mask default (Baseline = all ones)
    if encoder_attention_mask is None:
        encoder_attention_mask = torch.ones(
            encoder_hidden_states.size()[:-1], dtype=torch.long, device=dev
        )
    else:
        encoder_attention_mask = encoder_attention_mask.to(dev)

    # 4. Prepare decoder input IDs (BOS token)
    input_ids = (
        torch.LongTensor([[model.decoder_input_ids, model.config.text_config.eos_token_id]])
        .repeat(batch_size, 1)
        .to(dev)
    )
    input_ids[:, 0] = model.config.text_config.bos_token_id

    # 5. Bypass the outer model.generate() to avoid hardcoded mask conflicts
    with torch.no_grad():
        output_ids = model.text_decoder.generate(
            input_ids=input_ids[:, :-1],
            eos_token_id=model.config.text_config.sep_token_id,
            pad_token_id=model.config.text_config.pad_token_id,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )

    captions = processor.batch_decode(output_ids, skip_special_tokens=True)
    return captions


def save_ckpt(model, processor, optimizer, scheduler, step, epoch, cfg_dict, path):
    """
    Save model weights, processor, and training state.
    """
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
    print(f"✅ BLIP checkpoint saved: {path}")


def load_ckpt(model, optimizer, scheduler, path):
    """
    Load model + optimizer/scheduler from a checkpoint directory.
    """
    loaded_model = BlipForConditionalGeneration.from_pretrained(path)
    model.load_state_dict(loaded_model.state_dict())

    state_path = os.path.join(path, "train_state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location="cpu")
        if optimizer and state.get("optimizer"):
            optimizer.load_state_dict(state["optimizer"])
        if scheduler and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        print(f"✅ Resumed from step {state.get('step', '?')}, epoch {state.get('epoch', '?')}")
        return state.get("step", 0), state.get("epoch", 1)

    print("✅ Model weights loaded, no training state found.")
    return 0, 1
