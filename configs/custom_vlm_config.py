"""
configs/custom_vlm_config.py
=============================
Custom VLM (Visual Prefix-Tuning / Shakespeare Decoder) training configuration.

This model has unique hyperparameters for the character-level decoder:
  - block_size controls the maximum text sequence length
  - text_embed_dim, n_heads, n_layers define the decoder architecture
  - max_target_len is higher (128) because char-level tokens are finer-grained
"""

from dataclasses import dataclass
from .base_config import BaseConfig


@dataclass
class CustomVLMConfig(BaseConfig):
    # ─── Model ──────────────────────────────────────────────────────────────
    vlm_type: str = "custom"
    vit_encoder_id: str = "google/vit-base-patch16-224-in21k"

    # ─── Training Overrides ─────────────────────────────────────────────────
    epochs: int = 15
    lr: float = 1e-4
    batch_size: int = 16
    max_target_len: int = 128     # char-level needs more length than subword

    # ─── Custom Decoder Architecture ────────────────────────────────────────
    shakespeare_file: str = "./input.txt"
    shakespeare_weights_path: str = "./shakespeare_transformer.pt"
    text_embed_dim: int = 384
    n_heads: int = 8
    n_layers: int = 8
    block_size: int = 256
    dropout: float = 0.1
