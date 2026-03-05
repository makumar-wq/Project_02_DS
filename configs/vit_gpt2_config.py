"""
configs/vit_gpt2_config.py
===========================
ViT-GPT2 (Standard Cross-Attention) training configuration.
"""

from dataclasses import dataclass
from .base_config import BaseConfig


@dataclass
class ViTGPT2Config(BaseConfig):
    # ─── Model ──────────────────────────────────────────────────────────────
    vlm_type: str = "vit_gpt2"
    model_id: str = "nlpconnect/vit-gpt2-image-captioning"

    # ─── Training Overrides ─────────────────────────────────────────────────
    epochs: int = 3
    lr: float = 2e-5
