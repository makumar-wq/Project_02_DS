"""
configs/blip_config.py
=======================
BLIP (Multimodal Mixture Attention) training configuration.
"""

from dataclasses import dataclass
from .base_config import BaseConfig


@dataclass
class BlipConfig(BaseConfig):
    # ─── Model ──────────────────────────────────────────────────────────────
    vlm_type: str = "blip"
    model_id: str = "Salesforce/blip-image-captioning-base"

    # ─── Training Overrides ─────────────────────────────────────────────────
    epochs: int = 3
    lr: float = 1e-5
    train_samples: int = 30000
    val_samples: int = 2000
    batch_size: int = 16
