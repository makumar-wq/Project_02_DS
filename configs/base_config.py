"""
configs/base_config.py
======================
Shared configuration settings inherited by all model-specific configs.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class BaseConfig:
    # ─── Dataset ────────────────────────────────────────────────────────────
    dataset_id: str = "whyen-wang/coco_captions"
    train_samples: int = 15000
    val_samples: int = 1500
    seed: int = 42

    # ─── Image / Sequence ───────────────────────────────────────────────────
    image_size: int = 224
    max_target_len: int = 32

    # ─── Training (defaults, overridden per model) ──────────────────────────
    batch_size: int = 8
    grad_accum: int = 4
    epochs: int = 3
    lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0

    # ─── DataLoader ─────────────────────────────────────────────────────────
    num_workers: int = 0          # 0 is safest on macOS MPS
    log_every: int = 10

    # ─── Output ─────────────────────────────────────────────────────────────
    output_root: str = "./outputs"   # all checkpoints: outputs/{model}/best/ & latest/

    # ─── Ablation / Evaluation ──────────────────────────────────────────────
    ablation_mode: Literal["baseline", "random_dropout", "center_focus", "squint"] = "baseline"
    dropout_ratio: float = 0.50

    # ─── Data Preparation Strategy ──────────────────────────────────
    # 'raw'      — any random caption (no filtering)
    # 'filtered' — captions between caption_min_words and caption_max_words
    # 'short'    — captions <= caption_min_words words
    # 'long'     — captions >= caption_max_words words
    # 'mixed'    — randomly switch between short, medium, and long each batch
    caption_strategy: str = "filtered"    # recommended default
    caption_min_words: int = 5
    caption_max_words: int = 25
