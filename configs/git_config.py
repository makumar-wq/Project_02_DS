"""
configs/git_config.py
======================
GIT (Zero Cross-Attention / Self-Attention Prefix) training configuration.
"""

from dataclasses import dataclass
from .base_config import BaseConfig


@dataclass
class GitConfig(BaseConfig):
    # ─── Model ──────────────────────────────────────────────────────────────
    vlm_type: str = "git"
    model_id: str = "microsoft/git-base-coco"

    # ─── Training Overrides ─────────────────────────────────────────────────
    epochs: int = 3
    lr: float = 2e-5
