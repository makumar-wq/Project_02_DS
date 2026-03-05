"""
config.py
=========
Backward-compatible configuration wrapper.

This file now delegates to the per-model configs in configs/.
Existing code that does `from config import CFG` will continue to work.

Usage:
    from config import CFG
    cfg = CFG.load_from_env()          # loads default (BLIP) config
    cfg = CFG.load_for_model("git")    # loads GIT-specific config
    cfg.get_model_dir("blip")          # → "./outputs/blip"
"""

import os
from dataclasses import dataclass, field
from typing import Literal

from configs import get_config
from configs.base_config import BaseConfig


@dataclass
class CFG(BaseConfig):
    """
    Master config that merges all fields across all model types.
    This exists for backward compatibility with app.py, eval.py, etc.
    """
    # ─── Model Selection ────────────────────────────────────────────────────
    vlm_type: Literal["blip", "vit_gpt2", "git", "custom"] = "blip"

    # ─── Model IDs (all models so app.py can reference any) ─────────────────
    model_id: str = "Salesforce/blip-image-captioning-base"
    vit_gpt2_model_id: str = "nlpconnect/vit-gpt2-image-captioning"
    git_model_id: str = "microsoft/git-base-coco"
    vit_encoder_id: str = "google/vit-base-patch16-224-in21k"

    # ─── Custom VLM (Shakespeare Decoder) ───────────────────────────────────
    shakespeare_file: str = "./input.txt"
    shakespeare_weights_path: str = "./shakespeare_transformer.pt"
    text_embed_dim: int = 384
    n_heads: int = 8
    n_layers: int = 8
    block_size: int = 256
    dropout: float = 0.1

    # ─── Unified Output ─────────────────────────────────────────────────────
    # All checkpoints go under: outputs/{model}/best/ and outputs/{model}/latest/
    output_root: str = "./outputs"

    def get_model_dir(self, model_name: str) -> str:
        """Return the output directory for a specific model: outputs/{model_name}/"""
        return os.path.join(self.output_root, model_name)

    @classmethod
    def load_from_env(cls):
        """Load the default (backward-compat) config."""
        return cls()

    @classmethod
    def load_for_model(cls, model_type: str):
        """
        Load a model-specific config from configs/ and merge into CFG.

        This lets train.py use optimized per-model hyperparameters while
        keeping the CFG dataclass compatible with the rest of the codebase.
        """
        model_cfg = get_config(model_type)
        base = cls()
        # Overwrite fields that the model config provides
        for field_name in model_cfg.__dataclass_fields__:
            if hasattr(base, field_name):
                setattr(base, field_name, getattr(model_cfg, field_name))
        return base
