"""
experiments/
============
Pluggable experiment modules for the VLM Image Captioning pipeline.

  ablation_study.py       — Cross-attention mask ablation (BLIP / ViT-GPT2)
  cross_attention_patterns.py — Architecture comparison table
  parameter_sweep.py      — beam_size / length_penalty / max_length sweep
  data_prep_analysis.py   — Before vs after caption quality filtering
  vqa_experiment.py       — Visual Question Answering demo (BLIP-VQA)
"""

from .ablation_study import (
    build_ablation_mask,
    evaluate_blip_ablation,
    run_ablation_study,
    ABLATION_MODES,
)

__all__ = [
    "build_ablation_mask",
    "evaluate_blip_ablation",
    "run_ablation_study",
    "ABLATION_MODES",
]
