"""
configs/__init__.py
===================
Config package — exposes a get_config() factory function.
"""

from .base_config import BaseConfig
from .blip_config import BlipConfig
from .vit_gpt2_config import ViTGPT2Config
from .git_config import GitConfig
from .custom_vlm_config import CustomVLMConfig


def get_config(model_type: str):
    """
    Return the appropriate config dataclass for the given model type.

    Args:
        model_type: one of 'blip', 'vit_gpt2', 'git', 'custom'

    Returns:
        Populated config dataclass instance.
    """
    registry = {
        "blip": BlipConfig,
        "vit_gpt2": ViTGPT2Config,
        "git": GitConfig,
        "custom": CustomVLMConfig,
    }
    cls = registry.get(model_type)
    if cls is None:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Choose from: {list(registry.keys())}"
        )
    return cls()
