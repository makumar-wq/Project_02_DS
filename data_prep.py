"""
data_prep.py
============
Unified data loading for all VLM architectures:
  - BLIP         → BlipProcessor
  - ViT-GPT2     → ViTImageProcessor + GPT-2 tokenizer
  - GIT          → AutoProcessor  
  - Custom VLM   → ViTImageProcessor + character-level tokenizer

Data Preparation Strategies (controlled via cfg.caption_strategy):
  'raw'      — any random caption (no filtering)
  'filtered' — captions between cfg.caption_min_words and cfg.caption_max_words
  'short'    — captions ≤ cfg.caption_min_words words
  'long'     — captions ≥ cfg.caption_max_words words
  'mixed'    — randomly choose among short / medium / long each call
"""

import random
import aiohttp
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Seeding
# ─────────────────────────────────────────────────────────────────────────────

def seed_all(seed: int):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# BLIP DataLoader (original, kept for backward-compat)
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(cfg, processor):
    """
    Backward-compatible BLIP dataloader.
    Uses BlipProcessor to build pixel_values + input_ids + labels.
    """
    seed_all(cfg.seed)

    print(f"Loading dataset: {cfg.dataset_id}...")
    ds = load_dataset(
        cfg.dataset_id,
        storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}},
    )

    train_split = "train"
    val_split = "validation" if "validation" in ds else ("val" if "val" in ds else "train")

    train_ds = ds[train_split].shuffle(seed=cfg.seed).select(
        range(min(cfg.train_samples, len(ds[train_split])))
    )
    val_ds = ds[val_split].shuffle(seed=cfg.seed + 1).select(
        range(min(cfg.val_samples, len(ds[val_split])))
    )

    print(f"✅ Training samples: {len(train_ds)} | Validation samples: {len(val_ds)}")

    def collate_fn(examples):
        images = [ex["image"].convert("RGB") for ex in examples]
        captions = []
        for ex in examples:
            caps = [c for c in ex["captions"] if len(c.split()) > 3] or ex["captions"]
            captions.append(random.choice(caps))

        encoding = processor(
            images=images,
            text=captions,
            padding="max_length",
            truncation=True,
            max_length=cfg.max_target_len,
            return_tensors="pt",
        )
        encoding["labels"] = encoding["input_ids"].clone()
        return encoding

    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Unified HuggingFace Model DataLoader (BLIP / ViT-GPT2 / GIT)
# ─────────────────────────────────────────────────────────────────────────────
# ───────────────────────────────────────────────────────────────────────────────
# Caption Quality Filtering
# ───────────────────────────────────────────────────────────────────────────────

def filter_low_quality_captions(captions: list, min_words: int = 5,
                                max_words: int = 25) -> list:
    """
    Filter captions to only those within the specified word count range.

    Args:
        captions  : list of caption strings
        min_words : minimum word count (inclusive)
        max_words : maximum word count (inclusive)

    Returns:
        filtered list; may be empty if no captions pass the filter
    """
    return [
        c for c in captions
        if min_words <= len(c.split()) <= max_words
    ]


def pick_caption_by_strategy(captions: list, strategy: str = "filtered",
                              min_words: int = 5, max_words: int = 25) -> str:
    """
    Pick one caption from the list using the specified strategy.

    Strategies:
        'raw'      — random choice with no filter
        'filtered' — random from captions in [min_words, max_words]; fallback raw
        'short'    — random from captions ≤ min_words words; fallback raw
        'long'     — random from captions ≥ max_words words; fallback raw
        'mixed'    — each call randomly picks one of the above strategies

    Returns:
        one caption string
    """
    if strategy == "mixed":
        strategy = random.choice(["filtered", "short", "long"])

    if strategy == "raw":
        return random.choice(captions)

    elif strategy == "filtered":
        pool = filter_low_quality_captions(captions, min_words, max_words)
        return random.choice(pool) if pool else random.choice(captions)

    elif strategy == "short":
        pool = [c for c in captions if len(c.split()) <= min_words]
        return random.choice(pool) if pool else random.choice(captions)

    elif strategy == "long":
        pool = [c for c in captions if len(c.split()) >= max_words]
        return random.choice(pool) if pool else random.choice(captions)

    else:
        # Treat unknown strategy as filtered
        pool = filter_low_quality_captions(captions, min_words, max_words)
        return random.choice(pool) if pool else random.choice(captions)



def _pick_caption(example, cfg=None):
    """
    Pick one caption using cfg.caption_strategy (default: 'filtered').
    Falls back to any caption > 3 words if cfg is None.
    """
    if cfg is None:
        caps = [c for c in example["captions"] if len(c.split()) > 3]
        return random.choice(caps) if caps else random.choice(example["captions"])
    return pick_caption_by_strategy(
        example["captions"],
        strategy=getattr(cfg, "caption_strategy", "filtered"),
        min_words=getattr(cfg, "caption_min_words", 5),
        max_words=getattr(cfg, "caption_max_words", 25),
    )


def get_dataloaders_for_model(cfg, model_type: str, processor, tokenizer=None):
    """
    Unified dataloader factory for BLIP, ViT-GPT2, and GIT.

    Args:
        cfg         : CFG dataclass
        model_type  : 'blip' | 'vit_gpt2' | 'git'
        processor   : image processor / AutoProcessor
        tokenizer   : text tokenizer (required only for 'vit_gpt2')

    Returns:
        train_loader, val_loader
    """
    seed_all(cfg.seed)

    print(f"Loading dataset ({model_type}): {cfg.dataset_id}...")
    ds = load_dataset(
        cfg.dataset_id,
        storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}},
    )

    train_split = "train"
    val_split = "validation" if "validation" in ds else ("val" if "val" in ds else "train")

    train_ds = ds[train_split].shuffle(seed=cfg.seed).select(
        range(min(cfg.train_samples, len(ds[train_split])))
    )
    val_ds = ds[val_split].shuffle(seed=cfg.seed + 1).select(
        range(min(cfg.val_samples, len(ds[val_split])))
    )

    print(f"✅ Training: {len(train_ds)} | Validation: {len(val_ds)}")

    if model_type == "blip":
        def collate_fn(examples):
            images = [ex["image"].convert("RGB") for ex in examples]
            captions = [_pick_caption(ex) for ex in examples]
            encoding = processor(
                images=images, text=captions,
                padding="max_length", truncation=True,
                max_length=cfg.max_target_len, return_tensors="pt",
            )
            encoding["labels"] = encoding["input_ids"].clone()
            return encoding

    elif model_type == "vit_gpt2":
        assert tokenizer is not None, "tokenizer required for vit_gpt2"
        def collate_fn(examples):
            images = [ex["image"].convert("RGB") for ex in examples]
            captions = [_pick_caption(ex) for ex in examples]
            pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]
            text_enc = tokenizer(
                captions, padding="max_length", truncation=True,
                max_length=cfg.max_target_len, return_tensors="pt",
            )
            labels = text_enc["input_ids"].clone()
            labels[labels == tokenizer.pad_token_id] = -100
            return {
                "pixel_values": pixel_values,
                "labels": labels,
                "decoder_attention_mask": text_enc["attention_mask"],
            }

    elif model_type == "git":
        def collate_fn(examples):
            images = [ex["image"].convert("RGB") for ex in examples]
            captions = [_pick_caption(ex) for ex in examples]
            encoding = processor(
                images=images, text=captions,
                padding="max_length", truncation=True,
                max_length=cfg.max_target_len, return_tensors="pt",
            )
            labels = encoding["input_ids"].clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100
            encoding["labels"] = labels
            return encoding

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Custom VLM DataLoader (Character-Level Tokenization)
# ─────────────────────────────────────────────────────────────────────────────

class COCOCharDataset(Dataset):
    """
    Maps COCO images → (pixel_values, text_input_ids, text_targets)
    using a character-level vocabulary built from the Shakespeare corpus.
    """

    def __init__(self, hf_dataset, image_processor, char_to_idx, max_target_len):
        self.ds = hf_dataset
        self.image_processor = image_processor
        self.char_to_idx = char_to_idx
        self.max_target_len = max_target_len
        self.unk_idx = char_to_idx.get(" ", 0)

    def _encode_text(self, text):
        """Encode a string to a fixed-length char index tensor."""
        ids = [self.char_to_idx.get(c, self.unk_idx) for c in text[:self.max_target_len]]
        # Pad with 0s if shorter
        ids += [0] * (self.max_target_len - len(ids))
        return ids

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        image = ex["image"].convert("RGB")
        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        # Pick one caption
        caps = [c for c in ex["captions"] if len(c.split()) > 3] or ex["captions"]
        caption = random.choice(caps).lower()

        src_ids = self._encode_text(caption[:-1])   # input: all but last char
        tgt_ids = self._encode_text(caption[1:])    # target: shifted right by 1

        return {
            "pixel_values": pixel_values,
            "text_input_ids": torch.tensor(src_ids, dtype=torch.long),
            "text_targets": torch.tensor(tgt_ids, dtype=torch.long),
        }


def get_custom_vlm_dataloader(cfg, char_to_idx):
    """
    Returns (train_loader, val_loader) for the Custom VLM using COCO images
    and character-level tokenization.

    Requires the ViT image processor separately.
    """
    from transformers import ViTImageProcessor

    seed_all(cfg.seed)

    image_processor = ViTImageProcessor.from_pretrained(cfg.vit_encoder_id, use_fast=True)

    print(f"Loading dataset (Custom VLM): {cfg.dataset_id}...")
    ds = load_dataset(
        cfg.dataset_id,
        storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}},
    )

    train_split = "train"
    val_split = "validation" if "validation" in ds else ("val" if "val" in ds else "train")

    train_hf = ds[train_split].shuffle(seed=cfg.seed).select(
        range(min(cfg.train_samples, len(ds[train_split])))
    )
    val_hf = ds[val_split].shuffle(seed=cfg.seed + 1).select(
        range(min(cfg.val_samples, len(ds[val_split])))
    )

    train_ds = COCOCharDataset(train_hf, image_processor, char_to_idx, cfg.max_target_len)
    val_ds = COCOCharDataset(val_hf, image_processor, char_to_idx, cfg.max_target_len)

    print(f"✅ Custom VLM — Training: {len(train_ds)} | Validation: {len(val_ds)}")

    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader
