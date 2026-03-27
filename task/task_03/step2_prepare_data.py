"""
step2_prepare_data.py
======================
Task 3 — Component 2: Prepare 500 COCO validation images for inference.

Loads 500 images from the MS-COCO 2017 validation split (via HuggingFace
Datasets) and wraps them in a standard PyTorch DataLoader.

Public API
----------
    load_val_data(processor, n=500, batch_size=8, seed=42)
        -> torch.utils.data.DataLoader

Each batch yields a dict:
    {
        "pixel_values" : FloatTensor (B, 3, 384, 384),
        "labels"       : LongTensor  (B, max_len),      # reference caption ids
        "captions"     : list[str]                       # raw reference strings
    }

Standalone usage
----------------
    export PYTHONPATH=.
    venv/bin/python task/task_03/step2_prepare_data.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BlipProcessor


# ─────────────────────────────────────────────────────────────────────────────
# Dataset wrapper
# ─────────────────────────────────────────────────────────────────────────────

DATASET_ID = "nlphuji/flickr30k"   # fallback if COCO unavailable
COCO_ID    = "phiyodr/coco2017"


class COCOValDataset(Dataset):
    """
    Wraps a HuggingFace dataset split into a torch Dataset.

    Args:
        hf_dataset : HuggingFace Dataset object with 'image' and 'captions' fields.
        processor  : BlipProcessor instance.
        max_len    : Maximum tokenization length for reference captions.
    """

    def __init__(self, hf_dataset, processor: BlipProcessor, max_len: int = 64):
        self.data      = hf_dataset
        self.processor = processor
        self.max_len   = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        image   = example["image"].convert("RGB")

        # Pick the first reference caption
        captions = example.get("captions", example.get("caption", ["<no caption>"]))
        if isinstance(captions, str):
            captions = [captions]
        caption = captions[0]

        enc = self.processor(
            images=image,
            text=caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )

        return {
            "pixel_values": enc["pixel_values"].squeeze(0),   # (3, H, W)
            "labels":       enc["input_ids"].squeeze(0),       # (max_len,)
            "caption":      caption,
        }


def _collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels":       torch.stack([b["labels"] for b in batch]),
        "captions":     [b["caption"] for b in batch],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public loader
# ─────────────────────────────────────────────────────────────────────────────

def load_val_data(
    processor: BlipProcessor,
    n: int         = 500,
    batch_size: int = 8,
    seed: int       = 42,
    max_len: int    = 64,
) -> DataLoader:
    """
    Download and prepare n COCO validation images.

    Falls back to Flickr30k if COCO is unavailable (e.g. firewall/proxy).

    Args:
        processor  : BlipProcessor (from step1_load_model)
        n          : Number of validation images to use (default 500)
        batch_size : DataLoader batch size
        seed       : Random seed for reproducible shuffle
        max_len    : Max caption token length for labels

    Returns:
        DataLoader that yields batches with keys:
            pixel_values, labels, captions
    """
    from datasets import load_dataset

    print("=" * 60)
    print("  Task 3 — Step 2: Prepare Validation Data")
    print("=" * 60)
    print(f"  Target images : {n}")
    print(f"  Batch size    : {batch_size}")

    # ── Try COCO first ────────────────────────────────────────────────────────
    ds = None
    try:
        print(f"  Loading dataset: {COCO_ID} ...")
        raw = load_dataset(COCO_ID, split="validation", trust_remote_code=True)
        ds  = raw.shuffle(seed=seed).select(range(min(n, len(raw))))
        print(f"  ✅ COCO loaded  ({len(ds)} images)")
    except Exception as e:
        print(f"  ⚠️  COCO unavailable ({e}). Falling back to Flickr30k …")

    # ── Fallback to Flickr30k ─────────────────────────────────────────────────
    if ds is None:
        raw = load_dataset(DATASET_ID, split="test", trust_remote_code=True)
        ds  = raw.shuffle(seed=seed).select(range(min(n, len(raw))))
        print(f"  ✅ Flickr30k loaded  ({len(ds)} images)")

    dataset    = COCOValDataset(ds, processor, max_len=max_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=_collate_fn,
    )

    print(f"  Batches       : {len(dataloader)}")
    print("=" * 60)
    return dataloader


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from step1_load_model import load_model

    _, processor, _ = load_model()
    loader = load_val_data(processor, n=500, batch_size=8)

    # Peek at first batch
    batch = next(iter(loader))
    print(f"\n✅  DataLoader ready!")
    print(f"   pixel_values shape : {batch['pixel_values'].shape}")
    print(f"   labels shape       : {batch['labels'].shape}")
    print(f"   Sample caption     : {batch['captions'][0][:80]}")
    print(f"\nImport in notebooks:")
    print("  from task.task_03.step2_prepare_data import load_val_data")
