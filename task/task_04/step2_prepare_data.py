"""
step2_prepare_data.py
======================
Task 4 — Component 2: Load COCO validation data for diversity analysis
and build style-labelled caption sets for steering vector extraction.

Two public APIs
---------------
1. load_val_data(processor, n=200, batch_size=4) -> DataLoader
   Loads the first ``n`` MS-COCO 2017 validation images.
   Each batch yields {"pixel_values": Tensor, "captions": list[str], "image_ids": list[int]}.

2. build_style_sets(n=500) -> dict[str, list[str]]
   Loads COCO validation captions and partitions them by word-count length:
       short    : ≤ 8 words
       medium   : 9–14 words
       detailed : ≥ 15 words
   Returns {"short": [...], "medium": [...], "detailed": [...]}

Standalone usage
----------------
    export PYTHONPATH=.
    venv/bin/python task/task_04/step2_prepare_data.py
"""

import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ─────────────────────────────────────────────────────────────────────────────
# COCO Diversity Dataset
# ─────────────────────────────────────────────────────────────────────────────

class COCODiversityDataset(Dataset):
    """Wraps COCO validation images for diversity analysis.

    Each item returns:
        pixel_values : (3, H, W) float tensor (processor-normalised)
        caption      : str  — first reference caption from COCO
        image_id     : int
    """

    def __init__(self, hf_dataset, processor, max_n: int = 200):
        self.data      = hf_dataset.select(range(min(max_n, len(hf_dataset))))
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item   = self.data[idx]
        image  = item["image"].convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        # First COCO reference caption
        captions = item.get("captions", item.get("sentences_raw", [""]))
        caption  = captions[0] if isinstance(captions, list) else captions

        return {
            "pixel_values": pixel_values,
            "caption": caption,
            "image_id": idx,
            "image": image,
        }


def _collate(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    captions     = [b["caption"]  for b in batch]
    image_ids    = [b["image_id"] for b in batch]
    images       = [b["image"]    for b in batch]
    return {
        "pixel_values": pixel_values, 
        "captions": captions, 
        "image_ids": image_ids,
        "images": images
    }


def load_val_data(processor, n: int = 200, batch_size: int = 4):
    """
    Load the first ``n`` COCO 2017 validation images as a DataLoader.

    Args:
        processor  : BlipProcessor (from step1_load_model)
        n          : number of images to load (default 200)
        batch_size : images per batch (default 4)

    Returns:
        torch.utils.data.DataLoader
    """
    from datasets import load_dataset
    print("=" * 62)
    print("  Task 4 — Step 2: Prepare COCO Validation Data")
    print("=" * 62)
    print(f"  Loading COCO 2017 validation split (first {n} images)…")

    hf_ds = load_dataset("whyen-wang/coco_captions", split="validation", trust_remote_code=True)
    print(f"  ✅ COCO val split loaded ({len(hf_ds):,} total images)")

    dataset = COCODiversityDataset(hf_ds, processor, max_n=n)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=0,
    )

    print(f"  Dataset size : {len(dataset)} images")
    print(f"  Batch size   : {batch_size}")
    print(f"  Num batches  : {len(loader)}")
    print("=" * 62)
    return loader


# ─────────────────────────────────────────────────────────────────────────────
# Style caption sets (for steering vector extraction)
# ─────────────────────────────────────────────────────────────────────────────

def build_style_sets(n: int = 500) -> dict:
    """
    Partition the first ``n`` COCO validation captions into three style buckets
    based on word count:

        short    : ≤ 8 words
        medium   : 9–14 words
        detailed : ≥ 15 words

    Args:
        n: Maximum number of COCO captions to scan

    Returns:
        dict with keys 'short', 'medium', 'detailed', each a list[str].
    """
    from datasets import load_dataset
    print("  Building style caption sets from COCO val …")

    hf_ds  = load_dataset("whyen-wang/coco_captions", split="validation", trust_remote_code=True)
    short, medium, detailed = [], [], []

    for item in hf_ds.select(range(min(n, len(hf_ds)))):
        captions = item.get("captions", item.get("sentences_raw", []))
        if isinstance(captions, str):
            captions = [captions]
        for cap in captions:
            wc = len(cap.split())
            if wc <= 8:
                short.append(cap)
            elif wc <= 14:
                medium.append(cap)
            else:
                detailed.append(cap)

    print(f"  Style sets: short={len(short)}, medium={len(medium)}, detailed={len(detailed)}")
    return {"short": short, "medium": medium, "detailed": detailed}


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from step1_load_model import load_model
    model, processor, device = load_model()

    # Test data loader
    loader = load_val_data(processor, n=20, batch_size=4)
    batch  = next(iter(loader))
    print(f"\n  Sample batch — pixel_values shape : {batch['pixel_values'].shape}")
    print(f"  Sample captions : {batch['captions'][:2]}")

    # Test style sets
    sets = build_style_sets(n=100)
    for style, caps in sets.items():
        sample = caps[0] if caps else "(none)"
        print(f"  {style:8s} ({len(caps):3d} caps)  e.g. '{sample[:60]}'")
