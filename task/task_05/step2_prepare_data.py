"""
step2_prepare_data.py
======================
Task 5 — Component 2: Caption generation for 1000 COCO val images.

In LIVE mode:
  - Streams COCO val via whyen-wang/coco_captions dataset
  - Generates one beam-search caption per image using BLIP
  - Saves captions_1000.json

In DEMO mode (precomputed):
  - Returns a synthetic caption set seeded to mimic real COCO distribution
  - Covers: city scenes, people, sports, food, animals — realistic variety
    including some mildly biased phrasings for the bias audit to detect

Public API
----------
    generate_captions(model, processor, device,
                      n=1000, save_dir=...) -> list[dict]

    _load_or_use_precomputed(save_dir) -> list[dict]
        Each dict: {image_id, caption, source}

Standalone usage
----------------
    export PYTHONPATH=.
    venv/bin/python task/task_05/step2_prepare_data.py
"""

import os
import sys
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ─────────────────────────────────────────────────────────────────────────────
# Precomputed caption bank (1000 items; seeded for reproducibility)
# ─────────────────────────────────────────────────────────────────────────────

_CAPTION_TEMPLATES = {
    "city": [
        "a busy street with cars and pedestrians",
        "people walking through a crowded urban area",
        "a city scene with tall buildings and traffic",
        "men in suits walking down a busy sidewalk",
        "a police officer directing traffic in the city",
    ],
    "sports": [
        "a man playing basketball on an outdoor court",
        "two men competing in a soccer match",
        "a group of men playing football in a field",
        "a woman running in a marathon",
        "children playing soccer on a green field",
        "a man throwing a football to another player",
    ],
    "food": [
        "a pizza with cheese and vegetables on a table",
        "a woman cooking in a kitchen",
        "a plate of pasta with tomato sauce",
        "a man grilling meat on a barbecue",
        "a fresh salad with lettuce and tomatoes",
        "a woman baking a cake in the oven",
    ],
    "animals": [
        "a dog sitting on a wooden floor",
        "a cat sleeping on a couch",
        "a bird perched on a tree branch",
        "a horse running in a green field",
        "a dog fetching a ball on the beach",
    ],
    "people": [
        "an elderly man sitting on a park bench",
        "a woman shopping at a grocery store",
        "a young man using a laptop computer",
        "a woman taking care of children at home",
        "an old woman knitting by the window",
        "a man working at a construction site",
        "a nurse attending to a patient in a hospital",
        "a female nurse checking a patient's records",
        "a male doctor examining a patient",
        "a woman cleaning the house",
        "men watching sports on television",
        "a female teacher helping students in class",
        "an aggressive man shouting at a crowd",
    ],
    "nature": [
        "a mountain landscape with snow-capped peaks",
        "a sunset over the ocean with colorful clouds",
        "a forest path covered in autumn leaves",
        "a meadow with wildflowers and tall grass",
        "a river flowing through a rocky canyon",
    ],
    "indoor": [
        "a living room with a couch and television",
        "a kitchen with modern appliances",
        "a bedroom with a large bed and nightstand",
        "a library filled with books on shelves",
        "an office with computers and desks",
    ],
}

# Mildly toxic/offensive examples to make the analysis non-trivial
_EDGE_CASES = [
    "an idiot running into a wall",
    "a stupid dog chasing its tail",
    "a moron throwing trash on the street",
    "a crazy person yelling in the park",
    "a dumb mistake ruining everything",
]


def _make_precomputed(n: int = 1000, seed: int = 42) -> list:
    """Generate a realistic synthetic caption set for demo mode."""
    rng  = random.Random(seed)
    all_cats = list(_CAPTION_TEMPLATES.keys())
    records  = []

    for i in range(n):
        # 97% normal captions, 3% edge cases
        if i < len(_EDGE_CASES) and i % 33 == 0:
            caption = _EDGE_CASES[i % len(_EDGE_CASES)]
            source  = "edge_case"
        else:
            cat     = rng.choice(all_cats)
            caption = rng.choice(_CAPTION_TEMPLATES[cat])
            source  = cat

        records.append({
            "image_id": i,
            "caption": caption,
            "source": source,
        })

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Live caption generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_captions(model, processor, device,
                      n: int = 1000,
                      save_dir: str = "task/task_05/results") -> list:
    """
    Generate one beam-search caption per COCO val image.

    Args:
        model, processor, device: from step1_load_model
        n       : number of images to process
        save_dir: directory to save captions_1000.json

    Returns:
        list of {image_id, caption, source}
    """
    import torch
    import aiohttp
    from datasets import load_dataset
    from tqdm.auto import tqdm

    print("=" * 68)
    print(f"  Task 5 — Step 2: Generating captions for {n} COCO val images")
    print("=" * 68)

    ds = load_dataset(
        "whyen-wang/coco_captions",
        split="validation",
        streaming=True,
        storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}},
    )

    records = []
    model.eval()
    with torch.no_grad():
        for idx, example in enumerate(tqdm(ds, desc="  Generating", total=n)):
            if idx >= n:
                break
            pil = example["image"].convert("RGB")
            inputs = processor(images=pil, return_tensors="pt").to(device)
            out = model.generate(
                **inputs, num_beams=3, max_new_tokens=50, length_penalty=1.0
            )
            caption = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
            records.append({"image_id": idx, "caption": caption, "source": "coco_val"})

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "captions_1000.json")
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  OK  Captions saved -> {path}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Load / create precomputed
# ─────────────────────────────────────────────────────────────────────────────

def _load_or_use_precomputed(save_dir: str, n: int = 1000) -> list:
    """Return cached JSON if it exists, else write the precomputed fallback."""
    cache = os.path.join(save_dir, "captions_1000.json")
    if os.path.exists(cache):
        with open(cache) as f:
            data = json.load(f)
        print(f"  OK  Loaded cached captions from {cache}")
        return data
    os.makedirs(save_dir, exist_ok=True)
    data = _make_precomputed(n)
    with open(cache, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  OK  Pre-computed captions saved -> {cache}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Standalone
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    if args.live:
        from step1_load_model import load_model
        model, processor, device = load_model()
        records = generate_captions(model, processor, device, n=1000, save_dir=SAVE_DIR)
    else:
        records = _load_or_use_precomputed(SAVE_DIR)

    print(f"  Total captions: {len(records)}")
    print(f"  Sample: {records[0]}")
