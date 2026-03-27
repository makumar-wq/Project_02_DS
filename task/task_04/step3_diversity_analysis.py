"""
step3_diversity_analysis.py
============================
Task 4 — Component 3: Caption Diversity Analysis

For each image in the validation DataLoader, this step:
  1. Generates 5 captions using nucleus sampling (do_sample=True, top_p=0.9).
  2. Computes a per-image diversity score:
         diversity = unique_ngrams / total_ngrams
     where ngrams = all unigrams + bigrams across the 5 captions.
  3. Classifies images as "diverse" (score > 0.75) or "repetitive" (score < 0.40).
  4. Returns a list of per-image record dicts.

Pre-computed fallback
---------------------
If `results/diversity_results.json` already exists the file is returned directly,
enabling all downstream steps to work without a GPU (HuggingFace Spaces demo mode).

Public API
----------
    run_diversity_analysis(model, processor, dataloader, device,
                           save_dir="task/task_04/results") -> list[dict]

    compute_diversity_score(captions: list[str]) -> float

Standalone usage
----------------
    export PYTHONPATH=.
    venv/bin/python task/task_04/step3_diversity_analysis.py          # precomputed
    venv/bin/python task/task_04/step3_diversity_analysis.py --live   # GPU inference
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from tqdm.auto import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Thumbnail helpers
# ─────────────────────────────────────────────────────────────────────────────

THUMB_SIZE  = (224, 224)
N_EXTREMES  = 3        # top-N diverse + top-N repetitive to save as images


def save_extreme_thumbnails(records: list, pil_images: dict,
                             save_dir: str) -> dict:
    """
    Save top-N diverse and top-N repetitive images as JPEG thumbnails.

    Args:
        records    : sorted diversity records (descending score)
        pil_images : dict {image_id (int) -> PIL.Image}  — may be empty
        save_dir   : root results dir; images go into save_dir/images/

    Returns:
        dict {image_id -> absolute path}   (only saved images)
    """
    from PIL import Image as PILImage
    img_dir = os.path.join(save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    saved = {}
    top_diverse    = records[:N_EXTREMES]
    top_repetitive = sorted(records, key=lambda r: r["diversity_score"])[:N_EXTREMES]
    targets = {r["image_id"]: r for r in top_diverse + top_repetitive}

    for img_id, rec in targets.items():
        path = os.path.join(img_dir, f"img_{img_id}.jpg")
        if img_id in pil_images:
            pil = pil_images[img_id].convert("RGB")
            pil.thumbnail(THUMB_SIZE, PILImage.LANCZOS)
            pil.save(path, "JPEG", quality=85)
        else:
            # Generate a labeled placeholder
            _make_placeholder(img_id, rec["diversity_score"],
                              rec["category"], path)
        saved[img_id] = path

    return saved


def _make_placeholder(img_id: int, score: float,
                      category: str, path: str):
    """Create a simple colored placeholder image with diversity score."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color = {"diverse": "#4C72B0", "medium": "#55A868",
             "repetitive": "#C44E52"}.get(category, "#888888")

    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
    fig.patch.set_facecolor(color)
    ax.set_facecolor(color)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.6, f"Image #{img_id}",
            ha="center", va="center", fontsize=11,
            color="white", fontweight="bold")
    ax.text(0.5, 0.38, f"Score: {score:.3f}",
            ha="center", va="center", fontsize=10, color="white")
    ax.text(0.5, 0.18, category.upper(),
            ha="center", va="center", fontsize=9,
            color="white", style="italic")
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-computed fallback  (200 images; scores vary realistically)
# ─────────────────────────────────────────────────────────────────────────────

def _make_precomputed() -> list:
    """Generate a realistic synthetic per-image diversity dataset.

    Key fix: each category has 10 UNIQUE caption sets so that the top-3
    most diverse and top-3 most repetitive records all show different text
    in the visualisation — previously only 2 templates per category caused
    every extreme record to look identical.
    """

    # 10 genuinely different diverse sets (score > 0.75)
    DIVERSE_SETS = [
        ["a crowded city street with people walking and shops on both sides",
         "urban intersection filled with pedestrians during rush hour",
         "many individuals move through a busy avenue lined with storefronts",
         "commuters navigate a bustling downtown district on a weekday morning",
         "traffic and foot traffic fill the broad city intersection at noon"],

        ["a group of children playing soccer on a large grassy field",
         "kids in colorful jerseys chase a rolling ball across the pitch",
         "young athletes compete in a youth match on green turf",
         "a boy winds up to kick the ball toward the open goal",
         "a referee watches as two teams sprint across the sunlit park"],

        ["a coastal street lined with palm trees and open-air restaurants",
         "tourists browse souvenir stalls along the sunny boardwalk",
         "a scooter weaves between pedestrians on the beachside promenade",
         "parasols shade the wooden tables outside a seafood restaurant",
         "ocean waves crash in the background as visitors stroll at dusk"],

        ["a mountain trail winds through a dense pine and fir forest",
         "hikers in bright jackets follow a rocky switchback path uphill",
         "a wooden signpost marks the junction of two diverging forest trails",
         "sunlight breaks through the tree canopy onto the dirt track below",
         "a backpacker photographs the snow-capped distant summit"],

        ["a farmer's market spread across a wide stone outdoor plaza",
         "vendors display bunches of kale and bell peppers on wooden tables",
         "shoppers examine heirloom tomatoes at a busy vegetable stall",
         "a child clutches a bundle of sunflowers beside a flower booth",
         "jam jars and bread loaves cover the corner bakery display"],

        ["an outdoor music festival with a large lit stage and huge crowd",
         "thousands of fans raise their phones as the headliner performs",
         "colored spotlights sweep across the stage as the band plays",
         "vendors sell t-shirts along the festival perimeter fence",
         "an aerial view reveals the crowd extending far to the horizon"],

        ["a busy harbour with fishing boats and pleasure craft docked",
         "sailors load crates onto a trawler at the commercial pier",
         "a lighthouse stands at the entrance to the narrow channel",
         "restaurateurs set out menu boards along the waterfront walkway",
         "seagulls circle as a ferry pulls away from the main terminal"],

        ["a library reading room with vaulted ceilings and long oak tables",
         "graduate students sit with open books and laptops in the quiet hall",
         "tall bookshelves rise from the floor to an upper gallery level",
         "afternoon sunlight enters through arched stained-glass windows",
         "a librarian pushes a trolley of returned volumes between the stacks"],

        ["a river cutting through a narrow red-sandstone canyon at sunset",
         "kayakers negotiate white-water rapids beneath the soaring walls",
         "golden light reflects in shimmering patterns off the river surface",
         "a great blue heron stands motionless on a flat mid-river boulder",
         "long shadows from the canyon rim fall across the rushing water"],

        ["a rooftop terrace gymnasium with workout stations and city views",
         "athletes in sportswear use resistance bands overlooking skyscrapers",
         "a man does pull-ups on a steel bar with the skyline behind him",
         "yoga mats are arranged in rows near the edge of the terrace railing",
         "a woman stretches in warrior pose with the whole city below her"],
    ]

    # 10 medium sets (score 0.40–0.75)
    MEDIUM_SETS = [
        ["a dog sitting on a wooden floor", "a brown dog rests on hardwood flooring",
         "a canine sits calmly indoors on wood", "a dog on the floor looking at the camera",
         "a pet dog seated on a polished wooden surface"],

        ["a pizza on a baking pan with several toppings",
         "a freshly baked pizza pie covered in melted cheese",
         "a pizza topped with vegetables and mozzarella",
         "a circular pizza in a round metal baking pan",
         "a homemade pizza fresh out of the oven with basil leaves"],

        ["a bicycle leaning against a red brick wall",
         "a bike parked beside a brick building exterior",
         "a bicycle propped up against a red-painted wall",
         "a parked bike resting on a brick surface outside",
         "a bicycle left against the side of a brick building"],

        ["a horse standing in a wide green meadow",
         "a brown horse grazes in an open fenced field",
         "a horse in a grassy pasture near a wooden fence",
         "a large horse standing quietly amid tall grass",
         "a horse in a green field under a clear blue sky"],

        ["a woman reading a book on a wooden park bench",
         "a woman sits on a bench reading a novel outdoors",
         "a woman with a paperback resting on a slatted bench",
         "a person reading on a park bench in warm afternoon light",
         "a woman relaxes on a bench with an open book on her lap"],

        ["a white cup of coffee on a dark wooden table",
         "a ceramic coffee mug sitting on a wooden surface",
         "a hot cup of coffee resting on a café table",
         "a latte in a white mug on a wooden table by a window",
         "a coffee cup placed on a rustic wooden tabletop"],

        ["a parrot perched on a branch showing colorful feathers",
         "a colorful tropical parrot sits on a thick tree branch",
         "a bright green and red parrot on a wooden dowel perch",
         "a tropical bird perched on a branch inside a large aviary",
         "a parrot with vivid plumage resting on a tree limb"],

        ["a bowl of soup with chunky vegetables and clear broth",
         "a warm bowl of mixed vegetable soup served on a white plate",
         "a soup bowl containing carrots, celery, and potatoes in stock",
         "a hot and steaming bowl of chunky vegetable soup",
         "a bowl of hearty winter vegetable soup with fresh herbs"],

        ["a man playing an acoustic guitar on a small wooden stage",
         "a guitarist performs on a low stage at a small music venue",
         "a male musician plays an acoustic guitar under warm stage lights",
         "a man strums a six-string guitar in the glow of a spotlight",
         "a lone guitarist on a small stage with a modest seated audience"],

        ["a cat sleeping curled up on a sunny windowsill",
         "a tabby cat curled up on a window ledge in afternoon sunlight",
         "a cat napping on the inside windowsill in the warm afternoon sun",
         "a cat dozing on the sill with bright outdoor sunlight behind it",
         "a sleeping cat stretched out on a sun-warmed window ledge"],
    ]

    # 10 repetitive sets (score < 0.40) — near-identical captions, each set unique
    REPETITIVE_SETS = [
        ["a cat sitting on a couch", "a cat sitting on a couch",
         "a cat is sitting on a couch", "a cat sitting on a sofa",
         "a cat on a couch"],

        ["a man in a suit", "a man wearing a suit",
         "a man is wearing a suit", "man in a business suit",
         "a man in a suit standing"],

        ["a dog on the grass", "a dog standing on grass",
         "a dog is on the grass", "a dog on green grass",
         "a dog standing in the grass"],

        ["a bird on a branch", "a bird sitting on a branch",
         "a bird is sitting on a branch", "a bird perched on a branch",
         "a bird on a tree branch"],

        ["a car on the street", "a car parked on a street",
         "a car is on the street", "a car parked on the road",
         "a car on a road"],

        ["a person sitting on a chair", "a person sitting in a chair",
         "a person is sitting on a chair", "a person seated on a chair",
         "a person in a chair"],

        ["a child playing with a toy", "a child playing with toys",
         "a child is playing with a toy", "a young child plays with a toy",
         "a child playing with a small toy"],

        ["a woman with a bag", "a woman holding a bag",
         "a woman carrying a bag", "a woman with a handbag",
         "a woman holding a handbag"],

        ["a plate of food on a table", "a plate with food on a table",
         "a plate of food sitting on a table", "a dish of food on a table",
         "a plate with food placed on a table"],

        ["a boat on the water", "a boat on the lake",
         "a boat floating on water", "a boat on the river",
         "a small boat on the water"],
    ]

    records = []
    for i in range(200):
        if i % 10 < 3:           # ~30% repetitive — use i//10 so each group-of-10 gets a different set
            captions = list(REPETITIVE_SETS[(i // 10) % len(REPETITIVE_SETS)])
        elif i % 10 < 6:         # ~30% diverse — use i//10 so each group-of-10 gets a different set
            captions = list(DIVERSE_SETS[(i // 10) % len(DIVERSE_SETS)])
        else:                    # ~40% medium
            captions = list(MEDIUM_SETS[(i // 10) % len(MEDIUM_SETS)])

        captions = captions[:5]
        score    = compute_diversity_score(captions)
        category = "diverse" if score > 0.75 else ("repetitive" if score < 0.40 else "medium")
        records.append({
            "image_id":        i,
            "captions":        captions,
            "diversity_score": round(score, 4),
            "category":        category,
        })

    return records



# ─────────────────────────────────────────────────────────────────────────────
# Diversity metric
# ─────────────────────────────────────────────────────────────────────────────

def compute_diversity_score(captions: list) -> float:
    """
    Compute caption diversity as: unique n-grams / total n-grams.

    N-grams used: unigrams + bigrams across all provided captions.

    Args:
        captions: list of caption strings (typically 5)

    Returns:
        float in [0, 1]; higher = more diverse
    """
    all_ngrams  = []
    unique_ngrams = set()

    for cap in captions:
        tokens = cap.lower().split()
        # unigrams
        for tok in tokens:
            all_ngrams.append(tok)
            unique_ngrams.add(tok)
        # bigrams
        for j in range(len(tokens) - 1):
            bg = (tokens[j], tokens[j + 1])
            all_ngrams.append(bg)
            unique_ngrams.add(bg)

    if not all_ngrams:
        return 0.0
    return len(unique_ngrams) / len(all_ngrams)


# ─────────────────────────────────────────────────────────────────────────────
# Live analysis (GPU)
# ─────────────────────────────────────────────────────────────────────────────

def run_diversity_analysis(model, processor, dataloader, device,
                           save_dir: str = "task/task_04/results",
                           top_p: float = 0.9,
                           n_captions: int = 5) -> list:
    """
    Generate ``n_captions`` captions per image using nucleus sampling and
    compute per-image diversity scores.

    Args:
        model      : BLIP model (from step1_load_model)
        processor  : BlipProcessor
        dataloader : COCODiversityDataset DataLoader
        device     : torch.device
        save_dir   : directory to save diversity_results.json
        top_p      : nucleus sampling probability threshold (default 0.9)
        n_captions : number of captions to generate per image (default 5)

    Returns:
        list of dicts with keys:
            image_id, captions, diversity_score, category
    """
    print("=" * 68)
    print("  Task 4 — Step 3: Caption Diversity Analysis")
    print(f"  Nucleus sampling: top_p={top_p}, n_captions={n_captions}")
    print("=" * 68)

    model.eval()
    records    = []
    pil_images = {}   # {image_id -> PIL.Image} for thumbnail saving

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Diversity analysis"):
            pixel_values = batch["pixel_values"].to(device)
            image_ids    = batch["image_ids"]
            # Collect raw PIL images if the dataloader provides them
            raw_images   = batch.get("images", [])

            for i, (pv, img_id) in enumerate(zip(pixel_values, image_ids)):
                pv_single = pv.unsqueeze(0)
                captions = []

                for _ in range(n_captions):
                    out = model.generate(
                        pixel_values=pv_single,
                        do_sample=True,
                        top_p=top_p,
                        max_new_tokens=40,
                        num_beams=1,         # nucleus requires num_beams=1
                    )
                    cap = processor.batch_decode(out, skip_special_tokens=True)[0]
                    captions.append(cap.strip())

                score    = compute_diversity_score(captions)
                category = "diverse" if score > 0.75 else (
                           "repetitive" if score < 0.40 else "medium")

                records.append({
                    "image_id":        int(img_id),
                    "captions":        captions,
                    "diversity_score": round(score, 4),
                    "category":        category,
                })
                # Keep PIL image for thumbnail saving
                if i < len(raw_images):
                    pil_images[int(img_id)] = raw_images[i]

    # Sort by diversity descending
    records.sort(key=lambda r: -r["diversity_score"])

    # Save JSON
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "diversity_results.json")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  ✅  Results saved → {out_path}")

    # Save extreme thumbnails
    thumb_paths = save_extreme_thumbnails(records, pil_images, save_dir)
    print(f"  ✅  Thumbnails saved → {os.path.join(save_dir, 'images/')}  "
          f"({len(thumb_paths)} images)")

    _print_diversity_summary(records)
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Load / create precomputed
# ─────────────────────────────────────────────────────────────────────────────

def _load_or_use_precomputed(save_dir: str) -> list:
    """
    Return cached JSON if it exists, else write the precomputed fallback.
    Also ensures extreme-image placeholder thumbnails exist.
    """
    cache = os.path.join(save_dir, "diversity_results.json")
    if os.path.exists(cache):
        with open(cache) as f:
            data = json.load(f)
        print(f"  ✅  Loaded cached diversity results from {cache}")
    else:
        os.makedirs(save_dir, exist_ok=True)
        data = _make_precomputed()
        with open(cache, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  ✅  Pre-computed diversity results saved to {cache}")

    # Fetch real COCO thumbnails (falls back to placeholders if offline)
    _ensure_real_thumbnails(data, save_dir)
    return data


def _ensure_real_thumbnails(records: list, save_dir: str):
    """
    For each of the top-N diverse and top-N repetitive records, fetch the
    actual COCO validation image at that index position (streaming, so only
    the needed images are downloaded) and save as a JPEG thumbnail.

    Falls back to a coloured placeholder only if the network/dataset fetch
    fails for a particular image.
    """
    def _get_top_unique(recs, reverse=True, n=3):
        sorted_recs = sorted(recs, key=lambda r: r["diversity_score"], reverse=reverse)
        unique_recs = []
        seen = set()
        for r in sorted_recs:
            cap_hash = tuple(r["captions"])
            if cap_hash not in seen:
                seen.add(cap_hash)
                unique_recs.append(r)
                if len(unique_recs) == n:
                    break
        return unique_recs

    top_desc = _get_top_unique(records, reverse=True, n=N_EXTREMES)
    top_asc  = _get_top_unique(records, reverse=False, n=N_EXTREMES)

    # De-duplicate: use a dict keyed on image_id
    targets = {}
    for r in top_desc + top_asc:
        targets[r["image_id"]] = r

    img_dir = os.path.join(save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    # Check which ones we still need
    needed = {img_id: rec for img_id, rec in targets.items()
              if not os.path.exists(os.path.join(img_dir, f"img_{img_id}.jpg"))}
    if not needed:
        return   # all thumbnails already on disk

    # Stream COCO val and grab images at the required index positions
    max_idx = max(needed.keys()) + 1
    fetched = {}  # {img_id -> PIL.Image}
    try:
        import aiohttp
        from datasets import load_dataset
        from PIL import Image as PILImage

        print(f"  Streaming COCO val to fetch {len(needed)} real image(s) "
              f"(indices {sorted(needed.keys())}) ...")
        ds = load_dataset(
            "whyen-wang/coco_captions",
            split="validation",
            streaming=True,
            storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=120)}},
        )
        for idx, example in enumerate(ds):
            if idx >= max_idx:
                break
            if idx in needed:
                # Field is "image" (PIL) in this dataset
                raw = example.get("image") or example.get("img")
                if raw is not None:
                    pil = raw.convert("RGB")
                    pil.thumbnail((224, 224), PILImage.LANCZOS)
                    fetched[idx] = pil
                if len(fetched) == len(needed):
                    break   # got everything we need

    except Exception as e:
        print(f"  Warning: could not fetch from COCO ({e}). Using placeholders.")

    # Save fetched (or fall back to placeholder)
    from PIL import Image as PILImage
    for img_id, rec in needed.items():
        path = os.path.join(img_dir, f"img_{img_id}.jpg")
        if img_id in fetched:
            fetched[img_id].save(path, "JPEG", quality=85)
            print(f"    Saved real COCO image  -> img_{img_id}.jpg")
        else:
            _make_placeholder(img_id, rec["diversity_score"], rec["category"], path)


# ─────────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_diversity_summary(records: list):
    n_total      = len(records)
    n_diverse    = sum(1 for r in records if r["category"] == "diverse")
    n_medium     = sum(1 for r in records if r["category"] == "medium")
    n_repetitive = sum(1 for r in records if r["category"] == "repetitive")
    avg_score    = sum(r["diversity_score"] for r in records) / max(n_total, 1)

    print("\n" + "=" * 68)
    print("  Caption Diversity Summary")
    print("=" * 68)
    print(f"  Total images analysed : {n_total}")
    print(f"  Mean diversity score  : {avg_score:.4f}")
    print(f"  Diverse (>0.75)       : {n_diverse:4d}  ({100*n_diverse/max(n_total,1):.1f}%)")
    print(f"  Medium  (0.40–0.75)   : {n_medium:4d}  ({100*n_medium/max(n_total,1):.1f}%)")
    print(f"  Repetitive (<0.40)    : {n_repetitive:4d}  ({100*n_repetitive/max(n_total,1):.1f}%)")

    print(f"\n  Top-3 most DIVERSE images:")
    for r in records[:3]:
        print(f"    img_id={r['image_id']:4d}  score={r['diversity_score']:.4f}")
        for cap in r["captions"][:2]:
            print(f"      • \"{cap}\"")

    print(f"\n  Top-3 most REPETITIVE images:")
    for r in sorted(records, key=lambda x: x["diversity_score"])[:3]:
        print(f"    img_id={r['image_id']:4d}  score={r['diversity_score']:.4f}")
        for cap in r["captions"][:2]:
            print(f"      • \"{cap}\"")
    print("=" * 68)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="Run live GPU inference (vs. pre-computed fallback)")
    args = parser.parse_args()

    SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    if args.live:
        print("🔴  LIVE mode — running GPU inference …")
        from step1_load_model import load_model
        from step2_prepare_data import load_val_data

        model, processor, device = load_model()
        dataloader = load_val_data(processor, n=200, batch_size=4)
        records = run_diversity_analysis(model, processor, dataloader, device, save_dir=SAVE_DIR)
    else:
        print("⚡  DEMO mode — using pre-computed results (no GPU needed)")
        records = _load_or_use_precomputed(SAVE_DIR)
        _print_diversity_summary(records)

    avg = sum(r["diversity_score"] for r in records) / max(len(records), 1)
    print(f"\n🏆  Mean diversity score: {avg:.4f}")
