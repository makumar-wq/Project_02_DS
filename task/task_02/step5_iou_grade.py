"""
step5_iou_grade.py
==================
STEP 5 — Quantitative Alignment Grading via Zero-Shot Object Detection.

Responsibilities:
- Load OWL-ViT (zero-shot open-vocabulary object detector).
- For each meaningful word in the caption, find its bounding box in the image.
- Binarize the Attention Flow heatmap with Otsu's thresholding.
- Compute Intersection over Union (IoU) between heatmap mask and bounding box.
- Plot and save a word-position vs IoU scatter chart.

Why OWL-ViT?
    OWL-ViT is a zero-shot detector: it can find *any* object in an image
    just by reading its name.  This means we do NOT need any pre-annotated
    bounding boxes — just our generated caption words.  It acts as a fully
    automated judge of how well the AI's attention was grounded.

Why Otsu's Thresholding?
    Otsu's method automatically finds the optimal binary split point of the
    heatmap histogram, separating "looking here" from "not looking here"
    without needing a hand-tuned cut-off value.

IoU Interpretation:
    0.0       = No overlap (attention fired in the wrong place).
    0.1–0.3   = Weak grounding (partial overlap, some drift).
    0.3+      = Good grounding (attention focused on the right region).
"""

import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ── Stop-word filter ──────────────────────────────────────────────────────────
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
    "in", "on", "at", "to", "for", "with", "by", "it", "this", "that",
    "there", "here", "of", "up", "out", ".", ",", "!", "##",
}


def load_detector(device, verbose: bool = True):
    """
    Load the OWL-ViT zero-shot object detection pipeline.

    Args:
        device  : torch.device (MPS, CUDA, or CPU).
        verbose : Print loading message.

    Returns:
        detector – transformers.pipeline object.
    """
    from transformers import pipeline

    if verbose:
        print("🔭 Loading OWL-ViT zero-shot object detector …")

    detector = pipeline(
        task="zero-shot-object-detection",
        model="google/owlvit-base-patch32",
        device=device,
    )

    if verbose:
        print("✅ OWL-ViT ready")

    return detector


def binarize_heatmap(heatmap_np: np.ndarray, target_hw: tuple) -> np.ndarray:
    """
    Resize and Otsu-threshold a float heatmap into a boolean mask.

    Args:
        heatmap_np  : (H, W) float32 heatmap in [0, 1].
        target_hw   : (height, width) of the original image.

    Returns:
        (H, W) boolean mask.
    """
    hm = cv2.resize(heatmap_np, (target_hw[1], target_hw[0]))
    hm_u8 = np.uint8(255.0 * hm)
    _, binary = cv2.threshold(hm_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary > 0


def calculate_iou(mask: np.ndarray, box: list, img_shape: tuple) -> float:
    """
    Calculate Intersection over Union between a boolean mask and a bounding box.

    Args:
        mask      : (H, W) boolean numpy array.
        box       : [xmin, ymin, xmax, ymax] in image pixel coords.
        img_shape : (H, W) of the image.

    Returns:
        IoU score in [0, 1].
    """
    box_mask = np.zeros(img_shape, dtype=bool)
    xmin, ymin, xmax, ymax = map(int, box)
    # Clamp to image bounds
    xmin = max(0, xmin); ymin = max(0, ymin)
    xmax = min(img_shape[1], xmax); ymax = min(img_shape[0], ymax)
    box_mask[ymin:ymax, xmin:xmax] = True

    inter = np.logical_and(mask, box_mask).sum()
    union = np.logical_or(mask, box_mask).sum()
    return float(inter) / union if union > 0 else 0.0


def grade_alignment(
    image_pil,
    tokens: list,
    heatmaps: list,
    detector,
    min_detection_score: float = 0.05,
    verbose: bool = True,
) -> list:
    """
    For each meaningful token, attempt to detect its object in the image
    and compute IoU against the Attention Flow heatmap.

    Args:
        image_pil           : Original PIL image (un-resized).
        tokens              : List of decoded word strings.
        heatmaps            : Parallel list of (H, W) numpy heatmaps.
        detector            : OWL-ViT pipeline.
        min_detection_score : Only accept detections above this confidence.
        verbose             : Print per-token results.

    Returns:
        List of dicts: { 'word', 'position', 'iou', 'det_score' }
    """
    img_shape      = (image_pil.height, image_pil.width)
    results        = []

    if verbose:
        print("\n📊 Grading alignment (Attention Flow IoU)…")

    for idx, (word, hm) in enumerate(zip(tokens, heatmaps)):
        clean_word = word.replace("##", "").lower()
        if len(clean_word) < 3 or clean_word in _STOP_WORDS or not clean_word.isalpha():
            continue

        detections = detector(image_pil, candidate_labels=[clean_word])
        best_box, best_score = None, 0.0

        for d in detections:
            if d["score"] > best_score and d["score"] >= min_detection_score:
                best_score = d["score"]
                best_box   = [d["box"]["xmin"], d["box"]["ymin"],
                               d["box"]["xmax"], d["box"]["ymax"]]

        if best_box is not None:
            mask = binarize_heatmap(hm, img_shape)
            iou  = calculate_iou(mask, best_box, img_shape)
            if verbose:
                print(f"  '{clean_word}' (pos {idx+1}): det_score={best_score:.2f}, IoU={iou:.4f}")
            results.append({"word": clean_word, "position": idx + 1,
                            "iou": iou, "det_score": best_score})
        else:
            if verbose:
                print(f"  '{clean_word}' (pos {idx+1}): no detection found")

    mean_iou = np.mean([r["iou"] for r in results]) if results else 0.0
    if verbose:
        print(f"\n  Mean Alignment IoU: {mean_iou:.4f}")

    return results


def plot_iou_chart(
    all_results: list,
    out_path: str,
    verbose: bool = True,
) -> str:
    """
    Save a scatter plot of word position vs Attention Flow IoU.

    Args:
        all_results : Flat list of result dicts from grade_alignment().
        out_path    : Absolute path to save the PNG.
        verbose     : Print save confirmation.

    Returns:
        out_path.
    """
    if not all_results:
        if verbose:
            print("⚠️  No IoU results to plot.")
        return out_path

    positions = [r["position"] for r in all_results]
    ious      = [r["iou"] for r in all_results]
    words     = [r["word"] for r in all_results]

    fig, ax = plt.subplots(figsize=(9, 5))
    sc = ax.scatter(positions, ious, zorder=5, alpha=0.8, s=80,
                    c=ious, cmap="RdYlGn", vmin=0, vmax=0.5)
    plt.colorbar(sc, ax=ax, label="IoU")

    # Annotate each point with the word
    for pos, iou, word in zip(positions, ious, words):
        ax.annotate(word, (pos, iou), textcoords="offset points",
                    xytext=(4, 4), fontsize=8, alpha=0.9)

    # Trend line
    if len(positions) > 1:
        z = np.polyfit(positions, ious, 1)
        p_fn = np.poly1d(z)
        xs = sorted(positions)
        ax.plot(xs, [p_fn(x) for x in xs], "b--", alpha=0.5, label="Trend")
        ax.legend()

    ax.set_title("Word Position vs. Attention Flow Alignment (IoU)\n"
                 "(Higher = model actually looked at the right region)", fontsize=13)
    ax.set_xlabel("Word position in caption")
    ax.set_ylabel("Alignment IoU")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    if verbose:
        print(f"✅ IoU plot saved → {out_path}")

    return out_path
