"""
step4_visualize.py
==================
STEP 4 — Overlay heatmaps on the image and save a 2×5 grid PNG.

Responsibilities:
- Accept the original image, list of tokens, and parallel list of heatmaps.
- Overlay each heatmap on the image using the INFERNO colormap.
- Compose a 2×5 grid (original image + up to 9 word panels).
- Save as a high-DPI PNG for clean presentation.

The overlay uses a pixel-level transparency mask so that only the
truly "hot" regions of the heatmap are blended, keeping the rest of
the image fully visible and unobscured.
"""

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


# ────────────────────────────────────────────────────────────────────────────
def overlay_heatmap_on_image(
    image_pil: Image.Image,
    heatmap_np: np.ndarray,
    alpha: float = 0.50,
    hot_threshold: float = 0.10,
    colormap=cv2.COLORMAP_INFERNO,
) -> Image.Image:
    """
    Blend a normalised [0,1] heatmap on top of image_pil.

    Args:
        image_pil     : PIL image (will be resized to match heatmap).
        heatmap_np    : (H, W) float32 array, values in [0,1].
        alpha         : Opacity of hot regions (0 = invisible, 1 = fully colored).
        hot_threshold : Pixels below this value are NOT blended (stays the original image).
        colormap      : OpenCV colormap to apply.

    Returns:
        blended PIL image.
    """
    h, w = heatmap_np.shape
    img  = np.array(image_pil.resize((w, h), Image.LANCZOS))

    # Apply colormap
    hm_u8    = np.uint8(255.0 * heatmap_np)
    colored  = cv2.applyColorMap(hm_u8, colormap)
    colored  = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    # Pixel-level mask: only blend "hot" spots
    mask     = (heatmap_np > hot_threshold).astype(np.float32)[..., None]
    blended  = img * (1 - mask * alpha) + colored * (mask * alpha)

    return Image.fromarray(blended.astype(np.uint8))


def save_attention_grid(
    image_pil: Image.Image,
    tokens: list,
    heatmaps: list,
    out_path: str,
    n_rows: int = 2,
    n_cols: int = 5,
    dpi: int = 150,
    alpha: float = 0.50,
    verbose: bool = True,
) -> str:
    """
    Build and save a (n_rows × n_cols) grid of per-word attention overlays.

    Layout:
        Cell [0]: Original image.
        Cells [1..n_rows*n_cols-1]: Word heatmaps in generation order.

    Args:
        image_pil   : Original (or 224×224) PIL image.
        tokens      : List of decoded word strings.
        heatmaps    : Parallel list of (224, 224) numpy heatmaps.
        out_path    : Absolute path for the saved PNG.
        n_rows, n_cols : Grid dimensions.
        dpi         : Output DPI (150 recommended for presentation).
        alpha       : Heatmap blend opacity.
        verbose     : Print save confirmation.

    Returns:
        out_path (so callers can chain this call).
    """
    n_panels = n_rows * n_cols           # total slots
    n_words  = min(n_panels - 1, len(tokens))   # slot 0 = original

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 4.5))
    axes = axes.flatten()

    # Panel 0 — original image
    axes[0].imshow(image_pil)
    axes[0].set_title("Original Image", fontsize=13, fontweight="bold", pad=4)
    axes[0].axis("off")

    # Panels 1…n_words — per-word heatmap overlays
    for i in range(n_words):
        overlay = overlay_heatmap_on_image(image_pil, heatmaps[i], alpha=alpha)
        ax = axes[i + 1]
        ax.imshow(overlay)
        ax.set_title(f"'{tokens[i]}'", fontsize=13, fontweight="bold", pad=4)
        ax.axis("off")

    # Turn off unused panels
    for j in range(n_words + 1, n_panels):
        axes[j].axis("off")

    caption = " ".join(tokens)
    fig.suptitle(
        f"Attention Flow (Multi-Layer GradCAM)\nCaption: \"{caption}\"",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    if verbose:
        print(f"✅ Attention grid saved → {out_path}")

    return out_path
