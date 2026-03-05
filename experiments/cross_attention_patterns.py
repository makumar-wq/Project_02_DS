"""
experiments/cross_attention_patterns.py
========================================
Documents and compares the four distinct cross-attention (fusion) patterns
used by each architecture in this pipeline.

This module does NOT require loading any model — it produces a static
analysis table and inline architecture diagrams, and can optionally
compute the number of cross-attention parameter counts from loaded models.

Usage (standalone):
    python -m experiments.cross_attention_patterns

Architecture Summary
--------------------

┌─────────────────┬───────────────────────────┬──────────────────────────────────┐
│ Architecture    │ Fusion Mechanism          │ Cross-Attention Exists?           │
├─────────────────┼───────────────────────────┼──────────────────────────────────┤
│ ViT-GPT2        │ Standard Full CA          │ ✅ Yes — at every GPT-2 layer     │
│ BLIP (MED)      │ Gated Cross-Attention MED │ ✅ Yes — between SA and FFN       │
│ GIT             │ Self-Attn Prefix          │ ❌ No — unified causal SA         │
│ Custom VLM      │ Visual Prefix-Tuning      │ ❌ No — linear projection + SA    │
└─────────────────┴───────────────────────────┴──────────────────────────────────┘
"""


# ─────────────────────────────────────────────────────────────────────────────
# Static Architecture Descriptions
# ─────────────────────────────────────────────────────────────────────────────

PATTERNS = [
    {
        "name": "ViT-GPT2",
        "model_id": "nlpconnect/vit-gpt2-image-captioning",
        "cross_attention": True,
        "ca_type": "Standard Full Cross-Attention",
        "description": (
            "Every GPT-2 decoder layer has an explicit cross-attention block. "
            "Each text token attends to ALL 197 ViT patch embeddings "
            "(1 CLS + 196 spatial) at every layer. "
            "This is the brute-force approach — maximum information, highest compute."
        ),
        "fusion_formula": "h_text = CrossAttn(Q=h_text, K=h_vis, V=h_vis)",
        "ablation_support": True,
        "ablation_method": "encoder_attention_mask on generate()",
    },
    {
        "name": "BLIP (MED)",
        "model_id": "Salesforce/blip-image-captioning-base",
        "cross_attention": True,
        "ca_type": "Gated Multimodal Encoder-Decoder (MED)",
        "description": (
            "BLIP's MED architecture injects a cross-attention sub-layer "
            "BETWEEN the self-attention and FFN sub-layers at each decoder block. "
            "A learnable gate controls how much visual information passes through. "
            "This is more targeted than ViT-GPT2's brute-force attention."
        ),
        "fusion_formula": (
            "h = SA(h_text)  "
            "→  h = h + gate * CrossAttn(Q=h, K=h_vis, V=h_vis)  "
            "→  h = FFN(h)"
        ),
        "ablation_support": True,
        "ablation_method": "encoder_attention_mask via generate_with_mask()",
    },
    {
        "name": "GIT",
        "model_id": "microsoft/git-base-coco",
        "cross_attention": False,
        "ca_type": "Zero Cross-Attention (Self-Attention Prefix)",
        "description": (
            "GIT concatenates image patch embeddings directly in front of text tokens "
            "to form a flat joint sequence: [img_tokens | text_tokens]. "
            "A single causal self-attention Transformer processes the whole thing. "
            "There is NO dedicated cross-attention block. "
            "Modality fusion is implicit via positional self-attention."
        ),
        "fusion_formula": "h = CausalSA([h_vis; h_text])",
        "ablation_support": False,
        "ablation_method": "N/A — no encoder_attention_mask concept",
    },
    {
        "name": "Custom VLM (Shakespeare)",
        "model_id": "google/vit-base-patch16-224-in21k (ViT) + char-level decoder",
        "cross_attention": False,
        "ca_type": "Visual Prefix-Tuning (Linear Bridge + Causal SA)",
        "description": (
            "A frozen ViT extracts 197 patch embeddings (768-dim). "
            "A single trainable Linear(768→384) projects these to the decoder's "
            "embedding space. Projected visual tokens are prepended to character "
            "embeddings and the Shakespeare causal decoder processes them jointly. "
            "Only the linear projection is trained (~294K params, <0.2% of total). "
            "\nKey insight: cross-attention is provably unnecessary when modalities "
            "are aligned in the same embedding space via prefix concatenation."
        ),
        "fusion_formula": (
            "v = Linear(ViT(img))  "
            "→  x = CausalSA([v; char_emb])  "
            "→  logits = LMHead(x[len(v):])"
        ),
        "ablation_support": False,
        "ablation_method": "N/A — visual prefix is part of unified sequence",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Comparison Table Printer
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table():
    """Print a formatted comparison table to stdout."""
    print("\n" + "=" * 80)
    print("  Cross-Attention Pattern Comparison")
    print("=" * 80)
    print(f"  {'Architecture':<22} {'CA?':>5}  {'Type':<35}  {'Ablation?':>9}")
    print("  " + "-" * 76)
    for p in PATTERNS:
        ca  = "  ✅" if p["cross_attention"] else "  ❌"
        abl = "    ✅" if p["ablation_support"] else "    ❌"
        print(f"  {p['name']:<22} {ca:>5}  {p['ca_type']:<35} {abl:>9}")
    print("=" * 80)

    for p in PATTERNS:
        print(f"\n  ── {p['name']} ──────────────────────────────────────────────")
        print(f"  Model  : {p['model_id']}")
        print(f"  CA Type: {p['ca_type']}")
        print(f"  Formula: {p['fusion_formula']}")
        for line in p["description"].split("\n"):
            print(f"  {line.strip()}")
        if p["ablation_support"]:
            print(f"  Ablation: {p['ablation_method']}")
        else:
            print(f"  ⚠️  Ablation: {p['ablation_method']}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Optional: Parameter Count from Loaded Models
# ─────────────────────────────────────────────────────────────────────────────

def count_cross_attention_params(model, model_name: str) -> dict:
    """
    Count parameters in cross-attention layers for BLIP or ViT-GPT2.

    For GIT / Custom VLM (no CA), returns zero.

    Args:
        model      : loaded PyTorch model
        model_name : 'blip' | 'vit_gpt2' | 'git' | 'custom'

    Returns:
        dict with 'total', 'cross_attn', 'cross_attn_pct'
    """
    total = sum(p.numel() for p in model.parameters())
    ca_params = 0

    if model_name == "blip":
        for name, p in model.named_parameters():
            if "crossattention" in name.lower():
                ca_params += p.numel()

    elif model_name == "vit_gpt2":
        for name, p in model.named_parameters():
            if "crossattention" in name.lower() or "cross_attn" in name.lower():
                ca_params += p.numel()

    # GIT / custom: 0 cross-attention params by design

    return {
        "model": model_name,
        "total_params": total,
        "cross_attn_params": ca_params,
        "cross_attn_pct": ca_params / total * 100 if total > 0 else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print_comparison_table()

    # Optionally count params for all four models
    count_params = input(
        "\nCount cross-attention parameters in all models? "
        "(requires downloading BLIP+ViT-GPT2+GIT) [y/N]: "
    ).strip().lower()

    if count_params == "y":
        import torch
        device = torch.device("cpu")

        print("\nLoading models to count parameters...\n")

        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        from config import CFG
        from models.blip_tuner import get_blip_model
        from models.vit_gpt2_tuner import get_vit_gpt2_model
        from models.git_tuner import get_git_model
        from models.custom_vlm import CustomVLM, build_char_vocab

        cfg = CFG()

        rows = []

        model_b, _ = get_blip_model(cfg, device)
        rows.append(count_cross_attention_params(model_b, "blip"))
        del model_b

        model_v, _, _ = get_vit_gpt2_model(cfg, device)
        rows.append(count_cross_attention_params(model_v, "vit_gpt2"))
        del model_v

        model_g, _ = get_git_model(cfg, device)
        rows.append(count_cross_attention_params(model_g, "git"))
        del model_g

        with open(cfg.shakespeare_file, "r") as f:
            text = f.read()
        _, c2i, i2c, vs = build_char_vocab(text)
        model_c = CustomVLM(vocab_size=vs)
        rows.append(count_cross_attention_params(model_c, "custom"))
        del model_c

        print("\n" + "=" * 65)
        print("  Cross-Attention Parameter Counts")
        print("=" * 65)
        print(f"  {'Model':<15}  {'Total':>12}  {'CA Params':>12}  {'CA %':>8}")
        print("  " + "-" * 58)
        for r in rows:
            print(f"  {r['model']:<15}  {r['total_params']:>12,}  "
                  f"{r['cross_attn_params']:>12,}  {r['cross_attn_pct']:>7.2f}%")
        print("=" * 65)


if __name__ == "__main__":
    main()
