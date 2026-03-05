"""
models/custom_vlm.py
=====================
Advanced Master-Hack — Visual Prefix-Tuning (Shakespeare + ViT)

Architecture: A frozen pre-trained ViT (google/vit-base-patch16-224-in21k)
is fused with a custom character-level causal Transformer decoder trained on
Shakespeare text. A trainable MLP projection layer bridges the ViT's
768-dim output to the decoder's 384-dim embedding space.

MODALITY FUSION:
  ViT → Project(768→384) → [visual_prefix | char_embeddings] → CausalSelfAttention
  
TRAINING REGIME:
  - ViT:              FROZEN (always)
  - Shakespeare Decoder: UNFROZEN during fine-tuning (adapts to COCO captions)
  - visual_projection:   TRAINABLE (learned bridge)

Weight Loading Strategy:
  The Shakespeare checkpoint uses a custom per-head architecture with keys like:
    blocks.N.sa_head.heads.M.{key,query,value}.weight
  These are remapped to PyTorch nn.TransformerEncoder's fused format:
    decoder_blocks.layers.N.self_attn.in_proj_weight
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel


# ─────────────────────────────────────────────────────────────────────────────
# Character Vocabulary Helper
# ─────────────────────────────────────────────────────────────────────────────

def build_char_vocab(text_corpus: str):
    """
    Build a character-level vocabulary from a raw text corpus string.

    Returns:
        chars        : sorted list of unique characters
        char_to_idx  : dict mapping char → int index
        idx_to_char  : dict mapping int index → char
        vocab_size   : int
    """
    chars = sorted(set(text_corpus))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}
    return chars, char_to_idx, idx_to_char, len(chars)


# ─────────────────────────────────────────────────────────────────────────────
# Model Definition
# ─────────────────────────────────────────────────────────────────────────────

class CustomVLM(nn.Module):
    """
    Visual Prefix-Tuning VLM.

    Combines:
      1. Frozen ViT image encoder  (768-dim output)
      2. Trainable MLP projection  (768 → text_embed_dim)
      3. Character-level causal Transformer decoder
         (initialized from shakespeare_transformer.pt, then fine-tuned)
    """

    NUM_VISUAL_TOKENS = 197   # ViT: 196 patches + 1 [CLS]

    def __init__(self, vocab_size, text_embed_dim=384, n_heads=8, n_layers=8,
                 block_size=256, dropout=0.1):
        super().__init__()

        # ── 1. Vision Encoder (Frozen) ──────────────────────────────────────
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        for param in self.vit.parameters():
            param.requires_grad = False

        vit_hidden_size = self.vit.config.hidden_size  # 768

        # ── 2. Trainable Bridge (MLP — like LLaVA) ──────────────────────────
        self.visual_projection = nn.Sequential(
            nn.Linear(vit_hidden_size, vit_hidden_size * 2),
            nn.GELU(),
            nn.Linear(vit_hidden_size * 2, text_embed_dim)
        )

        # ── 3. Character-Level Causal Transformer Decoder ───────────────────
        self.token_embedding_table = nn.Embedding(vocab_size, text_embed_dim)
        # Position table covers visual prefix (197) + max text (block_size)
        self.position_embedding_table = nn.Embedding(
            self.NUM_VISUAL_TOKENS + block_size, text_embed_dim
        )

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=text_embed_dim,
            nhead=n_heads,
            dim_feedforward=4 * text_embed_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder_blocks = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)

        self.ln_f = nn.LayerNorm(text_embed_dim)
        self.lm_head = nn.Linear(text_embed_dim, vocab_size)

        self.block_size = block_size
        self.text_embed_dim = text_embed_dim
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.n_layers = n_layers

    # ─────────────────────────────────────────────────────────────────────────
    # Weight Loading — with architecture remapping
    # ─────────────────────────────────────────────────────────────────────────

    def load_shakespeare_weights(self, path: str, device: str = "cpu") -> dict:
        """
        Load pre-trained Shakespeare Transformer weights with full key remapping.

        The Shakespeare checkpoint uses a custom per-head architecture:
          blocks.N.sa_head.heads.M.{key,query,value}.weight  (head_dim, embed_dim)
          blocks.N.sa_head.proj.{weight,bias}
          blocks.N.ffwd.net.{0,2}.{weight,bias}
          blocks.N.ln{1,2}.{weight,bias}

        These are remapped into PyTorch nn.TransformerEncoder's fused format:
          decoder_blocks.layers.N.self_attn.in_proj_weight  (3*embed_dim, embed_dim)
          decoder_blocks.layers.N.self_attn.out_proj.{weight,bias}
          decoder_blocks.layers.N.linear1.{weight,bias}
          decoder_blocks.layers.N.linear2.{weight,bias}
          decoder_blocks.layers.N.norm1.{weight,bias}
          decoder_blocks.layers.N.norm2.{weight,bias}
        """
        print(f"📖 Loading Shakespeare weights from: {path}")

        raw = torch.load(path, map_location=device)

        # Unwrap common checkpoint structures
        if isinstance(raw, dict):
            if "model_state" in raw:
                state_dict = raw["model_state"]
            elif "model" in raw:
                state_dict = raw["model"]
            elif "state_dict" in raw:
                state_dict = raw["state_dict"]
            else:
                state_dict = raw
        else:
            raise TypeError(f"Unexpected checkpoint type: {type(raw)}")

        # ── Discover Shakespeare architecture ────────────────────────────────
        shk_blocks = set()
        shk_heads = set()
        for key in state_dict:
            if key.startswith("blocks."):
                parts = key.split(".")
                shk_blocks.add(int(parts[1]))
                if "heads" in key:
                    shk_heads.add(int(parts[4]))

        n_shk_blocks = len(shk_blocks)
        n_shk_heads = len(shk_heads) if shk_heads else self.n_heads
        head_dim = self.text_embed_dim // self.n_heads

        print(f"  📊 Shakespeare arch: {n_shk_blocks} blocks, {n_shk_heads} heads, "
              f"head_dim={head_dim}")
        print(f"  📊 Model arch: {self.n_layers} layers, {self.n_heads} heads")

        # How many layers to load (min of checkpoint and model)
        n_load = min(n_shk_blocks, self.n_layers)
        n_heads_load = min(n_shk_heads, self.n_heads)

        remapped = {}

        # ── Remap decoder blocks ─────────────────────────────────────────────
        for layer_idx in range(n_load):
            prefix_src = f"blocks.{layer_idx}"
            prefix_dst = f"decoder_blocks.layers.{layer_idx}"

            # 1. Self-Attention: Fuse per-head Q, K, V into in_proj_weight
            #    Shakespeare: heads.M.query.weight (head_dim, embed_dim)
            #    Target: self_attn.in_proj_weight (3*embed_dim, embed_dim)
            q_parts, k_parts, v_parts = [], [], []
            for h in range(n_heads_load):
                qk = f"{prefix_src}.sa_head.heads.{h}.query.weight"
                kk = f"{prefix_src}.sa_head.heads.{h}.key.weight"
                vk = f"{prefix_src}.sa_head.heads.{h}.value.weight"
                if qk in state_dict and kk in state_dict and vk in state_dict:
                    q_parts.append(state_dict[qk])
                    k_parts.append(state_dict[kk])
                    v_parts.append(state_dict[vk])

            if q_parts:
                # Concatenate heads: each (head_dim, embed_dim) → (embed_dim, embed_dim)
                Q_full = torch.cat(q_parts, dim=0)  # (n_heads*head_dim, embed_dim)
                K_full = torch.cat(k_parts, dim=0)
                V_full = torch.cat(v_parts, dim=0)
                # Fuse into in_proj_weight: [Q; K; V] → (3*embed_dim, embed_dim)
                in_proj_weight = torch.cat([Q_full, K_full, V_full], dim=0)
                remapped[f"{prefix_dst}.self_attn.in_proj_weight"] = in_proj_weight

                # Create zero bias (Shakespeare has no Q/K/V bias)
                remapped[f"{prefix_dst}.self_attn.in_proj_bias"] = torch.zeros(
                    3 * self.text_embed_dim
                )

            # 2. Output projection
            proj_w = f"{prefix_src}.sa_head.proj.weight"
            proj_b = f"{prefix_src}.sa_head.proj.bias"
            if proj_w in state_dict:
                remapped[f"{prefix_dst}.self_attn.out_proj.weight"] = state_dict[proj_w]
            if proj_b in state_dict:
                remapped[f"{prefix_dst}.self_attn.out_proj.bias"] = state_dict[proj_b]

            # 3. Feed-Forward Network
            #    Shakespeare: ffwd.net.0 → linear1, ffwd.net.2 → linear2
            for shk_idx, tgt_name in [("0", "linear1"), ("2", "linear2")]:
                wk = f"{prefix_src}.ffwd.net.{shk_idx}.weight"
                bk = f"{prefix_src}.ffwd.net.{shk_idx}.bias"
                if wk in state_dict:
                    remapped[f"{prefix_dst}.{tgt_name}.weight"] = state_dict[wk]
                if bk in state_dict:
                    remapped[f"{prefix_dst}.{tgt_name}.bias"] = state_dict[bk]

            # 4. Layer Norms: ln1 → norm1, ln2 → norm2
            for shk_ln, tgt_ln in [("ln1", "norm1"), ("ln2", "norm2")]:
                for suffix in ("weight", "bias"):
                    sk = f"{prefix_src}.{shk_ln}.{suffix}"
                    if sk in state_dict:
                        remapped[f"{prefix_dst}.{tgt_ln}.{suffix}"] = state_dict[sk]

        # ── Non-decoder module weights ───────────────────────────────────────
        # token_embedding_table
        if "token_embedding_table.weight" in state_dict:
            shk_emb = state_dict["token_embedding_table.weight"]
            own_emb = self.token_embedding_table.weight
            if shk_emb.shape == own_emb.shape:
                remapped["token_embedding_table.weight"] = shk_emb
            elif shk_emb.shape[1] == own_emb.shape[1]:
                # Vocab size difference: copy what fits
                n_copy = min(shk_emb.shape[0], own_emb.shape[0])
                new_emb = own_emb.data.clone()
                new_emb[:n_copy] = shk_emb[:n_copy]
                remapped["token_embedding_table.weight"] = new_emb

        # position_embedding_table: Shakespeare (256, 384) → Model (453, 384)
        if "position_embedding_table.weight" in state_dict:
            shk_pos = state_dict["position_embedding_table.weight"]  # (256, 384)
            own_pos = self.position_embedding_table.weight           # (197+block_size, 384)
            if shk_pos.shape == own_pos.shape:
                remapped["position_embedding_table.weight"] = shk_pos
            else:
                # Expand: zero-init the full table, then copy Shakespeare positions
                # into the TEXT portion (positions 197..197+256)
                new_pos = torch.zeros_like(own_pos.data)
                # Visual positions (0..196) get small random init
                nn.init.normal_(new_pos[:self.NUM_VISUAL_TOKENS], std=0.02)
                # Text positions: copy Shakespeare's first N positions
                n_text_slots = own_pos.shape[0] - self.NUM_VISUAL_TOKENS
                n_copy = min(shk_pos.shape[0], n_text_slots)
                new_pos[self.NUM_VISUAL_TOKENS:self.NUM_VISUAL_TOKENS + n_copy] = shk_pos[:n_copy]
                remapped["position_embedding_table.weight"] = new_pos
                print(f"  📐 Position embeddings expanded: {shk_pos.shape} → {own_pos.shape}")

        # ln_f (final layer norm)
        for suffix in ("weight", "bias"):
            k = f"ln_f.{suffix}"
            if k in state_dict:
                own_shape = getattr(self.ln_f, suffix).shape
                if state_dict[k].shape == own_shape:
                    remapped[k] = state_dict[k]

        # lm_head
        if "lm_head.weight" in state_dict:
            shk_lm = state_dict["lm_head.weight"]
            own_lm = self.lm_head.weight
            if shk_lm.shape == own_lm.shape:
                remapped["lm_head.weight"] = shk_lm
            elif shk_lm.shape[1] == own_lm.shape[1]:
                n_copy = min(shk_lm.shape[0], own_lm.shape[0])
                new_lm = own_lm.data.clone()
                new_lm[:n_copy] = shk_lm[:n_copy]
                remapped["lm_head.weight"] = new_lm

        if "lm_head.bias" in state_dict:
            shk_b = state_dict["lm_head.bias"]
            own_b = self.lm_head.bias
            if own_b is not None and shk_b.shape == own_b.shape:
                remapped["lm_head.bias"] = shk_b
            elif own_b is not None:
                n_copy = min(shk_b.shape[0], own_b.shape[0])
                new_b = own_b.data.clone()
                new_b[:n_copy] = shk_b[:n_copy]
                remapped["lm_head.bias"] = new_b

        # ── Load remapped weights ─────────────────────────────────────────────
        # Verify shapes before loading
        own_state = self.state_dict()
        valid_remapped = {}
        shape_mismatches = []
        for k, v in remapped.items():
            if k in own_state:
                if own_state[k].shape == v.shape:
                    valid_remapped[k] = v
                else:
                    shape_mismatches.append(
                        f"    {k}: ckpt={v.shape} vs model={own_state[k].shape}"
                    )
            else:
                shape_mismatches.append(f"    {k}: not in model state_dict")

        result = self.load_state_dict(valid_remapped, strict=False)

        print(f"  ✅ Successfully loaded {len(valid_remapped)} weight tensors (of {len(state_dict)} in checkpoint)")

        if shape_mismatches:
            print(f"  ⚠️  {len(shape_mismatches)} shape mismatches (skipped):")
            for msg in shape_mismatches[:5]:
                print(msg)

        # Count decoder keys that were successfully loaded
        decoder_loaded = sum(1 for k in valid_remapped if k.startswith("decoder_blocks"))
        total_decoder = sum(1 for k in own_state if k.startswith("decoder_blocks"))
        print(f"  📊 Decoder coverage: {decoder_loaded}/{total_decoder} tensors loaded")

        return {
            "loaded": list(valid_remapped.keys()),
            "missing": result.missing_keys,
            "unexpected": result.unexpected_keys,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Freezing / Unfreezing / Parameter Counting
    # ─────────────────────────────────────────────────────────────────────────

    def freeze_decoder(self):
        """Freeze the Shakespeare decoder so only visual_projection trains."""
        for name, param in self.named_parameters():
            if not name.startswith("visual_projection"):
                param.requires_grad = False
        # Ensure ViT is frozen
        for param in self.vit.parameters():
            param.requires_grad = False

    def unfreeze_decoder(self):
        """
        Unfreeze the decoder for fine-tuning while keeping ViT frozen.
        
        This allows the decoder to adapt from Shakespeare text to COCO captions.
        The visual_projection is also trainable.
        """
        # First, freeze everything
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze visual_projection (always trainable)
        for param in self.visual_projection.parameters():
            param.requires_grad = True

        # Unfreeze ALL decoder components
        for param in self.token_embedding_table.parameters():
            param.requires_grad = True
        for param in self.position_embedding_table.parameters():
            param.requires_grad = True
        for param in self.decoder_blocks.parameters():
            param.requires_grad = True
        for param in self.ln_f.parameters():
            param.requires_grad = True
        for param in self.lm_head.parameters():
            param.requires_grad = True

        # ViT stays FROZEN
        for param in self.vit.parameters():
            param.requires_grad = False

    def get_param_groups(self, projection_lr=1e-4, decoder_lr=5e-5):
        """
        Return optimizer param groups with discriminative learning rates.
        
        - visual_projection: higher LR (learning from scratch)
        - decoder: lower LR (gentle adaptation from Shakespeare)
        """
        projection_params = []
        decoder_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("visual_projection"):
                projection_params.append(param)
            else:
                decoder_params.append(param)

        return [
            {"params": projection_params, "lr": projection_lr},
            {"params": decoder_params, "lr": decoder_lr},
        ]

    def trainable_params(self):
        """Return count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ─────────────────────────────────────────────────────────────────────────
    # Forward Pass
    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, pixel_values, text_input_ids, text_targets=None):
        B, T = text_input_ids.shape

        # ── Image Encoding (frozen ViT) ──────────────────────────────────────
        with torch.no_grad():
            vit_outputs = self.vit(pixel_values=pixel_values)
        image_embeds = vit_outputs.last_hidden_state  # (B, 197, 768)

        # ── Project to text embedding space ──────────────────────────────────
        visual_prefix = self.visual_projection(image_embeds)  # (B, 197, 384)
        num_visual = visual_prefix.shape[1]                   # 197

        # ── Text Embeddings ───────────────────────────────────────────────────
        T_clipped = min(T, self.block_size)
        text_in = text_input_ids[:, :T_clipped]
        tok_emb = self.token_embedding_table(text_in)         # (B, T, 384)

        # ── Positional Embeddings (covers full combined sequence) ─────────────
        # Positions 0..196 → visual prefix, 197..197+T → text tokens
        total_len = num_visual + T_clipped
        pos_ids = torch.arange(total_len, device=text_in.device)
        pos_emb = self.position_embedding_table(pos_ids)      # (num_visual+T, 384)

        vis_pos = pos_emb[:num_visual]                        # (197, 384)
        txt_pos = pos_emb[num_visual:]                        # (T, 384)

        visual_emb = visual_prefix + vis_pos                  # (B, 197, 384)
        text_emb   = tok_emb + txt_pos                        # (B, T, 384)

        # ── Fusion: [visual_prefix | text_emb] ───────────────────────────────
        combined = torch.cat([visual_emb, text_emb], dim=1)   # (B, 197+T, 384)
        tot = combined.shape[1]

        # ── Causal Attention Mask ─────────────────────────────────────────────
        # Visual tokens attend to each other freely.
        # Text tokens attend to all visual tokens + causally to previous text.
        mask = torch.full((tot, tot), float("-inf"), device=text_in.device)
        mask[:num_visual, :num_visual] = 0.0          # visual→visual: free
        mask[num_visual:, :num_visual] = 0.0           # text→visual: free
        causal = torch.triu(
            torch.full((T_clipped, T_clipped), float("-inf"), device=text_in.device),
            diagonal=1,
        )
        mask[num_visual:, num_visual:] = causal         # text→text: causal

        # ── Decoder ───────────────────────────────────────────────────────────
        x = self.decoder_blocks(combined, mask=mask, is_causal=False)
        text_out = x[:, num_visual:, :]
        text_out = self.ln_f(text_out)
        logits = self.lm_head(text_out)                       # (B, T, vocab)

        # ── Loss (ignore padding index 0) ─────────────────────────────────────
        loss = None
        if text_targets is not None:
            tgt = text_targets[:, :T_clipped]
            loss = F.cross_entropy(
                logits.reshape(B * T_clipped, -1),
                tgt.reshape(B * T_clipped),
                ignore_index=0,
            )

        return logits, loss

    # ─────────────────────────────────────────────────────────────────────────
    # Generation
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, pixel_values, char_to_idx, idx_to_char,
                 max_new_tokens=100, temperature=0.8):
        """
        Autoregressive character-level caption generation (temperature sampling).

        Args:
            pixel_values   : (1, 3, H, W) pre-processed image tensor
            char_to_idx    : character → index mapping
            idx_to_char    : index → character mapping
            max_new_tokens : how many characters to generate
            temperature    : sampling temperature (0.8 = slightly sharper than uniform)

        Returns:
            generated_text : str
        """
        self.eval()
        device = pixel_values.device

        bos_idx = char_to_idx.get("\n", 0)
        idx_seq = torch.tensor([[bos_idx]], dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            # Clip text to block_size — the forward method handles the visual
            # prefix separately, so we only need to limit the text portion.
            idx_cond = idx_seq[:, -self.block_size:]
            logits, _ = self(pixel_values, idx_cond)
            # Take the last time step
            logits_last = logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(logits_last, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx_seq = torch.cat([idx_seq, next_idx], dim=1)

        # Decode, skip the leading BOS
        generated = "".join(
            idx_to_char.get(i.item(), "?") for i in idx_seq[0, 1:]
        )
        return generated

    @torch.no_grad()
    def generate_beam(self, pixel_values, char_to_idx, idx_to_char,
                      max_new_tokens=100, num_beams=4, length_penalty=1.0):
        """
        Beam-search character-level caption generation.

        At each step we keep the top `num_beams` partial sequences ranked by
        cumulative log-probability (with optional length penalty).

        Args:
            pixel_values   : (1, 3, H, W) image tensor
            char_to_idx    : char → idx mapping
            idx_to_char    : idx → char mapping
            max_new_tokens : max characters to generate
            num_beams      : beam width (1 = greedy)
            length_penalty : >1 favors longer sequences; <1 favors shorter

        Returns:
            generated_text : str (best beam)
        """
        self.eval()
        device = pixel_values.device

        bos_idx = char_to_idx.get("\n", 0)
        # Each beam: (score, token_sequence_tensor)
        beams = [(0.0, torch.tensor([[bos_idx]], dtype=torch.long, device=device))]

        for _ in range(max_new_tokens):
            candidates = []
            for score, seq in beams:
                idx_cond = seq[:, -self.block_size:]
                logits, _ = self(pixel_values, idx_cond)
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1)  # (1, vocab)
                topk_probs, topk_ids = log_probs.topk(num_beams, dim=-1)

                for k in range(num_beams):
                    new_score = score + topk_probs[0, k].item()
                    new_seq = torch.cat(
                        [seq, topk_ids[:, k:k+1]], dim=1
                    )
                    candidates.append((new_score, new_seq))

            # Apply length penalty and keep top beams
            candidates.sort(
                key=lambda x: x[0] / (x[1].shape[1] ** length_penalty),
                reverse=True,
            )
            beams = candidates[:num_beams]

        best_seq = beams[0][1]
        return "".join(idx_to_char.get(i.item(), "?") for i in best_seq[0, 1:])
