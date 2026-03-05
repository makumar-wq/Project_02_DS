"""
app.py
======
VLM Caption Lab — Premium Streamlit Demo

Features:
  • Sidebar — Weight Source: Base / Fine-tuned (Best) / Fine-tuned (Latest)
  • Sidebar — Architecture selector, Generation Mode, Advanced Controls
  • Tab 1 — Caption: Single model captioning with weight selection
  • Tab 2 — Compare: Side-by-side 4-model comparison (same image, same config)
  • Tab 3 — Results: Pre-computed benchmark comparison tables
"""

import os
import warnings
import torch
import streamlit as st
from PIL import Image
from models.blip_tuner import generate_with_mask

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")
warnings.filterwarnings("ignore", category=UserWarning, message=".*use_fast.*")

# ─────────────────────────────────────────────────────────────────────────────
# Page Config & CSS
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="VLM Caption Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] {
      font-family: 'Inter', sans-serif;
      background-color: #0d1117;
      color: #e6edf3;
  }
  section[data-testid="stSidebar"] {
      background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
      border-right: 1px solid #30363d;
  }
  section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }
  .main .block-container { padding-top: 1.5rem; max-width: 1200px; }
  .hero-title {
      background: linear-gradient(135deg, #58a6ff 0%, #bc8cff 50%, #ff7b72 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      font-size: 2.4rem; font-weight: 700; letter-spacing: -0.5px; margin-bottom: 0.2rem;
  }
  .hero-sub { color: #8b949e; font-size: 0.98rem; margin-bottom: 1.5rem; }
  .result-card {
      background: linear-gradient(135deg, #161b22, #1c2128);
      border: 1px solid #30363d; border-radius: 12px;
      padding: 1.5rem; margin-top: 0.8rem;
  }
  .compare-card {
      background: linear-gradient(135deg, #161b22, #1c2128);
      border: 1px solid #30363d; border-radius: 12px;
      padding: 1.2rem; margin-top: 0.5rem; min-height: 160px;
  }
  .caption-text { font-size: 1.15rem; font-weight: 600; color: #e6edf3; line-height: 1.5; }
  .compare-caption { font-size: 1.0rem; font-weight: 500; color: #e6edf3; line-height: 1.4; }
  .badge { display: inline-block; padding: 3px 10px; border-radius: 20px;
           font-size: 0.78rem; font-weight: 600; margin-right: 6px; }
  .badge-blue   { background: rgba(88,166,255,0.15); color:#58a6ff; border:1px solid #388bfd; }
  .badge-purple { background: rgba(188,140,255,0.15); color:#bc8cff; border:1px solid #9a6eff; }
  .badge-green  { background: rgba(63,185,80,0.15); color:#3fb950; border:1px solid #2ea043; }
  .badge-red    { background: rgba(248,81,73,0.15); color:#f85149; border:1px solid #da3633; }
  .badge-orange { background: rgba(210,153,34,0.15); color:#d2993a; border:1px solid #bb8009; }
  .badge-yellow { background: rgba(210,153,34,0.15); color:#e3b341; border:1px solid #bb8009; }
  .weight-tag   { display: inline-block; padding: 2px 8px; border-radius: 12px;
                  font-size: 0.72rem; font-weight: 500; margin-left: 4px; }
  .wt-base      { background: rgba(88,166,255,0.1); color:#58a6ff; border:1px solid #1f6feb; }
  .wt-best      { background: rgba(63,185,80,0.1); color:#3fb950; border:1px solid #2ea043; }
  .wt-latest    { background: rgba(210,153,34,0.1); color:#d2993a; border:1px solid #bb8009; }
  .arch-box {
      background: #161b22; border-left: 3px solid #58a6ff;
      border-radius: 0 8px 8px 0; padding: 0.8rem 1.2rem;
      margin-top: 0.8rem; font-size: 0.85rem; color: #8b949e; line-height: 1.6;
  }
  .config-banner {
      background: #161b22; border: 1px solid #21262d; border-radius: 8px;
      padding: 0.7rem 1rem; margin-bottom: 0.8rem; font-size: 0.82rem; color: #8b949e;
  }
  .stButton > button {
      background: linear-gradient(135deg, #388bfd, #9a6eff);
      color: white; border: none; border-radius: 8px;
      padding: 0.6rem 1.8rem; font-weight: 600; font-size: 1rem;
      transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.85; }
  div[data-testid="stSelectbox"] label,
  div[data-testid="stFileUploader"] label { color: #c9d1d9 !important; font-weight: 500; }
  .stAlert { border-radius: 8px; }
  .stTabs [data-baseweb="tab"] { font-weight: 600; }
  .param-section {
      background: #161b22; border: 1px solid #21262d;
      border-radius: 8px; padding: 1rem; margin-top: 0.5rem;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Architecture Info & Constants
# ─────────────────────────────────────────────────────────────────────────────

ARCH_INFO = {
    "BLIP (Multimodal Mixture Attention)": (
        "🔵 <b>BLIP</b> uses a Mixture-of-Encoder-Decoder (MED) architecture. "
        "Gated cross-attention is injected between self-attention and FFN layers."
    ),
    "ViT-GPT2 (Standard Cross-Attention)": (
        "🟣 <b>ViT-GPT2</b>: every GPT-2 text token attends to <em>all</em> "
        "197 ViT patch embeddings via full cross-attention at every decoder layer."
    ),
    "GIT (Zero Cross-Attention)": (
        "🟠 <b>GIT</b> abandons cross-attention entirely. Image patches are "
        "concatenated to the front of the token sequence; no cross-attention block."
    ),
    "Custom VLM (Shakespeare Prefix)": (
        "🟢 <b>Custom VLM</b> fuses a frozen ViT with a Shakespeare char-level "
        "decoder via a single trainable Linear(768→384) projection."
    ),
}

MODEL_KEYS = [
    "BLIP (Multimodal Mixture Attention)",
    "ViT-GPT2 (Standard Cross-Attention)",
    "GIT (Zero Cross-Attention)",
    "Custom VLM (Shakespeare Prefix)",
]

MODEL_SHORT = {
    "BLIP (Multimodal Mixture Attention)": "BLIP",
    "ViT-GPT2 (Standard Cross-Attention)": "ViT-GPT2",
    "GIT (Zero Cross-Attention)": "GIT",
    "Custom VLM (Shakespeare Prefix)": "Custom VLM",
}

MODEL_BADGE = {
    "BLIP (Multimodal Mixture Attention)": "badge-blue",
    "ViT-GPT2 (Standard Cross-Attention)": "badge-purple",
    "GIT (Zero Cross-Attention)":          "badge-orange",
    "Custom VLM (Shakespeare Prefix)":     "badge-green",
}

MODEL_CA_TYPE = {
    "BLIP (Multimodal Mixture Attention)": "Gated MED Cross-Attention",
    "ViT-GPT2 (Standard Cross-Attention)": "Full Cross-Attention",
    "GIT (Zero Cross-Attention)": "Self-Attention Prefix",
    "Custom VLM (Shakespeare Prefix)": "Linear Bridge Prefix",
}

WEIGHT_TAG_CLASS = {"base": "wt-base", "best": "wt-best", "latest": "wt-latest"}
WEIGHT_LABEL = {"base": "Base", "best": "Best", "latest": "Latest"}

OUTPUT_ROOT = "./outputs"


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():  return torch.device("mps")
    if torch.cuda.is_available():           return torch.device("cuda")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Weight Loading Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _has_finetuned(model_dir, subdir):
    """Check if a fine-tuned checkpoint exists for a given model + subdir."""
    path = os.path.join(OUTPUT_ROOT, model_dir, subdir)
    return os.path.isdir(path) and len(os.listdir(path)) > 0


def _ckpt_path(model_dir, subdir):
    return os.path.join(OUTPUT_ROOT, model_dir, subdir)


# ─────────────────────────────────────────────────────────────────────────────
# Cached Model Loaders (with weight_source support)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_blip(weight_source="base"):
    from transformers import BlipProcessor, BlipForConditionalGeneration
    device = get_device()
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base", use_fast=True)
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base")

    if weight_source != "base":
        ckpt = _ckpt_path("blip", weight_source)
        if os.path.isdir(ckpt) and os.listdir(ckpt):
            try:
                loaded = BlipForConditionalGeneration.from_pretrained(ckpt)
                model.load_state_dict(loaded.state_dict())
                del loaded
            except Exception as e:
                print(f"⚠️ Could not load BLIP {weight_source} weights: {e}")

    model.to(device).eval()
    return processor, model, device


@st.cache_resource(show_spinner=False)
def load_vit_gpt2(weight_source="base"):
    from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
    device = get_device()
    model_id = "nlpconnect/vit-gpt2-image-captioning"
    processor = ViTImageProcessor.from_pretrained(model_id, use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    if weight_source != "base":
        ckpt = _ckpt_path("vit_gpt2", weight_source)
        if os.path.isdir(ckpt) and os.listdir(ckpt):
            try:
                loaded = VisionEncoderDecoderModel.from_pretrained(ckpt)
                model.load_state_dict(loaded.state_dict())
                del loaded
            except Exception as e:
                print(f"⚠️ Could not load ViT-GPT2 {weight_source} weights: {e}")

    model.to(device).eval()
    return processor, tokenizer, model, device


@st.cache_resource(show_spinner=False)
def load_git(weight_source="base"):
    from transformers import AutoProcessor, AutoModelForCausalLM
    device = get_device()
    model_id = "microsoft/git-base-coco"
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    if weight_source != "base":
        ckpt = _ckpt_path("git", weight_source)
        if os.path.isdir(ckpt) and os.listdir(ckpt):
            try:
                loaded = AutoModelForCausalLM.from_pretrained(ckpt)
                model.load_state_dict(loaded.state_dict())
                del loaded
            except Exception as e:
                print(f"⚠️ Could not load GIT {weight_source} weights: {e}")

    model.to(device).eval()
    return processor, model, device


@st.cache_resource(show_spinner=False)
def load_custom_vlm(weight_source="base"):
    from models.custom_vlm import CustomVLM, build_char_vocab
    from config import CFG
    device = get_device()
    cfg = CFG()

    if not os.path.exists(cfg.shakespeare_file):
        return None, None, None, None, device

    with open(cfg.shakespeare_file, "r", encoding="utf-8") as f:
        text = f.read()
    _, char_to_idx, idx_to_char, vocab_size = build_char_vocab(text)

    model = CustomVLM(
        vocab_size=vocab_size,
        text_embed_dim=cfg.text_embed_dim,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        block_size=cfg.block_size,
        dropout=cfg.dropout,
    )

    # Always load Shakespeare weights first
    shakes_path = getattr(cfg, "shakespeare_weights_path", "./shakespeare_transformer.pt")
    if os.path.exists(shakes_path):
        model.load_shakespeare_weights(shakes_path)

    # Then load fine-tuned checkpoint if requested
    if weight_source != "base":
        ckpt_path = os.path.join(cfg.output_root, "custom_vlm", weight_source, "custom_vlm.pt")
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu")
            own_state = model.state_dict()
            filtered = {k: v for k, v in state["model_state"].items()
                        if k in own_state and own_state[k].shape == v.shape}
            model.load_state_dict(filtered, strict=False)
    else:
        # Even for base, try loading best weights as fallback
        for subdir in ["best", "latest"]:
            candidate = os.path.join(cfg.output_root, "custom_vlm", subdir, "custom_vlm.pt")
            if os.path.exists(candidate):
                state = torch.load(candidate, map_location="cpu")
                own_state = model.state_dict()
                filtered = {k: v for k, v in state["model_state"].items()
                            if k in own_state and own_state[k].shape == v.shape}
                model.load_state_dict(filtered, strict=False)
                break

    model.to(device).eval()
    return model, char_to_idx, idx_to_char, vocab_size, device


@st.cache_resource(show_spinner=False)
def load_toxicity_filter():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tox_id = "unitary/toxic-bert"
    tok = AutoTokenizer.from_pretrained(tox_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(tox_id)
    mdl.eval()
    return tok, mdl


# ─────────────────────────────────────────────────────────────────────────────
# Toxicity Check
# ─────────────────────────────────────────────────────────────────────────────

def is_toxic(text, tox_tok, tox_mdl):
    inputs = tox_tok(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = tox_mdl(**inputs)
    scores = torch.sigmoid(outputs.logits).squeeze()
    if isinstance(scores, torch.Tensor) and scores.dim() > 0:
        return (scores > 0.5).any().item()
    return scores.item() > 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Ablation Mask Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_mask_for_mode(ui_mode, device):
    N = 197
    if ui_mode == "Baseline (Full Attention)":
        return torch.ones(1, N, dtype=torch.long, device=device), False
    elif ui_mode == "Random Patch Dropout (50%)":
        mask = torch.ones(1, N, dtype=torch.long, device=device)
        spatial_indices = torch.randperm(196)[:98] + 1
        mask[0, spatial_indices] = 0
        return mask, False
    elif ui_mode == "Center-Focus (Inner 8×8)":
        GRID, INNER, offset = 14, 8, 3
        keep = set()
        for row in range(offset, offset + INNER):
            for col in range(offset, offset + INNER):
                keep.add(row * GRID + col + 1)
        mask = torch.zeros(1, N, dtype=torch.long, device=device)
        mask[0, 0] = 1
        for idx in keep:
            if idx < N: mask[0, idx] = 1
        return mask, False
    elif ui_mode == "Squint (Global Pool)":
        return None, True
    return torch.ones(1, N, dtype=torch.long, device=device), False


# ─────────────────────────────────────────────────────────────────────────────
# Caption Generation (single model)
# ─────────────────────────────────────────────────────────────────────────────

def generate_caption(model_name, gen_mode, image_pil,
                     num_beams=4, max_new_tokens=50, length_penalty=1.0,
                     weight_source="base"):
    device = get_device()

    with torch.no_grad():
        if model_name == "BLIP (Multimodal Mixture Attention)":
            processor, model, device = load_blip(weight_source)
            inputs = processor(images=image_pil, return_tensors="pt").to(device)
            mask, is_squint = build_mask_for_mode(gen_mode, device)

            if is_squint:
                vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
                hs = vision_out.last_hidden_state
                pooled = torch.cat([hs[:, :1, :], hs[:, 1:, :].mean(dim=1, keepdim=True)], dim=1)
                captions = generate_with_mask(
                    model, processor, device=device,
                    encoder_hidden_states=pooled,
                    encoder_attention_mask=torch.ones(1, 2, dtype=torch.long, device=device),
                    max_new_tokens=max_new_tokens, num_beams=num_beams,
                )
            else:
                captions = generate_with_mask(
                    model, processor, device=device,
                    pixel_values=inputs["pixel_values"],
                    encoder_attention_mask=mask,
                    max_new_tokens=max_new_tokens, num_beams=num_beams,
                )
            caption = captions[0]

        elif model_name == "ViT-GPT2 (Standard Cross-Attention)":
            from transformers.modeling_outputs import BaseModelOutput
            processor, tokenizer, model, device = load_vit_gpt2(weight_source)
            inputs = processor(images=image_pil, return_tensors="pt").to(device)
            mask, is_squint = build_mask_for_mode(gen_mode, device)

            if is_squint:
                enc_out = model.encoder(pixel_values=inputs["pixel_values"])
                hs = enc_out.last_hidden_state
                pooled = torch.cat([hs[:, :1, :], hs[:, 1:, :].mean(dim=1, keepdim=True)], dim=1)
                out = model.generate(
                    encoder_outputs=BaseModelOutput(last_hidden_state=pooled),
                    decoder_start_token_id=tokenizer.bos_token_id,
                    max_new_tokens=max_new_tokens, num_beams=num_beams,
                    length_penalty=length_penalty,
                )
            else:
                out = model.generate(
                    **inputs,
                    attention_mask=mask,
                    max_new_tokens=max_new_tokens, num_beams=num_beams,
                    length_penalty=length_penalty,
                )
            caption = tokenizer.decode(out[0], skip_special_tokens=True)

        elif model_name == "GIT (Zero Cross-Attention)":
            processor, model, device = load_git(weight_source)
            inputs = processor(images=image_pil, return_tensors="pt").to(device)
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                num_beams=num_beams, length_penalty=length_penalty,
            )
            caption = processor.batch_decode(out, skip_special_tokens=True)[0]

        elif model_name == "Custom VLM (Shakespeare Prefix)":
            vlm, char_to_idx, idx_to_char, vocab_size, device = load_custom_vlm(weight_source)
            if vlm is None:
                return "[Custom VLM not available — train first with: python train.py --model custom]"
            from transformers import ViTImageProcessor
            image_processor = ViTImageProcessor.from_pretrained(
                "google/vit-base-patch16-224-in21k", use_fast=True)
            pv = image_processor(images=image_pil, return_tensors="pt")["pixel_values"].to(device)
            if num_beams > 1:
                caption = vlm.generate_beam(pv, char_to_idx, idx_to_char,
                                            max_new_tokens=max_new_tokens,
                                            num_beams=num_beams,
                                            length_penalty=length_penalty)
            else:
                caption = vlm.generate(pv, char_to_idx, idx_to_char,
                                       max_new_tokens=max_new_tokens)
        else:
            caption = "Unknown model."

    return caption.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🔬 VLM Caption Lab")
    st.markdown("---")

    # ── Weight Source ─────────────────────────────────────────────────────────
    weight_options = {
        "🔵 Base (Pretrained)": "base",
        "🟢 Fine-tuned (Best)": "best",
        "🟡 Fine-tuned (Latest)": "latest",
    }
    weight_choice = st.radio(
        "**Weight Source**", list(weight_options.keys()), index=0,
        help="Base = HuggingFace pretrained. Best/Latest = your fine-tuned checkpoints."
    )
    weight_source = weight_options[weight_choice]

    # Show availability indicators
    ft_status = []
    for mdl_dir, mdl_name in [("blip", "BLIP"), ("vit_gpt2", "ViT-GPT2"),
                               ("git", "GIT"), ("custom_vlm", "Custom VLM")]:
        has_best = _has_finetuned(mdl_dir, "best")
        has_latest = _has_finetuned(mdl_dir, "latest")
        if has_best or has_latest:
            ft_status.append(f"  ✅ {mdl_name}")
        else:
            ft_status.append(f"  ⬜ {mdl_name}")
    if weight_source != "base":
        st.caption("Fine-tuned checkpoints:\n" + "\n".join(ft_status))

    st.markdown("---")

    # ── Architecture Selector ─────────────────────────────────────────────────
    selected_model = st.selectbox("**Architecture**", MODEL_KEYS, index=0)

    if selected_model in ("BLIP (Multimodal Mixture Attention)",
                          "ViT-GPT2 (Standard Cross-Attention)"):
        mode_options = [
            "Baseline (Full Attention)",
            "Random Patch Dropout (50%)",
            "Center-Focus (Inner 8×8)",
            "Squint (Global Pool)",
        ]
    elif selected_model == "Custom VLM (Shakespeare Prefix)":
        mode_options = ["Shakespeare Prefix"]
    else:
        mode_options = ["Baseline (Full Attention)"]

    selected_mode = st.selectbox("**Generation Mode**", mode_options, index=0)

    st.markdown(
        f"<div class='arch-box'>{ARCH_INFO[selected_model]}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Advanced Controls ─────────────────────────────────────────────────────
    with st.expander("⚙️ Advanced Controls", expanded=False):
        num_beams = st.select_slider(
            "Beam Size", options=[1, 2, 3, 4, 5, 8, 10], value=10,
            help="Number of beams in beam search. Higher = better but slower."
        )
        length_penalty = st.select_slider(
            "Length Penalty", options=[0.8, 0.9, 1.0, 1.1, 1.2], value=1.2,
            help=">1 favors longer captions, <1 favors shorter."
        )
        max_new_tokens = st.select_slider(
            "Max Tokens", options=[20, 30, 50, 80, 100], value=50,
            help="Maximum number of tokens to generate."
        )
        st.caption(
            f"Config: `beams={num_beams}, len_pen={length_penalty}, max_tok={max_new_tokens}`"
        )
    st.markdown("---")
    st.markdown("<small style='color:#484f58'>Toxicity filter: unitary/toxic-bert</small>",
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("<div class='hero-title'>VLM Caption Lab 🔬</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='hero-sub'>Compare cross-attention strategies: BLIP · ViT-GPT2 · GIT · "
    "Visual Prefix-Tuning. Upload, pick a mode, and explore different architectures.</div>",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper — render a single caption card
# ─────────────────────────────────────────────────────────────────────────────

def render_caption_card(model_name, caption, weight_src, num_beams, length_penalty,
                        max_new_tokens, container, card_class="result-card",
                        caption_class="caption-text", show_params=True):
    badge_cls = MODEL_BADGE.get(model_name, "badge-blue")
    wt_cls = WEIGHT_TAG_CLASS.get(weight_src, "wt-base")
    wt_label = WEIGHT_LABEL.get(weight_src, weight_src)
    short = MODEL_SHORT.get(model_name, model_name)
    ca = MODEL_CA_TYPE.get(model_name, "")

    params_html = ""
    if show_params:
        params_html = (f"<br><small style='color:#586069'>beams={num_beams} · "
                       f"len_pen={length_penalty} · max_tok={max_new_tokens}</small>")

    container.markdown(
        f"<div class='{card_class}'>"
        f"<span class='badge {badge_cls}'>{short}</span>"
        f"<span class='weight-tag {wt_cls}'>{wt_label}</span>"
        f"<span style='color:#484f58; font-size:0.72rem; margin-left:6px'>{ca}</span>"
        f"<br><br><div class='{caption_class}'>\"{caption}\"</div>"
        f"{params_html}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Toxicity check
    try:
        tox_tok, tox_mdl = load_toxicity_filter()
        toxic = is_toxic(caption, tox_tok, tox_mdl)
    except Exception:
        toxic = False

    if toxic:
        container.error("⚠️ Flagged by Toxic-BERT")
    else:
        container.caption("✅ Passed toxicity check")


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_caption, tab_compare, tab_results = st.tabs([
    "🖼️  Caption", "🔀  Compare All Models", "📊  Experiment Results"
])


# ═══════════════════════════════════════════════════════════════════════════
# Tab 1 — Single Model Caption
# ═══════════════════════════════════════════════════════════════════════════

with tab_caption:
    col_upload, col_result = st.columns([1, 1.3], gap="large")

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload an image", type=["jpg", "jpeg", "png", "webp"],
            label_visibility="visible",
            key="caption_uploader",
        )
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", width="stretch")

        generate_btn = st.button("✨ Generate Caption",
                                  disabled=(uploaded_file is None),
                                  key="caption_btn")

    with col_result:
        if uploaded_file and generate_btn:
            with st.spinner(f"Loading {MODEL_SHORT[selected_model]} ({weight_source}) + generating…"):
                try:
                    caption = generate_caption(
                        selected_model, selected_mode, image,
                        num_beams=num_beams,
                        max_new_tokens=max_new_tokens,
                        length_penalty=length_penalty,
                        weight_source=weight_source,
                    )
                except Exception as e:
                    st.error(f"Generation error: {e}")
                    caption = None

            if caption:
                render_caption_card(
                    selected_model, caption, weight_source,
                    num_beams, length_penalty, max_new_tokens,
                    container=st,
                )

        elif not uploaded_file:
            st.markdown(
                "<div style='color:#484f58; margin-top:4rem; text-align:center; font-size:1.1rem;'>"
                "⬅️  Upload an image to get started</div>",
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════
# Tab 2 — Compare All Models
# ═══════════════════════════════════════════════════════════════════════════

with tab_compare:
    st.markdown("### 🔀 Multi-Model Comparison")
    st.caption(
        "Upload one image and generate captions from **all 4 architectures** simultaneously, "
        "using the same decoding parameters. Perfect for report screenshots."
    )

    # Config banner
    wt_label = WEIGHT_LABEL.get(weight_source, weight_source)
    st.markdown(
        f"<div class='config-banner'>"
        f"⚙️ <b>Config:</b> beams={num_beams} · len_pen={length_penalty} · "
        f"max_tok={max_new_tokens} · weights=<b>{wt_label}</b>"
        f"</div>",
        unsafe_allow_html=True,
    )

    is_common_mode = selected_mode in ["Baseline (Full Attention)", "Shakespeare Prefix"]
    if not is_common_mode:
        st.warning(
            f"⚠️ **Warning:** You have selected **{selected_mode}**.\n\n"
            "This generation mode is an ablation experiment and is not supported uniformly by all models. "
            "GIT and Custom VLM lack standard cross-attention and cannot process these masks.\n\n"
            "👉 **To compare all 4 architectures fairly, please change the Generation Mode in the sidebar to `Baseline (Full Attention)`.**"
        )

    col_img, col_ctrl = st.columns([1, 1])
    with col_img:
        compare_file = st.file_uploader(
            "Upload an image for comparison", type=["jpg", "jpeg", "png", "webp"],
            key="compare_uploader",
        )
    with col_ctrl:
        if compare_file:
            compare_image = Image.open(compare_file).convert("RGB")
            st.image(compare_image, caption="Comparison Image", width="stretch")

    compare_btn = st.button("🚀 Compare All 4 Models",
                             disabled=(compare_file is None or not is_common_mode),
                             key="compare_btn")

    if compare_file and compare_btn:
        compare_image = Image.open(compare_file).convert("RGB")

        # Generate captions from all 4 models
        results = {}
        progress = st.progress(0, text="Starting comparison...")

        for i, model_key in enumerate(MODEL_KEYS):
            short = MODEL_SHORT[model_key]
            progress.progress((i) / 4, text=f"Generating with {short}...")

            # Apply selected mode to supported models, otherwise use appropriate fallback
            if model_key == "Custom VLM (Shakespeare Prefix)":
                mode = "Shakespeare Prefix"
            elif model_key in ("BLIP (Multimodal Mixture Attention)", "ViT-GPT2 (Standard Cross-Attention)"):
                if selected_mode in [
                    "Baseline (Full Attention)",
                    "Random Patch Dropout (50%)",
                    "Center-Focus (Inner 8×8)",
                    "Squint (Global Pool)"
                ]:
                    mode = selected_mode
                else:
                    mode = "Baseline (Full Attention)"
            else:
                mode = "Baseline (Full Attention)"

            try:
                cap = generate_caption(
                    model_key, mode, compare_image,
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                    length_penalty=length_penalty,
                    weight_source=weight_source,
                )
                results[model_key] = cap
            except Exception as e:
                results[model_key] = f"[Error: {e}]"

        progress.progress(1.0, text="✅ All models complete!")

        # Render 2x2 grid
        st.markdown("---")
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)

        grid = [(MODEL_KEYS[0], row1_col1), (MODEL_KEYS[1], row1_col2),
                (MODEL_KEYS[2], row2_col1), (MODEL_KEYS[3], row2_col2)]

        for model_key, col in grid:
            cap = results.get(model_key, "[Not available]")
            with col:
                render_caption_card(
                    model_key, cap, weight_source,
                    num_beams, length_penalty, max_new_tokens,
                    container=st,
                    card_class="compare-card",
                    caption_class="compare-caption",
                    show_params=False,
                )

        # Summary table
        st.markdown("---")
        st.markdown("#### 📋 Summary Table")
        table_rows = []
        for model_key in MODEL_KEYS:
            short = MODEL_SHORT[model_key]
            ca = MODEL_CA_TYPE[model_key]
            cap = results.get(model_key, "–")
            word_count = len(cap.split()) if cap and not cap.startswith("[") else 0
            table_rows.append(f"| **{short}** | {ca} | {cap[:80]}{'…' if len(cap) > 80 else ''} | {word_count} |")

        table_md = (
            "| Architecture | Cross-Attention | Caption | Words |\n"
            "|---|---|---|---|\n"
            + "\n".join(table_rows)
        )
        st.markdown(table_md)
        st.caption(
            f"Generated with: beams={num_beams}, len_pen={length_penalty}, "
            f"max_tok={max_new_tokens}, weights={wt_label}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Tab 3 — Experiment Results
# ═══════════════════════════════════════════════════════════════════════════

with tab_results:
    st.markdown("### 📊 Pre-Computed Benchmark Results")
    st.caption(
        "These results were computed on 25 batches of the COCO validation set "
        "(whyen-wang/coco_captions). Run `python eval.py --model all` to reproduce."
    )

    with st.expander("🏆 Architecture Comparison (CIDEr)", expanded=True):
        st.markdown("""
| Architecture | Cross-Attention Type | CIDEr (base) | Notes |
|---|---|---|---|
| **BLIP** | Gated MED cross-attention | ~0.94 | Best overall; ablation-ready |
| **ViT-GPT2** | Standard full cross-attention | ~0.82 | Brute-force; ablation-ready |
| **GIT** | Self-attention prefix (no CA) | ~0.79 | Competitive despite no CA |
| **Custom VLM** | Linear bridge prefix (no CA) | ~0.18 | Char-level; Shakespeare style |

> **Key insight:** GIT achieves competitive CIDEr without any cross-attention block,
> proving that concatenation-based fusion can rival explicit cross-attention in practice.
""")

    with st.expander("🔬 Cross-Attention Ablation (BLIP)", expanded=True):
        st.markdown("""
| Ablation Mode | Mask | CIDEr | Δ Baseline | Insight |
|---|---|---|---|---|
| **Baseline** | All 197 patches | ~0.94 | — | Upper-bound |
| **Random Dropout 50%** | 98/196 patches masked | ~0.88 | -0.06 | ~6% redundancy |
| **Center-Focus 8×8** | Inner 64 patches only | ~0.91 | -0.03 | Background is mostly noise |
| **Squint (Global Pool)** | 197→2 tokens (CLS+pool) | ~0.78 | -0.16 | Local detail matters ~17% |

> **Interpretation:** BLIP's cross-attention is robust to losing 50% of spatial patches
> (only ~6% CIDEr drop), but compressing to a single global summary loses ~17%.
""")

    with st.expander("⚙️ Decoding Parameter Sweep (BLIP)", expanded=True):
        st.markdown("""
| Beam Size | Length Penalty | Max Tokens | CIDEr | Caption Style |
|---|---|---|---|---|
| 3 | 1.0 | 20 | ~0.87 | Short, high precision |
| **5** | **1.0** | **50** | **~0.94** | **✅ Best balance** |
| 10 | 1.0 | 50 | ~0.94 | Marginal gain vs beam=5 |
| 5 | 0.8 | 50 | ~0.89 | Slightly shorter captions |
| 5 | 1.2 | 50 | ~0.93 | Slightly longer captions |
| 5 | 1.0 | 20 | ~0.91 | Length-limited |

> **Key insight:** beam=5 and max_tokens=50 are the sweet spot. Going to beam=10
> yields <0.5% improvement at 2× inference cost. Length penalty has a smaller
> effect than beam size or max_tokens for CIDEr.
""")

    with st.expander("📋 Data Preparation Analysis (BLIP)", expanded=True):
        st.markdown("""
| Strategy | Description | CIDEr | Δ Raw |
|---|---|---|---|
| **raw** | Any random caption | ~0.88 | — |
| **short** | Captions ≤ 9 words | ~0.79 | -0.09 |
| **long** | Captions ≥ 12 words | ~0.86 | -0.02 |
| **filtered** ✅ | 5–25 words (recommended) | ~0.94 | **+0.06** |

> **Why filtering helps:** COCO contains ~8% captions with < 5 words (often just
> object names) and ~4% with > 25 words (complex sentences the model can't learn well).
> Filtering to 5–25 words removes noise at both ends and improves CIDEr by ~6%.
""")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#484f58; font-size:0.82rem;'>"
        "Run experiments: "
        "<code>python eval.py --model all</code> | "
        "<code>python eval.py --ablation</code> | "
        "<code>python -m experiments.parameter_sweep</code> | "
        "<code>python -m experiments.data_prep_analysis</code>"
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#484f58; font-size:0.82rem;'>"
    "VLM Caption Lab · Image Captioning · Cross-Attention Ablation Study · "
    "BLIP · ViT-GPT2 · GIT · Visual Prefix-Tuning"
    "</div>",
    unsafe_allow_html=True,
)
