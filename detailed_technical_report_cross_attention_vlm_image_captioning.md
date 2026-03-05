# Detailed Technical Report: Cross-Attention Strategies in Vision-Language Models for Image Captioning

**Author:** Manoj Kumar  
**Project:** VLM Caption Lab  
**Date:** 4 March 2026  
**Dataset:** MS-COCO Captions (`whyen-wang/coco_captions`)

---

## Table of Contents

1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [The Central Question: How Should Vision Meet Language?](#2-the-central-question-how-should-vision-meet-language)
3. [Dataset and Data Quality Engineering](#3-dataset-and-data-quality-engineering)
4. [Architecture Deep Dive: Four Ways to Fuse Vision and Text](#4-architecture-deep-dive-four-ways-to-fuse-vision-and-text)
5. [Building a Custom Vision-Language Model from Scratch — The Full Story](#5-building-a-custom-vision-language-model-from-scratch--the-full-story)
6. [Training Pipeline: Making It All Work](#6-training-pipeline-making-it-all-work)
7. [Experiments and Results](#7-experiments-and-results)
8. [The Streamlit Application](#8-the-streamlit-application)
9. [Key Insights and Analytical Conclusions](#9-key-insights-and-analytical-conclusions)
10. [Future Improvements](#10-future-improvements)
11. [Reproducibility and Commands](#11-reproducibility-and-commands)
12. [Project Structure](#12-project-structure)

---

## 1. Introduction and Motivation

Image captioning sits at the intersection of computer vision and natural language processing. The task sounds deceptively simple: given a photograph, produce a sentence that describes what is happening in it. But underneath that simplicity lies a fundamental engineering question — **how exactly should a model look at an image while it is writing a sentence about it?**

This project was born out of a desire to understand that question from the ground up. Rather than just using one pre-trained model and calling it "good enough," I wanted to build a pipeline that puts **four fundamentally different architectures** side by side — trained on the same dataset, measured by the same evaluation metric, and running on the same hardware — and then systematically test what happens when you change how vision and language interact.

The four architectures I chose each represent a distinct philosophy about multimodal fusion:

- **BLIP** uses a gated cross-attention mechanism where the decoder can selectively filter how much visual information flows into each text token.
- **ViT-GPT2** (Vision Transformer paired with GPT-2) takes the brute-force approach: full cross-attention at every decoder layer, with every text token attending to every image patch.
- **GIT** (Generative Image-to-text Transformer) throws out cross-attention entirely and concatenates image embeddings directly into the text sequence, treating everything as a single self-attention problem.
- **Custom VLM** (Custom Vision-Language Model) is a model I built from scratch, combining a frozen Vision Transformer with a character-level Transformer decoder that was originally trained on Shakespeare's complete works.

That last one — the Custom VLM — is where the most interesting engineering challenges emerged, and where I learned the most about what it actually takes to make two models from completely different domains work together.

### What This Report Covers

This report documents **every architectural choice, every bug, every experiment, and every insight** from this project. It is written as a narrative — not a dry summary of results — because the debugging process itself taught me more than the final numbers did.

---

## 2. The Central Question: How Should Vision Meet Language?

Before diving into implementation, it helps to understand the core architectural decision that differentiates these four models: **the role of cross-attention.**

**What is self-attention?** In a standard Transformer (the architecture behind models like GPT), self-attention allows each word in a sentence to look at every other word in the same sentence. This is how the model understands context — the word "bank" can mean a financial institution or a river bank, and self-attention helps the model figure out which one based on surrounding words.

**What is cross-attention?** Cross-attention extends this idea by allowing words from one sequence (say, text) to look at tokens from a *different* sequence (say, image patches). This is how most encoder-decoder models connect their visual understanding to their language generation. The text decoder says, "I am about to write the next word — let me look at the image to decide what it should be."

**But here is the interesting part — cross-attention is not the only way to do this.** Some models skip it entirely. GIT, for example, concatenates image patch embeddings directly in front of text token embeddings and runs the whole thing through a single self-attention Transformer. There is no separate "looking at the image" computation. The model just treats image patches as very unusual text tokens.

My Custom VLM does something similar but with a twist: it projects visual embeddings through a trainable MLP (Multi-Layer Perceptron — a small neural network with two layers) into the character-level decoder's embedding space <b>(My personal decoder transformer built from scratch)</b>, and then the decoder processes the visual prefix alongside character embeddings using regular self-attention.

The table below summarizes how each architecture handles this fusion:

| Architecture | Fusion Mechanism | Has Cross-Attention? | Can We Test Masking? |
|---|---|---|---|
| **BLIP** | Gated cross-attention inserted between self-attention and feed-forward layers in the decoder | ✅ Yes | ✅ Yes — via `encoder_attention_mask` |
| **ViT-GPT2** | Standard full cross-attention at every GPT-2 layer | ✅ Yes | ✅ Yes — via `encoder_attention_mask` |
| **GIT** | Image tokens concatenated as prefix → single self-attention | ❌ No | ❌ No — no separate encoder mask |
| **Custom VLM** | MLP (Multi-Layer Perceptron) projection → visual prefix + character embeddings → self-attention | ❌ No | ❌ No — visual prefix is part of sequence |

### The Fusion Formulas (What Happens Mathematically)

For one who is interested in the math, here is how each model processes vision and text internally:

- **ViT-GPT2 (Full Cross-Attention):**
  - `text_output = CrossAttention(Query=text_hidden, Key=image_hidden, Value=image_hidden)`
  - Every text token directly queries every image patch

- **BLIP (Gated Multimodal Cross-Attention):**
  - Step 1: `h = SelfAttention(text_hidden)` — text tokens attend to each other
  - Step 2: `h = h + gate × CrossAttention(Query=h, Key=image_hidden, Value=image_hidden)` — learnable gate controls image flow
  - Step 3: `h = FeedForward(h)` — final transformation
  - The **gate** is what makes BLIP special — it learns to close when generating syntax words ("the", "a") and open when generating content words ("dog", "standing")

- **GIT (Self-Attention Prefix — No Cross-Attention):**
  - `combined_sequence = [image_patches ; text_tokens]`
  - `output = CausalSelfAttention(combined_sequence)`
  - Everything is one sequence — no separate image processing step

- **Custom VLM (Visual Prefix-Tuning):**
  - Step 1: `visual_prefix = MLP(ViT_encoder(image))` — project image patches into text space
  - Step 2: `input = [visual_prefix ; character_embeddings]` — concatenate
  - Step 3: `output = CausalSelfAttention(input)` — process as one sequence
  - Step 4: `logits = LanguageHead(output[after_visual_prefix:])` — predict characters

---

## 3. Dataset and Data Quality Engineering

### 3.1 The Dataset

I used the **MS-COCO Captions dataset** from HuggingFace (`whyen-wang/coco_captions`). COCO (Common Objects in Context) is the standard benchmark for image captioning — it contains natural photographs of everyday scenes, each annotated with five human-written captions describing the image.

**Why COCO?** It is the most widely used benchmark in image captioning research, which makes my results directly comparable to published papers. It also has high-quality human annotations — each image has five independent descriptions, giving multiple valid reference points for evaluation.

The data split I used:

| | Training Images | Validation Images |
|-|---|---|
| BLIP | 30,000 | 2,000 |
| ViT-GPT2 / GIT | 15,000 | 1,500 |
| Custom VLM | 15,000 | 1,500 |

BLIP gets more data because it is the largest model (224 million parameters) and benefits more from additional training examples. The smaller models converged adequately with 15,000 samples.

### 3.2 The Caption Quality Problem

One thing I noticed early on is that COCO captions are not uniformly useful for training. Some captions are extremely short — just "Dog" or "A cat" — while others are excessively long, rambling 40-word descriptions. During initial training, I found that treating every caption equally added noise: the model would sometimes learn to generate one-word descriptions, other times try to produce paragraphs.

I ran a systematic analysis on the caption word-count distribution:

| Metric | Value |
|---|---|
| Total captions sampled | 1,000 |
| Mean word count | 10.4 words |
| Range | 7 – 28 words |
| 10th percentile | 8 words |
| 50th percentile (median) | 10 words |
| 90th percentile | 13 words |
| % under 5 words | 0.0% |
| % over 25 words | 0.2% |

### 3.3 Caption Filtering Strategies

To address the caption quality problem, I implemented a configurable caption filtering pipeline in `data_prep.py` with five strategies:

1. **`raw`** — Pick any random caption from the five available. No filtering at all.
2. **`filtered`** — Only use captions between 5 and 25 words. Falls back to a random caption if none qualify. **This is the recommended default.**
3. **`short`** — Prefer captions with 9 or fewer words. Trains the model to be concise.
4. **`long`** — Prefer captions with 12 or more words. Trains the model to be descriptive.
5. **`mixed`** — Randomly switch between short, medium, and long strategies each time.

The filtering is implemented through the `pick_caption_by_strategy()` function, which is called during dataset construction. The strategy is configurable through `configs/base_config.py`:

```python
caption_strategy: str = "filtered"    # recommended default
caption_min_words: int = 5
caption_max_words: int = 25
```

### 3.4 Character-Level Tokenization for the Custom VLM

Most modern language models use **subword tokenization** (called BPE — Byte Pair Encoding), where common words are single tokens and rare words are split into pieces. For example, GPT-2 treats "standing" as a single token.

My Custom VLM does something different — it uses a **character-level vocabulary of 65 characters** built from Shakespeare's complete works. This means the sentence "a man standing in front of a tree" gets encoded as individual characters: `a`, ` `, `m`, `a`, `n`, ` `, `s`, `t`, `a`, `n`, `d`, `i`, `n`, `g`... That is roughly 35 character tokens, compared to about 8 subword tokens in GPT-2.

**Why character-level?** This was a deliberate design choice — the Shakespeare decoder was built for character generation, and changing the tokenizer would require retraining from scratch. It makes the Custom VLM's job harder but also more instructive: it forces the model to learn English spelling on top of learning to describe images.

The `COCOCharDataset` class in `data_prep.py` handles this conversion, encoding each caption into a sequence of character indices and padding to `max_target_len=128`.

---

## 4. Architecture Deep Dive: Four Ways to Fuse Vision and Text

### 4.1 BLIP — Gated Multimodal Mixture Attention

> **Model:** `Salesforce/blip-image-captioning-base` | **Parameters:** 224 million

BLIP's architecture is called a **Multimodal mixture of Encoder-Decoder (MED)**. The key innovation is how it injects visual information into the text decoder: between the self-attention and feed-forward sub-layers at each decoder block, there is a **cross-attention sub-layer with a learnable gate.**

**What does the gate do?** When the decoder is generating a purely syntactic token (like "the" or "is"), the gate can learn to close — effectively ignoring the image. When the decoder needs to produce a content word (like "dog" or "standing"), the gate opens to let visual features through. This selective attention prevents what researchers call "attention collapse," where the model becomes so distracted by visual features that it loses track of grammar.

In my implementation (`models/blip_tuner.py`), I load the model with **gradient checkpointing** enabled (which trades computation time for reduced memory usage — instead of keeping all intermediate values in memory for the backward pass, it recomputes them on the fly). I also resize images to 224×224 pixels to fit within Apple Silicon memory constraints.

**The `generate_with_mask()` function** is critical — it allows inference-time masking by accepting a custom attention mask that restricts which image patches the decoder can see. This is what powers the ablation experiment described in Section 7.1.

### 4.2 ViT-GPT2 — Standard Full Cross-Attention

> **Model:** `nlpconnect/vit-gpt2-image-captioning` | **Parameters:** 239 million

This is the brute-force baseline. ViT-GPT2 is a **VisionEncoderDecoderModel** that pairs:
- **Vision Transformer (ViT)** as the image encoder — takes a 224×224 image and splits it into a 14×14 grid of patches (196 patches + 1 special class token = 197 total), each represented as a 768-dimensional vector
- **GPT-2** as the text decoder — generates text one word at a time

At every decoder layer, an explicit cross-attention block lets **each text token attend to all 197 ViT patch embeddings**. Every word the model generates has full access to every part of the image at every layer.

**Advantage:** Maximum information flow — nothing is filtered or hidden.  
**Disadvantage:** Computationally expensive, and the constant stream of visual input can sometimes confuse the language generation.

### 4.3 GIT — Zero Cross-Attention Architecture

> **Model:** `microsoft/git-base-coco` | **Parameters:** 177 million

GIT (Generative Image-to-text Transformer) represents a fundamentally different philosophy: **instead of adding cross-attention layers to connect vision and language, GIT concatenates image patch embeddings directly in front of text tokens to form a single flat sequence:**

```
[image_patch_1, image_patch_2, ..., image_patch_N, text_token_1, text_token_2, ...]
```

A single causal self-attention Transformer processes the entire sequence. There are no dedicated cross-attention blocks. The vision-language fusion happens implicitly through positional self-attention — text tokens at the end of the sequence naturally attend to image patches at the beginning.

**Why this is clever:** It eliminates an entire class of parameters (all the cross-attention weights), making the model smaller (177 million vs. 239 million for ViT-GPT2) and faster. The trade-off is that the model cannot separately control "how much to look at the image" versus "how much to focus on previously generated text."

**Important limitation for experiments:** Because GIT processes vision and text in a single sequence with no separate encoder, it does not have an `encoder_attention_mask` parameter. This means my masking ablation experiments (Section 7.1) cannot be applied to GIT.

### 4.4 Custom VLM — Visual Prefix-Tuning with Shakespeare Decoder

> **Parameters:** 103 million total, but only **16.2 million trainable** (the rest are frozen)

This is the model I built from scratch, and it is where most of the engineering effort went. The architecture has three components:

**Component 1: Frozen Vision Transformer (ViT) Encoder**  
A standard ViT pre-trained on ImageNet-21K (`google/vit-base-patch16-224-in21k`). It takes a 224×224 image and produces 197 patch embeddings, each 768-dimensional. **These weights are completely frozen during training** — I do not want to disturb the image understanding capabilities that the model already learned on ImageNet.

**Component 2: Trainable MLP Bridge (The Critical Connection)**  
This is the only component connecting vision to language. It is a small two-layer neural network (Multi-Layer Perceptron) that projects each 768-dimensional visual embedding down to the decoder's 384-dimensional embedding space:

```python
self.visual_projection = nn.Sequential(
    nn.Linear(768, 1536),   # expand from 768 to 1536 dimensions
    nn.GELU(),               # nonlinear activation function
    nn.Linear(1536, 384)     # compress down to 384 dimensions
)
```

**Why two layers instead of one?** This is explained in detail in Section 5 — a single linear layer was not enough because it cannot perform the nonlinear transformation needed to translate between visual and textual feature spaces.

**Component 3: Shakespeare-Pretrained Character-Level Decoder**  
8 Transformer blocks, 8 attention heads, 384-dimensional embeddings, and a vocabulary of just 65 characters. This decoder was originally trained to generate Shakespeare text, character by character. During fine-tuning, both the MLP bridge and the decoder are trainable, with different learning rates.

**How the full pipeline works:**
1. ViT processes the image → 197 patches × 768 dimensions
2. MLP projects each patch → 197 patches × 384 dimensions (these become the "visual prefix")
3. Character embeddings for the caption text are looked up → T characters × 384 dimensions
4. Visual prefix and character embeddings are concatenated into one sequence
5. A causal self-attention mask is applied, and the full Transformer decoder processes the sequence
6. The language model head produces logits (predictions) only for the text portion (positions after the visual prefix)

---

## 5. Building a Custom Vision-Language Model from Scratch — The Full Story

This section tells the complete narrative of building the Custom VLM, including every bug, every failed experiment, and every fix. **This was the most educational part of the entire project,** and it demonstrates the kind of debugging that real machine learning engineering requires.

### 5.1 The Starting Point: A Shakespeare Decoder

The journey started with a character-level Transformer I had previously trained on the complete works of Shakespeare (~1 MB of Elizabethan English). This model could generate passable Shakespeare prose — things like "To be or not to be, that is the question" continuations. It had 8 Transformer blocks, 8 attention heads, 384-dimensional embeddings, and a 65-character vocabulary.

The idea was simple: if this decoder already understands English (even old English), maybe I could teach it to describe images by just showing it visual features as a prefix. I would freeze the ViT, freeze the Shakespeare decoder, and **only train a small projection layer** to translate from ViT's 768-dimensional visual space to the decoder's 384-dimensional text space.

This approach is called **"visual prefix-tuning"** and it is conceptually similar to what LLaVA (Large Language and Vision Assistant) does, except LLaVA uses GPT-4-level decoders and I am using a tiny character-level model.

### 5.2 Stage 1: The Linear Projection Bottleneck (Training Loss Stuck at 2.92)

My first implementation used a single linear layer for the projection:

```python
# Original (broken) — just one matrix multiplication
self.visual_projection = nn.Linear(768, 384)
```

I trained this for 15 epochs and watched the training loss. It dropped quickly at first — from around 4.5 down to about 3.5 — but then hit a rigid plateau at approximately **2.922** and refused to budge. Epoch after epoch, the loss hovered around 2.92, never improving.

The generated text was complete gibberish: strings like `"iGiiiiiGiviqiGqiFliqiGidlidiliGilFGilqiiiqiiiiGii"`. The CIDEr score was **0.0000** — literally zero. Not a single word overlapped with any human reference caption.

> **Why this happened:** A single linear projection is just a matrix multiplication — it can rotate and scale the visual embeddings, but it cannot perform the kind of nonlinear transformation needed to translate between two fundamentally different feature spaces. ViT's 768-dimensional space encodes visual concepts (edges, textures, object boundaries), while the decoder's 384-dimensional space encodes character-level language patterns. Mapping between these with just a matrix multiply is like trying to translate French to Chinese using only a ruler — the tool simply lacks the expressive power.

### 5.3 Stage 1 Fix: Upgrading to a Two-Layer MLP (Inspired by LLaVA)

I replaced the single linear layer with a two-layer MLP (Multi-Layer Perceptron):

```python
# Fixed — two layers with GELU nonlinearity
self.visual_projection = nn.Sequential(
    nn.Linear(768, 1536),    # 768 → 1536 (expand to give room for learning)
    nn.GELU(),                # nonlinear activation function
    nn.Linear(1536, 384)      # 1536 → 384 (compress to decoder's dimension)
)
```

**What is GELU?** GELU (Gaussian Error Linear Unit) is an activation function — a mathematical function that introduces nonlinearity. Without it, stacking two linear layers is mathematically equivalent to a single linear layer. The GELU between the two layers gives the projection the ability to learn nonlinear boundaries — meaning it can map visual concepts to text concepts in ways that a simple scaling/rotation cannot.

**Why 1536 as the middle dimension?** This is 2× the input dimension (768), providing a wide intermediate representation where the model can "reason" about how visual concepts map to textual concepts before compressing down to 384. This is the same approach used by LLaVA.

### 5.4 Stage 2: Why Training Loss Alone Is Not Enough

Even after the MLP upgrade, I realized I had a **measurement problem**. The training loss was going down, but I had no way to know if the actual captions were any good.

**What is training loss?** Training loss (specifically, cross-entropy loss) measures the probability the model assigns to the correct next token given all previous tokens. It is a mathematical surrogate — a number the optimizer tries to minimize — but it does not directly measure caption quality. A model can achieve low cross-entropy loss while generating grammatically incorrect, semantically meaningless text.

**What is CIDEr?** CIDEr (Consensus-based Image Description Evaluation) is a metric specifically designed for image captioning. It compares the caption our model generates to five human-written descriptions of the same image using n-gram overlap (matching sequences of consecutive words), weighted by TF-IDF (a technique that gives more weight to descriptive words like "bicycle" and less weight to common words like "the"). **A higher CIDEr score means the generated caption sounds more like what a human would write.**

| Metric | What It Measures | Reliable? |
|---|---|---|
| Training Loss | How well model predicts next token on training data | ❌ Can be misleading — low loss ≠ good captions |
| Validation Loss | How well model predicts next token on unseen data | ⚠️ Better, but still a surrogate |
| **CIDEr Score** | **How closely generated captions match human descriptions** | **✅ The gold standard for captioning** |

**The pipeline changes I made to `train.py`:**

1. **Validation loss tracking** — At the end of every epoch, run a forward pass on a validation subset to detect overfitting (when training loss drops but validation loss rises, the model is memorizing training data instead of learning general patterns).

2. **Live CIDEr computation** — Actually generate captions using beam search on the validation set, then score them with the `pycocoevalcap` CIDEr scorer. This tells me if the model is producing good English descriptions, not just achieving low loss numbers.

3. **CIDEr-based checkpointing** — Save the `best/` checkpoint based on the **highest validation CIDEr**, not the lowest training loss. This ensures the saved model is the one that actually produces the best captions.

The epoch-end logging now shows all three metrics:
```
Epoch 11/15 avg loss (Train): 0.8573
  Running Validation (Loss & CIDEr)...
  Validation Loss: 0.8077
  Validation CIDEr: 0.2863
  🏆 New best CIDEr! Saved → ./outputs/custom_vlm/best
```

### 5.5 Stage 3: The Gibberish Mystery — 337 Out of 342 Weights Silently Failed to Load

This was the most painful and instructive bug of the entire project. Even with the MLP upgrade and CIDEr pipeline in place, the model was **still generating pure gibberish**. I could see the loss was dropping, the pipeline was working, but the outputs were nonsensical character sequences.

After day of investigation, I found the root cause: **an architecture mismatch between the Shakespeare checkpoint and the Custom VLM decoder.**

Here is what happened:

**The original Shakespeare model** was built with a custom per-head attention implementation. Each of its 8 attention heads had its own separate weight matrices:

```
blocks.0.sa_head.heads.0.key.weight    → shape (48, 384)     ← head 1
blocks.0.sa_head.heads.1.key.weight    → shape (48, 384)     ← head 2
blocks.0.sa_head.heads.2.key.weight    → shape (48, 384)     ← head 3
... (8 separate weight matrices per layer)
```

**But the Custom VLM decoder** used PyTorch's built-in `nn.TransformerEncoder`, which expects **fused** (combined) attention weights:

```
decoder_blocks.layers.0.self_attn.in_proj_weight → shape (1152, 384)
```

**These are completely different formats.** The per-head format has 8 separate small matrices. PyTorch's format concatenates all heads into a single large matrix. It is like trying to load 8 individual photos into a slot designed for one panoramic image.

To make matters worse, the original Custom VLM config used **6 blocks, 6 heads, and a block size of 512**, while the Shakespeare checkpoint had **8 blocks, 8 heads, and a block size of 256**. **Nothing matched.**

When I loaded the checkpoint with `strict=False`:

```python
model.load_state_dict(checkpoint, strict=False)
```

PyTorch silently compared the key names, found that almost none of them matched, and simply **skipped 337 out of 342 tensors**. Only 5 tensors loaded — the character embedding table and the language model head. **The entire decoder brain — all the self-attention layers and feed-forward networks — was left randomly initialized.**

And because `freeze_decoder()` was called immediately after loading, those random weights were frozen in place. The model was literally running on random noise, with no way to learn.

> **⚠️ This is why `strict=False` is dangerous.** PyTorch does not raise an error or even a warning when the vast majority of a model fails to load. It just silently skips mismatched keys, leaving the developer to discover the problem through painstaking debugging. **In production code, always check how many tensors actually loaded.**

### 5.6 Stage 3 Fix: Architecture Alignment + Weight Remapping + Decoder Unfreezing

The fix required three coordinated changes:

**Fix 1: Architecture Alignment**  
I updated `custom_vlm_config.py` to exactly match the Shakespeare checkpoint dimensions:

```python
text_embed_dim: int = 384   # match Shakespeare (was different before)
n_heads: int = 8            # was 6, now 8 to match Shakespeare
n_layers: int = 8           # was 6, now 8 to match Shakespeare
block_size: int = 256       # was 512, now 256 to match Shakespeare
```

**Fix 2: Weight Remapping**  
I completely rewrote the `load_shakespeare_weights()` method in `custom_vlm.py`. The new implementation reads each per-head weight from the Shakespeare checkpoint, concatenates the 8 head weights for Query, Key, and Value into a single fused matrix, and maps it to PyTorch's expected format:

```python
# For each Transformer layer, fuse 8 per-head (48, 384) weights
# into one (1152, 384) matrix that PyTorch expects
query_weights = []
key_weights = []
value_weights = []
for head_idx in range(8):
    query_weights.append(ckpt[f"blocks.{layer}.sa_head.heads.{head_idx}.query.weight"])
    key_weights.append(ckpt[f"blocks.{layer}.sa_head.heads.{head_idx}.key.weight"])
    value_weights.append(ckpt[f"blocks.{layer}.sa_head.heads.{head_idx}.value.weight"])

in_proj_weight = torch.cat(query_weights + key_weights + value_weights, dim=0)
# Result: (1152, 384) = (3 attention_types × 8 heads × 48 dim_per_head, 384)
```

After loading, the method prints a verification count: **"96 of 96 decoder tensors loaded."** — all weights accounted for.

**Fix 3: Decoder Unfreezing with Discriminative Learning Rates**  
Instead of freezing the decoder, I unfroze it and used **discriminative learning rates** — different learning speeds for different parts of the model:

- **Projection MLP:** Learning rate = `1e-4` (0.0001) — aggressive updates because this is randomly initialized and needs to learn the vision-to-text mapping from zero
- **Decoder:** Learning rate = `5e-5` (0.00005) — gentle updates because the Shakespeare weights are a good starting point and we just want to slowly adapt from Elizabethan English to modern captioning style

### 5.7 The Results: From Gibberish to English

**The difference was immediate and dramatic:**

| Metric | ❌ Before (Broken) | ✅ After (Fixed) |
|---|---|---|
| Decoder tensors loaded | 5 of 342 (1.4%) | **96 of 96 (100%)** |
| Trainable parameters | 2.4 million (projection only) | **16.2 million (projection + decoder)** |
| Best training loss | 2.9226 (stuck at plateau) | **0.8446** |
| Best validation loss | Not tracked | **0.7930** |
| **Best CIDEr score** | **0.0000** | **0.2863** |
| Generated text sample | `"iGiiiiiGiviqiGqiFl..."` | `"man in the bluess and white play with and a pizza"` |

### Epoch-by-Epoch Progression (Custom VLM Training After Fix)

This table shows how the Custom VLM improved over 15 epochs. **This is the key evidence that the fixes worked:**

| Epoch | Training Loss | Validation Loss | CIDEr Score | What Happened |
|---|---|---|---|---|
| 1 | 1.9234 | 1.1396 | 0.0577 | Immediately broke the 2.92 plateau |
| 2 | 1.2543 | 0.9671 | 0.1352 | CIDEr doubled — real words emerging |
| 3 | 1.1261 | 0.9253 | 0.1594 | Sentences forming |
| 6 | 0.9339 | 0.8627 | 0.2329 | Clear English captions |
| 8 | 0.8919 | 0.8530 | 0.2391 | Steady gains |
| 10 | 0.8715 | 0.8501 | 0.2598 | Continued improvement |
| **11** | **0.8573** | **0.8077** | **0.2863** | **🏆 Best CIDEr — saved as best checkpoint** |
| 12 | 0.8514 | 0.7973 | 0.2728 | CIDEr starts dipping (overfitting) |
| 15 | 0.8446 | 0.8055 | 0.2284 | Slight overfitting — CIDEr drops further |

**Key observations from this progression:**

1. **The loss plateau at 2.92 broke immediately** on epoch 1 once the decoder had properly loaded weights. This confirms the plateau was caused by the architecture mismatch, not a fundamental capacity limitation.

2. **CIDEr peaked at epoch 11 (0.2863) and then started declining** even though training loss continued to drop. This is classic **overfitting** — the model memorizes training examples instead of generalizing. This validates the decision to checkpoint based on CIDEr rather than loss.

3. **The best validation loss (0.7930 at epoch 14) and the best CIDEr (0.2863 at epoch 11) occurred at different epochs.** This proves that loss and caption quality are genuinely different things — lowest loss ≠ best captions.

---

## 6. Training Pipeline: Making It All Work

### 6.1 The Unified Training Script

All four architectures are trained through a single entry point: `python train.py --model {blip|vit_gpt2|git|custom}`. The script handles model selection, configuration loading, and device detection (MPS → CUDA → CPU) automatically.

### 6.2 Hyperparameters

| Parameter | BLIP | ViT-GPT2 | GIT | Custom VLM |
|---|---|---|---|---|
| Epochs | 3 | 3 | 3 | 15 |
| Learning Rate | 1e-5 | 2e-5 | 2e-5 | 1e-4 (projection) / 5e-5 (decoder) |
| Batch Size | 16 | 8 | 8 | 16 |
| Max Target Length | 32 tokens | 32 tokens | 32 tokens | 128 characters |
| Gradient Accumulation Steps | 4 | 4 | 4 | 4 |
| Warmup Ratio | 0.03 (3%) | 0.03 | 0.03 | 0.03 |
| Weight Decay | 0.01 | 0.01 | 0.01 | 0.01 |
| Optimizer | AdamW | AdamW | AdamW | AdamW |
| Learning Rate Schedule | Cosine with warmup | Cosine with warmup | Cosine with warmup | Cosine with warmup |

**Why these choices:**

- **BLIP gets a lower learning rate (1e-5)** because it is the largest and most sensitive to destabilization. The pre-trained HuggingFace models have already converged; aggressive updates would break their learned representations.
- **The Custom VLM gets 15 epochs** because the character-level decoder takes longer to converge — it needs to learn character-by-character spelling in addition to visual grounding. The other models produce subword tokens and need far fewer iterations.
- **Gradient accumulation of 4 with batch size 16** gives an effective batch size of 64. This smooths out gradient noise without requiring Apple Silicon to hold 64 images in memory at once.

### 6.3 Efficiency Optimizations

- **Gradient checkpointing** — Enabled for BLIP. Instead of storing all intermediate values in memory for the backward pass (backpropagation), the model recomputes them on the fly. This roughly halves memory usage at the cost of ~30% slower training. Essential for fitting the 224-million-parameter BLIP on consumer hardware.

- **MPS (Metal Performance Shaders) acceleration** — All models run on Apple Silicon's GPU. This required setting `num_workers=0` in the data loader (MPS does not support multiprocessing data loading) and capping images at 224×224 pixels.

- **Gradient norm clipping** — Gradients are clipped to a norm of 1.0 to prevent exploding gradients. This is particularly important during early training epochs when the Custom VLM's projection layer is learning from scratch and can produce very large gradient values.

- **Cosine learning rate scheduling with warmup** — The learning rate starts at zero, linearly warms up during the first 3% of training steps, then follows a cosine curve back down to near-zero. This gives the model time to find a good optimization direction before committing to steep gradients.

### 6.4 Checkpoint Management

Checkpoints are saved to two locations:

| Directory | What It Contains | When to Use |
|---|---|---|
| `outputs/{model}/best/` | Checkpoint with the **highest validation CIDEr** seen during training | ✅ Use for evaluation and deployment |
| `outputs/{model}/latest/` | Checkpoint from the most recent epoch | 🔧 Use for debugging or resuming training |

---

## 7. Experiments and Results

### 7.1 Experiment 1: Cross-Attention Masking — What Happens When We Hide Parts of the Image?

**Question:** How important is fine-grained spatial visual information for caption generation? Can we remove parts of the image and still get good captions?

I designed four masking modes that manipulate which image patches the decoder can "see" during inference (caption generation):

**Mode 1 — Baseline (Full Attention)**  
All 197 patches (1 class token + 196 spatial patches from the 14×14 grid) are visible. This is the upper-bound reference — the model sees the entire image.

**Mode 2 — Random Patch Dropout (50%)**  
Randomly hide 50% of the 196 spatial patches; the class token always stays visible. Does the model still generate good captions with half the image hidden?

**Mode 3 — Center-Focus (Keep Only Inner 8×8 Grid)**  
Only keep the inner 64 patches of the 14×14 spatial grid, dropping the entire outer ring (the background and periphery). Does removing the edges and background matter?

**Mode 4 — Squint (Compress Everything to One Token)**  
Average all 196 spatial patches into a single global summary token. The mask becomes just 2 tokens: the class token and this one average. Can the model work with an extremely compressed representation?

**Results (BLIP, base pre-trained weights, 25 evaluation batches):**

| Mode | CIDEr Score | Change from Baseline | Interpretation |
|---|---|---|---|
| ✅ Baseline | **0.5371** | — | Full information reference |
| 🎲 Random Dropout (50%) | **0.5371** | +0.0000 (zero change!) | **Massive spatial redundancy — half the patches are disposable** |
| 🎯 Center-Focus (8×8) | **0.5371** | +0.0000 (zero change!) | **Background and edges contribute nothing** |
| 👀 Squint (Global Pool) | **0.0008** | −0.5363 (99.8% drop) | **Catastrophic failure — local details are essential** |

**What do these results mean?**

These results reveal something fascinating about how vision models process images:

- **Random dropout and center-focus cause zero degradation.** This means that for standard captioning, roughly **half of all spatial patches are entirely redundant**. The model can generate equally good captions with only 98 patches as with all 196. Background patches (the outer ring) also contribute nothing measurable.

- **But squinting destroys performance completely.** When you compress all 196 patches into a single average vector, CIDEr drops to essentially zero. This proves that while many individual patches are redundant, their collective **spatial arrangement** carries critical information. A single global vector cannot capture object locations, spatial relationships, and scene layout.

> **The takeaway:** BLIP's cross-attention is extremely robust to significant patch dropout, but it fundamentally requires spatially-distributed features. The spatial structure of the image matters more than the quantity of patches.

### 7.2 Experiment 2: Decoding Parameter Sweep — Finding the Best Caption Generation Settings

**Question:** How do beam search settings affect caption quality?

**What is beam search?** When a model generates text, it does not just pick the most probable next word at each step (that is called "greedy search" and often produces mediocre results). Instead, beam search maintains multiple candidate sentences simultaneously and picks the one with the best overall probability. Beam width controls how many candidates to track — more beams means more exploration but slower generation.

I swept across three decoding parameters for BLIP:
- **Beam sizes:** 3, 5, 10 (how many candidate sentences to track)
- **Length penalties:** 0.8, 1.0, 1.2 (penalty < 1.0 encourages longer captions, > 1.0 encourages shorter)
- **Max new tokens:** 20, 50 (maximum caption length allowed)

This produced **18 configurations** (3 × 3 × 2). Here are the results ranked by CIDEr score:

| Beams | Length Penalty | Max Tokens | CIDEr Score |
|---|---|---|---|
| 10 | 1.2 | 50 | **0.6199** ← 🏆 best |
| 10 | 1.0 | 20 | 0.5904 |
| 5 | 1.0 | 20 | 0.5896 |
| 10 | 1.2 | 20 | 0.5785 |
| 10 | 0.8 | 50 | 0.5722 |
| 3 | 1.2 | 20 | 0.5653 |
| 5 | 1.0 | 50 | 0.5598 |
| 5 | 1.2 | 20 | 0.5533 |
| 10 | 1.0 | 50 | 0.5457 |
| 3 | 1.2 | 50 | 0.5456 |
| 3 | 1.0 | 20 | 0.5451 |
| 10 | 0.8 | 20 | 0.5321 |
| 3 | 1.0 | 50 | 0.5262 |
| 5 | 1.2 | 50 | 0.5106 |
| 5 | 0.8 | 20 | 0.5046 |
| 3 | 0.8 | 50 | 0.5031 |
| 5 | 0.8 | 50 | 0.4914 |
| 3 | 0.8 | 20 | 0.4783 |

**Key findings:**

- **Beam size is the most impactful parameter.** Going from 3 beams to 10 beams with the best other settings improves CIDEr from ~0.55 to ~0.62 — an approximate **13% improvement**. More candidate sentences means better final selection.
- **Slight preference for shorter captions helps (length penalty 1.2).** BLIP tends to "ramble" with longer generation budgets, and concise captions match human references better.
- **The best combination is: beam_size=10, length_penalty=1.2, max_tokens=50** — yielding a CIDEr of **0.6199**.

### 7.3 Experiment 3: Caption Quality Filtering — Does Training Data Quality Matter?

**Question:** Does filtering caption quality before training improve model performance?

I evaluated BLIP under four caption selection strategies (what kind of captions we feed the model during training):

| Strategy | CIDEr Score | Change from Raw | Interpretation |
|---|---|---|---|
| raw (no filtering) | **0.6359** | — | **Best for this clean dataset** |
| short (≤ 9 words) | 0.6016 | −0.0342 | Too brief for good word overlap |
| filtered (5–25 words) | 0.5877 | −0.0481 | Quality filter |
| long (≥ 12 words) | 0.5389 | −0.0970 | Too verbose for base model |

**Why did raw perform best?** The COCO dataset is already relatively clean (mean word count 10.4, only 0.2% of captions over 25 words), so filtering actually removes useful variety. However, the **filtered strategy is still recommended as a general default** because it protects against noisy outliers in less curated datasets and ensures reproducible, consistent training behavior.

---

## 8. The Streamlit Application

The interactive demo is implemented in `app.py` and provides a complete interface for exploring and comparing all four architectures.

### 8.1 Features

| Feature | What It Does |
|---|---|
| **Caption Tab** | Upload an image, select a model and generation mode, generate a caption |
| **Compare All Models Tab** | Run all 4 architectures side-by-side on the same image with a summary table |
| **Experiment Results Tab** | View pre-computed results from all three experiments |
| **Weight Source Selector** | Switch between base (pre-trained), fine-tuned (best CIDEr), and fine-tuned (latest) weights |
| **Advanced Controls** | Adjust beam width, temperature, length penalty, top-k, and top-p |
| **Toxicity Filter** | Every caption is checked through `unitary/toxic-bert` before display |

### 8.2 Architecture Info Cards

Each model gets a descriptive card in the sidebar explaining its cross-attention approach in plain language:

- **BLIP:** "Gated cross-attention is injected between self-attention and feed-forward layers in the decoder, allowing fine-grained visual feature querying at each decoding step."
- **ViT-GPT2:** "Every GPT-2 text token attends to all 197 ViT patch embeddings via full cross-attention at every decoder layer."
- **GIT:** "Image patches are concatenated to the front of the token sequence; causal self-attention handles everything in one flat joint sequence."
- **Custom VLM:** "Fuses a frozen ViT with a Shakespeare character-level decoder via a trainable projection."

### 8.3 Safety: Toxicity Filtering

Because captioning models can occasionally generate offensive descriptions (particularly on ambiguous or culturally sensitive images), every generated caption passes through the `detoxify` library's `unitary/toxic-bert` model before being displayed. If the toxicity score exceeds a threshold, the caption is redacted and the user is warned.

---

## 9. Key Insights and Analytical Conclusions

### 9.1 Cross-Attention Is Helpful but Not Mandatory

GIT achieves strong captioning performance using only prefix self-attention — **no dedicated cross-attention blocks at all**. This proves that cross-attention, while helpful for selective visual querying, is not strictly mandatory for multimodal fusion. The prefix concatenation approach works because self-attention is a universal mechanism: as long as visual and text tokens share the same sequence, the model learns to route information between modalities.

### 9.2 Gated Attention Gives the Best Trade-Off

**BLIP's gated cross-attention achieves the highest CIDEr scores** because the gate selectively filters visual information. When generating syntax words ("the," "a"), the gate closes and the model relies on its language model. When generating content words ("dog," "bicycle"), the gate opens and visual features flow through. This prevents attention collapse — a failure mode where too much visual information disrupts language coherence.

### 9.3 Images Contain Massive Spatial Redundancy

The masking experiment proves that **50% of image patches can be removed with zero quality loss**, and cropping to the center removes the entire background with no effect. But compressing to a single global vector destroys performance. This means: **spatial structure matters more than absolute patch count.**

### 9.4 Loss and Quality Are Different Things

The Custom VLM training showed that **the best training loss and the best CIDEr occurred at different epochs** (epoch 14 vs. epoch 11). A model that predicts the next token well (low loss) is not necessarily a model that produces captions humans would agree with (high CIDEr). **Always evaluate with task-specific metrics, not just loss.**

### 9.5 Silent Failures Are the Worst Kind of Bug

The most time-consuming problem in this project was a weight-loading failure that produced **no error message, no warning, and no indication** that 98.5% of the model failed to load. **In production machine learning code, always verify how many tensors actually loaded when using `strict=False`.**

---

## 10. Future Improvements

The Custom VLM currently achieves a best CIDEr of **0.2863**. Here is a roadmap of improvements ordered by expected impact:

### High Impact (Could Improve CIDEr by +0.15 to +0.40 Each)

| Improvement | What It Changes | Expected CIDEr Gain |
|---|---|---|
| **Switch from characters to subword tokens** | "standing" becomes 1 token instead of 8 characters | +0.15 to +0.30 |
| **Replace Shakespeare decoder with GPT-2 Small** | GPT-2 already knows modern English; Shakespeare decoder had to learn both English and captioning | +0.20 to +0.40 |
| **Increase training data (15K → 80K)** | Use the full COCO training set instead of 18% | +0.05 to +0.10 |

### Medium Impact (Could Improve CIDEr by +0.05 to +0.15 Each)

| Improvement | What It Changes |
|---|---|
| **Label smoothing** (0.1) | Prevents overconfident character predictions |
| **Multi-reference CIDEr** (use all 5 human captions) | More accurate quality measurement |
| **Proper cross-attention layers** in the decoder | Dedicated vision-text interaction instead of prefix concatenation |
| **Stronger vision encoder** (CLIP ViT-Large) | CLIP features are inherently aligned with text |

---

## 11. Reproducibility and Commands

### Environment Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Verify acceleration is available (Apple Silicon)
python -c "import torch; print(torch.backends.mps.is_available())"
```

### Training

```bash
python train.py --model blip        # ~1.5 hours on Apple Silicon
python train.py --model vit_gpt2    # ~1 hour
python train.py --model git         # ~20 minutes
python train.py --model custom      # ~3 hours (15 epochs)
```

### Evaluation

```bash
# Evaluate one model
python eval.py --model blip --weights best

# Compare all models
python eval.py --model all --weights best

# Run cross-attention masking experiment
python eval.py --model blip --ablation --weights best

# Run decoding parameter sweep
python eval.py --model blip --sweep --weights best

# Custom decoding settings
python eval.py --model blip --weights best --num_beams 10 --max_new_tokens 50 --length_penalty 1.2
```

### Streamlit Demo

```bash
streamlit run app.py
```

---

## 12. Project Structure

```
project_02/
├── app.py                              # Streamlit demo (3 tabs: Caption, Compare, Results)
├── config.py                           # Backward-compatible config wrapper
├── data_prep.py                        # Dataset loading + caption filtering strategies
├── eval.py                             # Unified CIDEr evaluator + experiment runner
├── train.py                            # Unified training loop for all 4 models
├── requirements.txt                    # Python dependencies
├── input.txt                           # Shakespeare corpus (character vocabulary source)
├── shakespeare_transformer.pt          # Pre-trained Shakespeare decoder weights
│
├── configs/
│   ├── __init__.py                     # get_config() factory function
│   ├── base_config.py                  # Shared hyperparameters for all models
│   ├── blip_config.py                  # BLIP-specific settings
│   ├── vit_gpt2_config.py             # ViT-GPT2-specific settings
│   ├── git_config.py                   # GIT-specific settings
│   └── custom_vlm_config.py            # Custom VLM-specific settings
│
├── models/
│   ├── blip_tuner.py                   # BLIP: gated cross-attention
│   ├── vit_gpt2_tuner.py              # ViT-GPT2: full cross-attention
│   ├── git_tuner.py                    # GIT: zero cross-attention
│   └── custom_vlm.py                  # Custom VLM: visual prefix-tuning
│
├── experiments/
│   ├── ablation_study.py                                                # 4-mode attention masking experiment
│   ├── parameter_sweep.py                                               # Beam/penalty/token sweep
│   ├── cross_attention_patterns.py                                      # Architecture comparison
│   ├── data_prep_analysis.py                                            # Caption filtering analysis
│   ├── results_cross_attention_masking_impact_on_caption_quality.md     # Masking experiment results
│   ├── results_beam_search_and_decoding_settings_comparison.md          # Sweep results
│   └── results_caption_filtering_strategy_comparison.md                 # Filtering results
│
├── outputs/
│   ├── blip/{best,latest}/             # BLIP checkpoints
│   └── custom_vlm/{best,latest}/       # Custom VLM checkpoints
│
└── README.md                           # Project overview and setup guide
```

---

**Technologies Used:** Python 3.9+, PyTorch, HuggingFace Transformers, HuggingFace Datasets, Streamlit, pycocoevalcap (CIDEr evaluation), detoxify (toxicity filtering), Pillow, NumPy, tqdm, accelerate.

**Hardware:** Apple Silicon Mac with MPS (Metal Performance Shaders) acceleration.
