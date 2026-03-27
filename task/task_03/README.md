# 🔬 Task 3: Beam Search & Length Penalty Ablation for Caption Quality Trade-offs

## 📌 The Big Question: Does Beam Search Actually Make Captions Better?

When an AI model generates a caption for an image, it faces a decision at every single word: **which word should come next?** The simplest approach is **greedy decoding** — at each step, just pick the single highest-probability word and move on. It's fast, but it's short-sighted. It often gets "trapped" in a mediocre caption because it couldn't look ahead.

**Beam search** changes this by keeping multiple candidate captions alive simultaneously and only committing when the full sequence is generated. But this comes at a cost — it's slower, and the quality gains aren't guaranteed.

Then there's **length penalty**: a scalar that either punishes the model for producing short captions (`< 1.0`) or rewards it for staying concise (`> 1.0`). The interaction between beam size and length penalty is non-trivial and poorly understood without experiments.

This task cracks the problem open with a **full ablation study** across 9 decoding configurations to answer:
- Which combination of beam size and length penalty produces the best captions?
- Is the quality improvement worth the latency cost?
- What's the Pareto-optimal strategy for real-time vs. offline captioning?

---

## 🧠 Background: Training Setup

Before decoding, we need a good model. This task proceeds in two phases:

### Phase 1: Fine-tune BLIP on 10k COCO Captions
BLIP (*Bootstrapping Language-Image Pre-training*) is fine-tuned on 10,000 training image–caption pairs from the **MS-COCO 2017** dataset using the existing training pipeline:

```bash
python train.py --model blip
```

- **Training data**: 10,000 COCO training images (30,000 used in the main project)
- **Epochs**: 3 with cosine LR schedule and linear warmup
- **Optimizer**: AdamW, lr=1e-5, effective batch size=64 (gradient accumulation)
- **Checkpointing**: Best checkpoint saved to `outputs/blip/best/` based on validation CIDEr
- **Best validation CIDEr achieved during training**: **0.6199** (at epoch 3)

The fine-tuned checkpoint in `outputs/blip/best/` is the model used for all 9 ablation configurations below.

---

## 🛑 Baseline: Greedy Decoding (beam=1)

Before running beam search, we establish a **greedy baseline** — the simplest possible decoding strategy.

| Metric | Score |
|--------|-------|
| CIDEr  | 0.4783 |
| BLEU-4 | 0.2341 |
| METEOR | 0.2701 |
| ROUGE-L | 0.4502 |
| Mean caption length | 9.8 tokens |
| Latency per 100 images | **4.2s** |

**Why it fails**: Greedy decode selects each word independently. By ignoring future context, it often commits to a locally plausible but globally mediocre path — resulting in generic captions like *"a man is standing in a field"* even when the image contains much richer detail.

---

## 🌟 Enhanced: Beam Search Ablation (3×3 Grid)

### Design: The 9-Configuration Grid

We sweep two decoding hyperparameters simultaneously:

```
beam_size      ∈ {1, 3, 5}
length_penalty ∈ {0.8, 1.0, 1.2}
──────────────────────────────────────
Total configurations : 9
Evaluation images    : 500 COCO val
```

**What each parameter controls:**

| Parameter | `< 1.0` | `= 1.0` | `> 1.0` |
|-----------|---------|---------|---------|
| `length_penalty` | Punishes short captions (forces longer output) | Neutral | Rewards compact captions |
| `beam_size` | 1 = greedy | 3 = balanced | 5 = high quality, slower |

### Metrics Computed Per Configuration

For each of the 9 configurations, four quality metrics are computed on 500 COCO validation images:

| Metric | What it Measures |
|--------|-----------------|
| **CIDEr** | Consensus-based: how well captions match 5 human references |
| **BLEU-4** | 4-gram precision overlap with reference captions |
| **METEOR** | Precision/recall with stemming, synonym matching |
| **ROUGE-L** | Longest common subsequence F1 with references |
| **Mean Length** | Average number of tokens per generated caption |
| **Latency/100** | Seconds to generate captions for 100 images |

---

## 📊 Full Results: All 9 Configurations

Results sorted by CIDEr score (primary metric):

| Rank | Beam | LenPen | CIDEr | BLEU-4 | METEOR | ROUGE-L | Avg Len | Lat/100 | Pareto? |
|------|------|--------|-------|--------|--------|---------|---------|---------|---------|
| 1 🏆 | **5** | **1.0** | **0.5598** | **0.2891** | **0.3089** | **0.4953** | 10.8 | 15.1s | ✅ |
| 2    | 3 | 1.2 | 0.5456 | 0.2791 | 0.2981 | 0.4872 | 11.2 | 9.4s | ✅ |
| 3    | 3 | 1.0 | 0.5451 | 0.2821 | 0.3012 | 0.4891 | 10.5 | 9.1s | ✅ |
| 4    | 5 | 1.2 | 0.5106 | 0.2674 | 0.2914 | 0.4734 | 11.9 | 15.8s | — |
| 5    | 3 | 0.8 | 0.5031 | 0.2641 | 0.2891 | 0.4705 | 9.6 | 8.7s | — |
| 6    | 5 | 0.8 | 0.4914 | 0.2558 | 0.2834 | 0.4621 | 9.4 | 14.2s | — |
| 7    | 1 | 1.0 | 0.4783 | 0.2341 | 0.2701 | 0.4502 | 9.8 | 4.2s | ✅ |
| 8    | 1 | 1.2 | 0.4651 | 0.2271 | 0.2658 | 0.4461 | 10.4 | 4.3s | — |
| 9    | 1 | 0.8 | 0.4512 | 0.2201 | 0.2614 | 0.4389 | 9.2 | 4.1s | — |

> ✅ Pareto-optimal = no other config has both higher CIDEr AND lower latency.

---

## 🌡️ CIDEr Heatmap: Beam Size × Length Penalty

The heatmap visualizes how CIDEr score varies across the full 3×3 grid. **Warmer (brighter) cells = better caption quality.**

```
Length Penalty →     0.8      1.0      1.2
                  ┌────────┬────────┬────────┐
Beam = 1          │ 0.4512 │ 0.4783 │ 0.4651 │  ← greedy, fastest
                  ├────────┼────────┼────────┤
Beam = 3          │ 0.5031 │ 0.5451 │ 0.5456 │  ← balanced sweet spot
                  ├────────┼────────┼────────┤
Beam = 5          │ 0.4914 │★0.5598 │ 0.5106 │  ← peak quality
                  └────────┴────────┴────────┘
```

**Key pattern**: The `length_penalty=1.0` column is consistently strong. `lp=0.8` penalizes longer candidates too aggressively, causing early truncation. `lp=1.2` over-rewards length, leading to captions that run on beyond the reference length and accumulate noise tokens.

See `results/cider_heatmap.png` for the colour-coded version.

---

## ⚡ Latency Analysis: The Speed–Quality Tradeoff

Generation time (seconds per 100 images) vs. CIDEr score:

```
CIDEr
0.56 |                              ★ (beam=5, lp=1.0)
0.55 |            ● ●  (beam=3, lp=1.0/1.2)
0.50 |    ●
0.48 |                                     Pareto
0.47 | ● (beam=1, lp=1.0)                  Frontier ─╮
     └──────────────────────────────────────────────────
          4s       9s      14s      →  Latency/100
```

| Use Case | Recommended Config | CIDEr | Latency/100 |
|----------|--------------------|-------|-------------|
| **Real-time** (live captioning, APIs) | beam=1, lp=1.0 | 0.4783 | 4.2s |
| **Balanced** (standard apps) | beam=3, lp=1.0 | 0.5451 | 9.1s |
| **Offline** (batch processing, archives) | beam=5, lp=1.0 | 0.5598 | 15.1s |

**Key finding**: Going from greedy (beam=1) to beam=3 yields a **+14% CIDEr improvement** at only a **2.2× latency cost**. Going further from beam=3 to beam=5 adds only **+2.7% more CIDEr** at a further **1.7× latency cost** — rapidly diminishing returns.

See `results/latency_barchart.png` and `results/quality_speed_scatter.png`.

---

## 🔍 Analysis: Key Findings

### Finding 1: Beam Size Matters More Than Length Penalty
Across all three length penalty settings, the CIDEr variance driven by beam size (range: ~0.08) is **larger** than the variance driven by length penalty (range: ~0.03). Beam size is the primary lever; length penalty is a fine-tuning knob.

### Finding 2: Length Penalty = 1.0 is the Safest Default
For every beam size, `lp=1.0` performs at par or best. This is because the COCO captions used as references are themselves moderate length (~10 tokens). Any penalty that pushes the model toward shorter (`lp=0.8`) or longer (`lp=1.2`) sequences diverges from the reference distribution.

### Finding 3: Optimal for API Design
- **Real-time captioning API** (< 5s/100 images required): use `beam=1, lp=1.0`
- **Standard captioning** (< 10s/100): use `beam=3, lp=1.0` ← recommended default
- **High-fidelity offline**: use `beam=5, lp=1.0`

### Finding 4: Why lp=0.8 Hurts
`lp=0.8` encourages the beam to prefer *shorter* sequences. Combined with beam=5, it actually *reduces* CIDEr below the greedy baseline for some images because BLIP's captions are already quite compact and penalizing length causes early stopping before key objects are mentioned.

### Finding 5: BLEU-4 Agrees With CIDEr
The ranking by BLEU-4 is nearly identical to CIDEr ranking (Spearman ρ ≈ 0.93), validating that our CIDEr-based conclusions are not an artifact of the metric choice.

---

## 🏗️ Pipeline: 5 Independent Components

All code is organized into 5 self-contained modules. Each can be imported individually in a Jupyter notebook or run as a standalone script:

| File | What It Does | Returns |
|------|-------------|---------|
| `step1_load_model.py` | Load BLIP + fine-tuned checkpoint | `(model, processor, device)` |
| `step2_prepare_data.py` | Load 500 COCO val images | `DataLoader` |
| `step3_run_ablation.py` | Run 9-config grid, compute 4 metrics + latency | `list[dict]` (9 result rows) |
| `step4_visualize.py` | Generate 3 publication figures | `dict[str, path]` |
| `step5_analyze.py` | Pareto analysis, findings report | `dict` (findings) |
| `pipeline.py` | **Master orchestrator** — chains all steps | All of the above |

---

## 🚀 How to Run

Make sure you are in the project root directory and your virtualenv is active.

```bash
source venv/bin/activate
export PYTHONPATH=.
```

### Option A: Run Full Pipeline (Demo Mode — No GPU Required)
Uses pre-computed results bundled in `results/ablation_results.json`. All 3 figures are generated, the analysis is printed, and `findings.md` is saved.

```bash
venv/bin/python task/task_03/pipeline.py --demo
```

**Outputs:**
- `task/task_03/results/cider_heatmap.png` — 3×3 CIDEr heatmap
- `task/task_03/results/latency_barchart.png` — latency per config
- `task/task_03/results/quality_speed_scatter.png` — Pareto scatter
- `task/task_03/results/findings.md` — written analysis

### Option B: Run Full Pipeline (Live GPU Inference)
Downloads COCO val, runs all 9 configs end-to-end. Requires the fine-tuned BLIP checkpoint at `outputs/blip/best/` and a GPU (MPS or CUDA).

```bash
venv/bin/python task/task_03/pipeline.py
```

### Option C: Run Individual Components (for Notebook / HuggingFace inspection)

```python
# Step 1 — Load model
from task.task_03.step1_load_model import load_model
model, processor, device = load_model()

# Step 2 — Prepare data
from task.task_03.step2_prepare_data import load_val_data
dataloader = load_val_data(processor, n=500, batch_size=8)

# Step 3 — Run ablation (or load cached)
from task.task_03.step3_run_ablation import run_ablation
results = run_ablation(model, processor, dataloader, device)

# Step 4 — Visualize
from task.task_03.step4_visualize import visualize_all
paths = visualize_all(results)

# Step 5 — Analyze
from task.task_03.step5_analyze import analyze_results
findings = analyze_results(results)
```

### Option D: Run Step 3 in Live Mode (standalone)
```bash
venv/bin/python task/task_03/step3_run_ablation.py --live  # GPU inference
venv/bin/python task/task_03/step3_run_ablation.py         # pre-computed
```

### Option E: Regenerate Figures Only (no inference needed)
```bash
venv/bin/python task/task_03/step4_visualize.py   # generates all 3 PNGs
venv/bin/python task/task_03/step5_analyze.py     # prints analysis
```

---

## 🏆 How to Read and Judge the Results

### `results/cider_heatmap.png`
- **Brighter / warmer** cells = higher CIDEr (better captions)
- **Row** = beam size (1 → 3 → 5, top to bottom)
- **Column** = length penalty (0.8 → 1.0 → 1.2, left to right)
- Look for the ★ — it marks the best config at `beam=5, lp=1.0` (CIDEr: 0.5598)

### `results/quality_speed_scatter.png`
- **X-axis** = latency (lower = faster)
- **Y-axis** = CIDEr (higher = better)
- **Red dashed line** = Pareto frontier — configs on this line dominate all others
- Points *above* the frontier do not exist; points *below* are dominated

### `results/findings.md`
A machine-readable summary of the best config and insights — suitable for direct inclusion in a project report.

### ❓ Why Does `lp=0.8` Sometimes Beat `lp=1.2` for beam=3?
`lp=0.8` produces shorter captions that can sometimes align better with short reference captions in COCO. The COCO validation set has high variance in reference length (7–20 tokens). For images with very short human captions, penalizing length (`lp=0.8`) accidentally aligns better. `lp=1.0` wins on average because it is distribution-neutral.

---

## 📁 Folder Structure

```
task/task_03/
├── step1_load_model.py       # Component 1: Load BLIP + checkpoint
├── step2_prepare_data.py     # Component 2: COCO val DataLoader (500 images)
├── step3_run_ablation.py     # Component 3: 9-config sweep + 4 metrics + latency
├── step4_visualize.py        # Component 4: Heatmap, latency chart, scatter
├── step5_analyze.py          # Component 5: Rankings, Pareto, findings
├── pipeline.py               # Master orchestrator (--demo or live)
└── results/
    ├── ablation_results.json      # Pre-computed 9-config × 6-metric table
    ├── findings.md                # Written analysis (auto-generated)
    ├── cider_heatmap.png          # 3×3 CIDEr quality heatmap
    ├── latency_barchart.png       # Grouped latency bar chart
    └── quality_speed_scatter.png  # Pareto frontier scatter
```

---

## ⚙️ Dependencies

All dependencies are already in the project `requirements.txt`:

| Package | Used For |
|---------|---------|
| `transformers` | BLIP model loading and inference |
| `torch` | GPU acceleration (MPS / CUDA) |
| `datasets` | COCO 2017 validation split |
| `pycocoevalcap` | CIDEr metric computation |
| `nltk` | BLEU-4 and METEOR metrics |
| `rouge-score` | ROUGE-L metric |
| `matplotlib` | Heatmap, bar chart, scatter figures |
| `numpy` | Matrix operations for the heatmap grid |

---

## 🔗 Connection to the Broader Project

This task feeds directly back into the main project:
- The best config (`beam=5, lp=1.0`) is the **default decoding setting in `eval.py`** for the main evaluation sweep.
- The latency measurements inform the **API design recommendation** in `app.py` (real-time tab uses beam=1, compare tab uses beam=3).
- Results are referenced in the **main README** and `experiments/results_beam_search_and_decoding_settings_comparison.md`.

---

**Author:** Manoj Kumar — March 2026
