# 🔬 Task 4: Caption Diversity Analysis & Concept Activation Vectors for Style Steering

## 📌 The Big Question: Can We Steer a VLM to Write Longer or Shorter Captions—Without Retraining?

When a vision-language model generates a caption for an image, it doesn't just pick words at random—it navigates a high-dimensional **representation space** where different directions correspond to different caption properties. This task asks two deep questions:

1. **How diverse are the captions a model generates for the same image?** When we sample 5 captions using nucleus sampling (p=0.9), do we get genuinely different descriptions or minor paraphrases of the same sentence?

2. **Can we control caption style by directly manipulating the model's hidden states?** We extract "steering directions" from mean hidden states of short vs. detailed captions, then inject `h_steered = h + λ × direction` into the decoder at generation time—no gradient update, no retraining.

---

## 🧠 Part 1 — Caption Diversity Analysis

### What "Diversity" Means Here

For each image, we generate **5 captions** using **nucleus sampling** (`top_p=0.9`, `num_beams=1`). Nucleus sampling selects the next token from the smallest vocabulary subset whose cumulative probability exceeds `p=0.9`. This introduces stochasticity—but how much?

We quantify diversity with a single score:

```
diversity_score = unique_ngrams / total_ngrams
```

where ngrams = all **unigrams + bigrams** across the 5 captions for that image.

| Score Range | Category   | Meaning                                                   |
|-------------|------------|-----------------------------------------------------------|
| > 0.75      | 🌈 Diverse   | Captions use substantially different vocabulary each time |
| 0.40–0.75   | Medium     | Partial variation — similar structure, different words    |
| < 0.40      | 🔄 Repetitive| Almost identical captions — model is not exploring       |

### Results

| Metric                   | Value  |
|--------------------------|--------|
| Total images analysed    | 200    |
| Mean diversity score     | 0.5847 |
| Diverse (>0.75)          | 37 images (18.5%) |
| Medium (0.40–0.75)       | 118 images (59.0%) |
| Repetitive (<0.40)       | 45 images (22.5%) |

### Which Images Are Hard to Diversify?

**Repetitive**: Simple, prototypical scenes — a solitary animal on a plain surface, a single person in a common pose, a single food item on a plate. The model has high confidence and nucleus sampling collapses to the same phrase cluster.

**Diverse**: Complex multi-object scenes — busy city streets, sporting events, family gatherings. The model must choose which aspect to describe, leading to genuinely different captions.

> **Implication**: Caption diversity is an intrinsic property of image complexity. Extremely confident model predictions undermine the diversity even at high top-p values.

---

## 🧭 Part 2 — Concept Steering Vectors (CAV-style)

### The Idea: Representation Engineering for Caption Style

Language models store information about caption *style* in their hidden state geometry. Short captions cluster in one region; detailed captions in another. The **mean difference** between these regions is a **steering direction** — a vector that, when added to hidden states during decoding, nudges the model to generate more detailed (or shorter) text.

This is inspired by Concept Activation Vectors (CAVs) from interpretability research, adapted here for generative steering.

### Step 1 — Partitioning Captions by Style

From COCO validation captions (500 samples):

| Style     | Word Count | Example                                                    |
|-----------|------------|-------------------------------------------------------------|
| **Short**    | ≤ 8 words  | *"A dog on a couch"*                                        |
| **Medium**   | 9–14 words | *"A brown dog is resting on a dark leather couch"*          |
| **Detailed** | ≥ 15 words | *"A large brown Labrador is lying comfortably on a black leather sofa next to a cushion"* |

### Step 2 — Extracting Mean Hidden States

For each style group, we:
1. Tokenize all captions
2. Pass them through BLIP's **text encoder** (BERT-based)
3. **Mean-pool** the encoder output across all token positions
4. Average across all captions in the group → `μ_style ∈ ℝ^{768}`

### Step 3 — Computing Steering Directions

```python
d_short2detail = normalize(μ_detailed − μ_short)
d_short2medium = normalize(μ_medium   − μ_short)
```

Both vectors are L2-normalised so that λ has a consistent magnitude interpretation regardless of the dataset size used to compute the means.

### Step 4 — Applying the Steering Vector

At generation time, we register a **PyTorch forward hook** on every attention sub-layer inside BLIP's text decoder. The hook modifies the hidden state output before it flows to the next layer:

```python
h_steered = h + λ × d_short2detail
```

This is injected at **every decoder layer at every time step**, creating a persistent bias throughout the decoding process. No gradients are computed; the model weights are not changed.

---

## 📊 Steering Results — λ Sweep

We sweep λ from −1.0 (push toward shorter style) to +2.0 (push toward detailed style):

| λ   | Mean Length | Unique Words | Style Score |
|-----|-------------|--------------|-------------|
| −1.0  | 6.8 words   | 6.1          | 0.453       |
| −0.5  | 8.2 words   | 7.3          | 0.548       |
| **0.0** | **10.1 words** | **8.9**  | **0.673** ← baseline |
| +0.5  | 11.8 words  | 10.2         | 0.787       |
| +1.0  | 13.5 words  | 11.4         | 0.900       |
| +1.5  | 15.2 words  | 12.1         | 0.932       |
| +2.0  | 16.7 words  | 12.8         | 0.956       |

**Key finding**: λ=+2.0 adds +6.6 words (+65%) over baseline. λ=−1.0 shortens captions by −3.3 words (−33%). The effect is **monotonically increasing in λ**, confirming that the steering direction captures a real style axis.

---

## 🔍 Key Findings

### Finding 1: Diversity Is Bimodal

The distribution of diversity scores is not uniform — it is bimodal with peaks near 0.30 (repetitive) and 0.70 (diverse). Most images fall in the medium range, but simple and complex images cluster away from the center.

### Finding 2: Repetitive Images Are Visually Overconfident

When the model's visual encoder produces a very "clean" signal (e.g., a single dominant object on a plain background), the decoder probability distribution is sharply peaked. Even with p=0.9 nucleus sampling, the effective vocabulary at each step is very small, producing nearly identical captions.

### Finding 3: Steering Vectors Capture Real Style Axes

The steering effect is not noise — the mean caption length increases monotonically with λ across 7 different values and 20 images per λ. This replicates the core CAV hypothesis: mean representation differences encode interpretable semantic attributes.

### Finding 4: Optimal λ for Practical Use

- λ ∈ [0.5, 1.0]: Adds 2–4 words, keeps captions fluent and on-topic
- λ > 1.5: Captions start becoming verbose and diverge from the COCO reference distribution (CIDER would drop)
- λ < 0: Produces very terse captions, useful for summarization-style applications

### Finding 5: No Retraining Needed

The full steering effect requires zero gradient computation. The model weights are unchanged. The only requirement is access to a representative set of style-labelled captions to compute the mean vectors — making this technique immediately applicable to any BLIP-based deployment.

---

## 🏗️ Pipeline: 7 Independent Components

| File | What It Does | Returns |
|------|-------------|---------|
| `step1_load_model.py` | Load BLIP + fine-tuned checkpoint | `(model, processor, device)` |
| `step2_prepare_data.py` | COCO val DataLoader + style caption sets | `DataLoader`, `dict[style→list[str]]` |
| `step3_diversity_analysis.py` | 5 captions/image (nucleus p=0.9), diversity scores | `list[dict]` |
| `step4_steering_vectors.py` | Extract μ per style, compute d_short2detail | `dict[str, Tensor]` |
| `step5_steer_and_eval.py` | λ-sweep steered generation, length/richness metrics | `list[dict]` |
| `step6_visualize.py` | 3 publication figures (real COCO thumbnails in extremes panel) | `dict[str, path]` |
| `step7_analyze.py` | Rankings, findings, write findings.md | `dict` |
| `pipeline.py` | **Master orchestrator** (--demo or live) | All of the above |
| `demo_gradio.py` | **Interactive user-upload Gradio demo** (HF Spaces) | Gradio Blocks app |

---

## 🚀 How to Run

Make sure you are in the project root directory and your virtualenv is active:

```bash
source venv/bin/activate
export PYTHONPATH=.
```

### Option A: Demo Mode (No GPU Required) ✅ Recommended for HuggingFace Spaces

Uses pre-computed results bundled in `results/*.json`. Generates all 3 figures and findings.md in under 15 seconds.

```bash
venv/bin/python task/task_04/pipeline.py --demo
```

**Outputs:**
- `task/task_04/results/diversity_histogram.png` — diversity score distribution
- `task/task_04/results/diverse_vs_repetitive.png` — caption extremes panel
- `task/task_04/results/steering_lambda_sweep.png` — λ vs. caption length chart
- `task/task_04/results/findings.md` — written analysis

### Option B: Live GPU Inference

Downloads COCO val, runs nucleus sampling on 200 images and steering on 20 images. Requires a GPU (MPS or CUDA) and ~10 GB RAM.

```bash
venv/bin/python task/task_04/pipeline.py
```

### Option C: Individual Steps (Notebook / HuggingFace Inspection)

```python
# Step 1 — Load model
from task.task_04.step1_load_model import load_model
model, processor, device = load_model()

# Step 2 — Prepare data
from task.task_04.step2_prepare_data import load_val_data, build_style_sets
dataloader = load_val_data(processor, n=200, batch_size=4)
style_sets = build_style_sets(n=500)

# Step 3 — Diversity analysis
from task.task_04.step3_diversity_analysis import run_diversity_analysis
records = run_diversity_analysis(model, processor, dataloader, device)

# Step 4 — Steering vectors
from task.task_04.step4_steering_vectors import extract_steering_vectors
vectors = extract_steering_vectors(model, processor, style_sets, device)

# Step 5 — Steered generation
from task.task_04.step5_steer_and_eval import run_steering_eval
steering_results = run_steering_eval(model, processor, dataloader, device, vectors)

# Step 6 — Visualize
from task.task_04.step6_visualize import visualize_all
paths = visualize_all(records, steering_results)

# Step 7 — Analyze
from task.task_04.step7_analyze import analyze_results
findings = analyze_results(records, steering_results)
```

### Option D: Run Individual Steps Standalone

```bash
# Diversity analysis (precomputed)
venv/bin/python task/task_04/step3_diversity_analysis.py
venv/bin/python task/task_04/step3_diversity_analysis.py --live  # GPU inference

# Steering vectors (precomputed)
venv/bin/python task/task_04/step4_steering_vectors.py
venv/bin/python task/task_04/step4_steering_vectors.py --live

# λ sweep (precomputed)
venv/bin/python task/task_04/step5_steer_and_eval.py
venv/bin/python task/task_04/step5_steer_and_eval.py --live

# Regenerate figures only
venv/bin/python task/task_04/step6_visualize.py

# Print analysis only
venv/bin/python task/task_04/step7_analyze.py
```

---

## 🌡️ Understanding the Figures

### `results/diversity_histogram.png`
- **X-axis**: diversity score (unique n-grams / total n-grams)
- Red-shaded zone: repetitive (< 0.40)
- Blue-shaded zone: diverse (> 0.75)
- Dashed lines: thresholds; dotted line: mean score
- Look at the bimodal shape — it confirms that high and low diversity images are distinct populations

### `results/diverse_vs_repetitive.png`
- Left panel: top-3 most diverse images with all 5 captions
- Right panel: top-3 most repetitive images with all 5 captions
- **Image thumbnails**: actual COCO validation images are fetched via `datasets` streaming and embedded at the left column of each row. First run downloads 6 images; subsequent runs load from `results/images/`.
- Compare how different the captions look between the two groups

### `results/steering_lambda_sweep.png`
- X-axis: λ (negative = push toward shorter, positive = push toward detailed)
- Left Y-axis (orange): mean caption length in words
- Right Y-axis (purple): mean unique word count per caption
- The dashed vertical line at λ=0 is the unsteered baseline
- The slope of both lines confirms that steering is effective

---

## 📁 Folder Structure

```
task/task_04/
├── step1_load_model.py          # Component 1: Load BLIP + checkpoint
├── step2_prepare_data.py        # Component 2: COCO DataLoader + style sets
├── step3_diversity_analysis.py  # Component 3: Nucleus diversity (p=0.9)
├── step4_steering_vectors.py    # Component 4: BLIP hidden-state extraction
├── step5_steer_and_eval.py      # Component 5: Forward-hook λ sweep
├── step6_visualize.py           # Component 6: 3 publication figures
├── step7_analyze.py             # Component 7: Rankings & findings.md
├── demo_gradio.py               # Component 8: User-upload Gradio demo
├── pipeline.py                  # Master orchestrator (--demo or live)
└── results/
    ├── diversity_results.json       # Pre-computed per-image diversity records
    ├── steering_vectors.pt          # d_short2detail, d_short2medium tensors
    ├── steering_vectors_meta.json   # Steering vector metadata
    ├── steering_results.json        # λ-sweep metrics table
    ├── findings.md                  # Auto-generated written analysis
    ├── diversity_histogram.png      # Diversity score distribution
    ├── diverse_vs_repetitive.png    # Caption extremes panel (with real COCO images)
    ├── steering_lambda_sweep.png    # λ vs length/richness chart
    └── images/                      # Real COCO thumbnails (fetched on first run)
        ├── img_0.jpg
        ├── img_3.jpg
        └── ...                      # 6 total (top-3 diverse + top-3 repetitive)
```

---

## ⚙️ Dependencies

All dependencies are already in the project `requirements.txt`:

| Package | Used For |
|---------|---------|
| `transformers` | BLIP model loading, text encoder, text decoder |
| `torch` | Forward hooks, hidden-state arithmetic |
| `datasets` | COCO 2017 validation split |
| `matplotlib` | Histogram, text panel, dual-axis chart |
| `numpy` | Score aggregations |
| `tqdm` | Progress bars for live inference |

---

## 🔗 Connection to the Broader Project

- **Builds on Task 3**: Uses the same BLIP fine-tuned checkpoint (`outputs/blip/best/`) as the base model for caption generation and hidden-state extraction.
- **Complements the main app**: The diversity analysis exposes a limitation of the current inference pipeline — for simple images, multiple sampling behaves like greedy decode.
- **Novel capability**: Concept steering is a zero-shot technique that can be integrated into the Streamlit demo as a "style slider" — allowing users to interactively generate shorter or longer captions from the same image. The `demo_gradio.py` file provides a standalone Gradio interface for this.
- **Connects to Experiment 2 (beam search)**: Diversity analysis shows that nucleus sampling and beam search operate in different regimes — beam search maximises probability (low entropy), nucleus sampling controls entropy directly.
- **Leads into Task 5**: The caption diversity pipeline (step2–step3) is the data source for Task 5's toxicity analysis — the same BLIP caption generation flow is extended with safety classification.

---

**Author:** Manoj Kumar — March 2026
