# 🔬 VLM Caption Lab

**Compare how different Vision-Language Models look at images while writing captions — four architectures, one dataset, one evaluation metric.**

VLM Caption Lab is a complete Python toolkit for training, evaluating, and interactively comparing four fundamentally different approaches to **image captioning** (the task of generating a text description of a photograph). It includes a unified training pipeline, quality evaluation using CIDEr scores, three reproducible experiments, and an interactive Streamlit web demo.

### 📚 Key Documentation
- **[simplified_overview_vlm_image_captioning_project.md](./simplified_overview_vlm_image_captioning_project.md)**: A non-technical overview explaining the four models, experiments, and learnings from the project.
- **[detailed_technical_report_cross_attention_vlm_image_captioning.md](./detailed_technical_report_cross_attention_vlm_image_captioning.md)**: An in-depth technical report covering the architectures, training methodology, and evaluation metrics used in this project.

---

## Architecture Comparison

| Architecture | How It Looks at the Image | Total Parameters | Best CIDEr Score |
|---|---|---|---|
| **BLIP** | Selective gated attention — looks at image only when needed | 224M | **0.6199** (optimized) |
| **ViT-GPT2** | Full attention — looks at entire image for every word | 239M | ~0.55 |
| **GIT** | Memory-based — memorizes image first, writes from memory | 177M | ~0.54 |
| **Custom VLM** | Built from scratch — Shakespeare decoder + visual bridge | 103M (16.2M trainable) | **0.2863** |

> **What is CIDEr?** CIDEr (Consensus-based Image Description Evaluation) compares the model's caption to five human-written descriptions of the same image. Higher = better. A score of 1.0 means perfect overlap with human references.

---

## 🌐 Live Demo & Deployment

**The easiest way to test this project is via the live web demo.**
> 👉 **[Live Web Demo](https://huggingface.co/spaces/griddev/project_02_DS)**

*(If deploying yourself, see the `DEPLOYMENT_GUIDE.md` file for instructions on hosting this securely and for free on Hugging Face Spaces).*

---

## Quick Start (Local Run)

If you prefer to run this locally rather than using the web demo, follow these steps. 

> ⚠️ **Note on Weights**: You do *not* need to train the models yourself to test the app.
> - Base model weights (BLIP, ViT-GPT2) will download automatically from Hugging Face on the first run.
> - The Custom VLM text-decoder weights (`shakespeare_transformer.pt`) are included in this repo.
> - **To skip training completely**, you only need to run `streamlit run app.py`!

### Prerequisites

- Python 3.9 or newer
- macOS with Apple Silicon (MPS) or Linux with a CUDA GPU
- ~8 GB disk space for model checkpoints

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd project_02

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Verify that GPU acceleration is available
python -c "import torch; print('MPS:', torch.backends.mps.is_available()); print('CUDA:', torch.cuda.is_available())"
```

### Dependencies

| Package | What It Does |
|---|---|
| `torch` | Deep learning framework (training and inference) |
| `transformers` | Load pre-trained BLIP, ViT-GPT2, and GIT models from HuggingFace |
| `datasets` | Download and load MS-COCO caption dataset from HuggingFace |
| `streamlit` | Interactive web demo interface |
| `pycocoevalcap` | Compute CIDEr scores (caption quality metric) |
| `detoxify` | Safety filter — checks captions for toxic or offensive content |
| `Pillow` | Image loading and processing |
| `accelerate` | Training efficiency utilities |

---

## 🚀 What to Expect on First Run

When someone clones this repository and runs `streamlit run app.py` (or `train.py`) for the very first time, here is exactly what happens:

1. **Automatic Model Downloads**: You do *not* need to manually download any heavy weights for BLIP, ViT-GPT2, or GIT. The `transformers` library will automatically download the base weights from HuggingFace the first time you select them. 
2. **Download Time**: This initial download may take a few minutes depending on your internet connection (BLIP is ~900MB, ViT-GPT2 is ~1GB). It will be cached locally on your machine for all future runs, so subsequent loads will be nearly instant.
3. **Custom VLM Weights**: The `shakespeare_transformer.pt` file (~71MB) included in this repository contains the pre-trained text decoder for the Custom VLM. By including it in the repo, the Custom VLM is ready to generate Shakespearean text immediately without any downloading.
4. **Fine-Tuned Weights**: To use the "Fine-tuned (Best)" or "Fine-tuned (Latest)" options in the web app, you must first run the training scripts (`python train.py --model [name]`). The training scripts will automatically create an `outputs/` directory and save your fine-tuned weights there.

---

## Training

All four models are trained through one unified script:

```bash
# Train individual models
python train.py --model blip          # ~1.5 hours on Apple Silicon
python train.py --model vit_gpt2      # ~1 hour
python train.py --model git           # ~20 minutes
python train.py --model custom        # ~3 hours (15 epochs)
```

### What happens during training

1. **Dataset loading** — Downloads MS-COCO captions from HuggingFace (cached after first download)
2. **Training** — Images are processed by the vision encoder, captions by the text decoder
3. **Validation** — After each epoch, computes validation loss + CIDEr score on held-out images
4. **Checkpointing** — Saves two checkpoints:
   - `outputs/{model}/best/` — The model with the **highest CIDEr score** (use this for evaluation)
   - `outputs/{model}/latest/` — The most recent epoch (use for debugging or continuing training)

### Key hyperparameters

| | BLIP | ViT-GPT2 | GIT | Custom VLM |
|-|---|---|---|---|
| Training epochs | 3 | 3 | 3 | 15 |
| Learning rate | 1e-5 | 2e-5 | 2e-5 | 1e-4 / 5e-5 |
| Batch size | 16 | 8 | 8 | 16 |
| Effective batch size | 64 | 32 | 32 | 64 |
| Training images | 30,000 | 15,000 | 15,000 | 15,000 |

---

## Evaluation

### Basic evaluation

```bash
# Evaluate a single model (computes CIDEr score)
python eval.py --model blip --weights best

# Evaluate with pre-trained weights (no fine-tuning)
python eval.py --model blip --weights base

# Compare all models side by side
python eval.py --model all --weights best
```

### Experiments

```bash
# Cross-attention masking experiment: what happens when we hide parts of the image?
python eval.py --model blip --ablation --weights best

# Decoding parameter sweep: find the best beam search settings
python eval.py --model blip --sweep --weights best

# Caption filtering analysis: does training data quality matter?
python eval.py --model blip --data-prep-analysis --weights best
```

### Custom decoding settings

```bash
python eval.py --model blip --weights best \
    --num_beams 10 \
    --max_new_tokens 50 \
    --length_penalty 1.2
```

### All command-line options

| Flag | Values | Default | What It Controls |
|---|---|---|---|
| `--model` | blip, vit_gpt2, git, custom, all | blip | Which model(s) to evaluate |
| `--weights` | base, finetuned, best | base | Which checkpoint to load |
| `--eval_batches` | any integer | 25 | How many validation batches to evaluate |
| `--num_beams` | 1–10+ | 10 | Beam search width (more = better but slower) |
| `--max_new_tokens` | 10–100 | 50 | Maximum caption length |
| `--length_penalty` | 0.5–2.0 | 1.2 | < 1.0 = longer captions, > 1.0 = shorter |
| `--ablation` | flag | off | Run the cross-attention masking experiment |
| `--sweep` | flag | off | Run the decoding parameter sweep |
| `--data-prep-analysis` | flag | off | Run the caption filtering comparison |

---

## Streamlit Demo

```bash
streamlit run app.py
```

The demo provides three tabs:

### 🖼️ Caption Tab
Upload any image and generate a caption. Choose which model to use, which checkpoint (pre-trained or fine-tuned), and which generation mode.

### 📊 Compare All Models Tab
Run all four architectures simultaneously on the same image. Results appear in a side-by-side grid with a summary table showing each model's approach and caption.

### 📈 Experiment Results Tab
Browse pre-computed results from all three experiments.

### Sidebar Controls
- **Weight Source** — Switch between pre-trained models and your fine-tuned checkpoints
- **Architecture** — Select any of the four models (each has an info card explaining its approach)
- **Generation Mode** — Choose masking modes for BLIP/ViT-GPT2 or Shakespeare Prefix for Custom VLM
- **Advanced Controls** — Adjust beam width, temperature, length penalty, top-k, and top-p

> **Safety:** All captions pass through a toxicity filter (`detoxify`) before being displayed.

---

## Configuration

Hyperparameters are managed through Python dataclasses in `configs/`:

```
configs/
├── base_config.py          # Shared defaults (batch size, image size, optimizer settings)
├── blip_config.py          # BLIP-specific overrides
├── vit_gpt2_config.py      # ViT-GPT2-specific overrides
├── git_config.py           # GIT-specific overrides
└── custom_vlm_config.py    # Custom VLM overrides (decoder architecture, learning rates)
```

Access any config in code:

```python
from configs import get_config
cfg = get_config("blip")  # Returns BlipConfig instance with all settings
```

---

## Experiments & Key Results

### 1. Cross-Attention Masking: What Happens When We Hide Image Patches?

| What We Did | CIDEr Score | Change |
|---|---|---|
| Showed the full image | 0.5371 | — Baseline |
| Hid 50% of image patches randomly | 0.5371 | **No change** |
| Showed only the center of the image | 0.5371 | **No change** |
| Compressed entire image to 1 token | 0.0008 | **−99.8%** |

**Takeaway:** Half the image patches are redundant, but spatial structure is essential.

### 2. Beam Search Settings: What Produces the Best Captions?

**Best configuration found:** beam_size=10, length_penalty=1.2, max_tokens=50 → **CIDEr: 0.6199**

More beams and slight preference for conciseness improve caption quality by ~13%.

### 3. Caption Filtering: Does Training Data Quality Matter?

| Strategy | CIDEr |
|---|---|
| Raw (no filtering) | **0.6359** |
| Filtered (5–25 words) | 0.5877 |

Raw works best for this already-clean dataset. Filtering recommended for noisier data.

### 4. Caption Diversity & Concept Steering (Task 4)

| Analysis | Key Result |
|---|---|
| Mean diversity score (200 COCO images, 5 captions each, p=0.9) | **0.671** |
| Repetitive images (score < 0.40) | **30%** of images |
| Diverse images (score > 0.75) | **44%** of images |
| Steering at λ=+2.0 vs λ=0 (h += λ × d_short2detail) | **+65% caption length** |
| Optimal steering range (fluent style shift) | **λ ∈ [0.5, 1.0]** |

**Takeaway:** Simple objects generate repetitive captions even at high nucleus-sampling `p`. Concept steering via mean hidden-state differences shifts generation style controllably without retraining.

### 5. Toxicity & Bias Detection with Mitigation (Task 5)

| Analysis | Key Result |
|---|---|
| Captions scored (unitary/toxic-bert, 6-label) | **1000 COCO captions** |
| Toxicity flagging rate (max score ≥ 0.5) | **1.4%** — mild pejoratives only |
| Captions with gender/age stereotype pattern | **29.1%** — inherited from COCO data |
| Most common bias | Women → domestic roles, Men → sports |
| Mitigation (BadWords logit penalty, 200 token sequences) | **62.5%** of flagged captions cleaned |
| Mean toxicity score reduction after mitigation | **−0.55** score units |
| BLEU-2 degradation after filtering | **< 2%** — content preserved |

**Takeaway:** BLIP almost never generates overtly toxic captions, but gender and age stereotyping from the COCO corpus appears in ~29% of outputs. Logit penalty mitigation removes toxicity with minimal quality impact.

---

## Newer Implemented Tasks (Task 01 to Task 05)

The repository now includes five standalone task pipelines under `task/`, each with modular step scripts, a master `pipeline.py`, and reproducible outputs in each task's `results/` folder.

| Task | What Was Implemented | Key Outcome | Run |
|---|---|---|---|
| **Task 01** | BLIP optimization pipeline: memory-efficient fine-tuning (gradient checkpointing + AMP), ONNX export, CoreML conversion, 4-bit quantization, benchmarking | CoreML INT4 path achieved ~3.1x speedup and ~4.5x model size reduction with small BLEU drop | `venv/bin/python task/task_01/pipeline.py --demo` |
| **Task 02** | Attention grounding pipeline: BLIP cross-attention visualization evolved from raw averaging -> GradCAM -> multi-layer attention flow rollout + IoU auto-grading with OWL-ViT | Attention Flow significantly improved grounding precision (notably object-level IoU) | `venv/bin/python task/task_02/pipeline.py` |
| **Task 03** | Beam search x length-penalty ablation grid (9 configs) with CIDEr/BLEU/METEOR/ROUGE + latency + Pareto analysis | Best quality at `beam=5, length_penalty=1.0`; best balanced deployment config at `beam=3, length_penalty=1.0` | `venv/bin/python task/task_03/pipeline.py --demo` |
| **Task 04** | Caption diversity analysis (nucleus sampling) + hidden-state steering vectors for style control (`h + λd`) without retraining | Diversity varies strongly by scene complexity; λ-sweep shows controllable caption length/detail | `venv/bin/python task/task_04/pipeline.py --demo` |
| **Task 05** | Safety/fairness audit pipeline: 6-label toxicity scoring, stereotype-pattern bias audit, bad-words logit mitigation, fairness report generation | Low severe toxicity but measurable stereotype patterns; mitigation reduces flagged toxicity with minimal quality loss | `venv/bin/python task/task_05/pipeline.py --demo` |

### Task Directory Overview

```text
task/
├── task_01/  # BLIP optimization + ONNX/CoreML/quantization
├── task_02/  # Attention visualization + grounding auto-grading
├── task_03/  # Beam/length ablation + Pareto trade-off analysis
├── task_04/  # Diversity analysis + concept steering vectors
└── task_05/  # Toxicity/bias audit + mitigation + fairness report
```

---

## Project Structure

```
project_02/
├── app.py                              # Streamlit web demo (3 tabs)
├── config.py                           # Backward-compatible config wrapper
├── data_prep.py                        # Dataset loading + caption filtering
├── eval.py                             # CIDEr evaluator + experiment runner
├── train.py                            # Unified training loop for all 4 models
├── requirements.txt                    # Python dependencies
├── input.txt                           # Shakespeare corpus (vocabulary source)
├── shakespeare_transformer.pt          # Pre-trained Shakespeare decoder weights
│
├── configs/                            # Hyperparameter configs
│   ├── base_config.py                  # Shared defaults
│   ├── blip_config.py                  # BLIP settings
│   ├── vit_gpt2_config.py             # ViT-GPT2 settings
│   ├── git_config.py                   # GIT settings
│   └── custom_vlm_config.py            # Custom VLM settings
│
├── models/                             # Model implementations
│   ├── blip_tuner.py                   # BLIP (gated cross-attention)
│   ├── vit_gpt2_tuner.py              # ViT-GPT2 (full cross-attention)
│   ├── git_tuner.py                    # GIT (no cross-attention)
│   └── custom_vlm.py                  # Custom VLM (visual prefix-tuning)
│
├── experiments/                        # Experiment scripts and results
│   ├── ablation_study.py              # Image masking experiment
│   ├── parameter_sweep.py             # Beam search settings sweep
│   ├── data_prep_analysis.py          # Caption filtering comparison
│   └── cross_attention_patterns.py    # Architecture comparison table
│
├── task/                               # Standalone task analyses
│   ├── task_01/                        # BLIP fine-tuning + ONNX/CoreML export
│   ├── task_02/                        # Attention flow & GradCAM visualizations
│   ├── task_03/                        # Beam search × length penalty ablation (9 configs)
│   ├── task_04/                        # Caption diversity & concept steering vectors
│   │   ├── step1_load_model.py         # BLIP loader with fine-tuned checkpoint fallback
│   │   ├── step2_prepare_data.py       # COCO DataLoader + style caption sets
│   │   ├── step3_diversity_analysis.py # 5 nucleus captions/image, diversity score
│   │   ├── step4_steering_vectors.py   # Mean hidden states → steering directions
│   │   ├── step5_steer_and_eval.py     # h += λ×d forward-hook decode, λ sweep
│   │   ├── step6_visualize.py          # 3 publication figures (incl. real COCO thumbnails)
│   │   ├── step7_analyze.py            # Findings report + findings.md
│   │   ├── demo_gradio.py              # User-upload Gradio demo for HF Spaces
│   │   ├── pipeline.py                 # Master orchestrator (--demo or live)
│   │   └── results/                    # Pre-computed JSONs + generated PNGs + image thumbnails
│   └── task_05/                        # Toxicity & bias detection with mitigation
│       ├── step1_load_model.py         # BLIP + unitary/toxic-bert loader
│       ├── step2_prepare_data.py       # 1000-caption generator
│       ├── step3_toxicity_score.py     # 6-label toxicity scoring
│       ├── step4_bias_audit.py         # Lexicon stereotype detection & frequency table
│       ├── step5_mitigate.py           # BadWords logit penalty (200 token sequences)
│       ├── step6_visualize.py          # 3 fairness figures
│       ├── step7_fairness_report.py    # Full markdown fairness report generator
│       ├── pipeline.py                 # Master orchestrator (--demo or live)
│       └── results/                    # Scores, audit JSON, 3 PNGs, fairness_report.md
│
├── outputs/                            # Saved model checkpoints
│   ├── blip/{best,latest}/
│   └── custom_vlm/{best,latest}/
│
├── detailed_technical_report_cross_attention_vlm_image_captioning.md
├── simplified_overview_vlm_image_captioning_project.md
└── README.md                           # This file
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Training Framework | PyTorch + HuggingFace Transformers |
| Dataset | MS-COCO Captions (via HuggingFace Datasets) |
| Evaluation Metric | CIDEr (via pycocoevalcap) |
| Safety Filter | detoxify (toxicity detection) |
| Web Demo | Streamlit |
| Hardware | Apple Silicon Mac with MPS acceleration |

---

## Author

**Manoj Kumar** — March 2026
