# How I Built a System That Teaches Computers to Describe Photographs

**A non-technical overview of the VLM Caption Lab project**  
*Author: Manoj Kumar | 4 March 2026*

---

## What Is This Project About?

Imagine showing a photograph to a friend and asking them to describe it in one sentence. They might say, *"A man in a suit standing in front of a tree,"* or *"A tennis match in a large arena with a crowd watching."* For us, this is effortless — our brains process the entire image, identify the objects, understand the scene, and produce a fluent sentence in under a second.

For a computer, this is remarkably difficult. The technical name for this task is **"image captioning,"** and it lives at the crossroads of two hard problems: understanding what is in an image (computer vision) and writing grammatically correct, meaningful sentences (natural language generation).

This project explores that challenge — **but I did not just build one system. I built and compared four of them,** each with a fundamentally different approach to the core problem of looking at the image while writing about it.

---

## The Four Models I Built (And Why They Are Different)

Think of image captioning like a person looking at a painting while narrating what they see into a microphone. The four models I compared differ in **how the person glances at the painting while they talk.**

---

### 🔵 Model 1: BLIP — The Selective Glancer

**How it works :** BLIP is like a narrator who has trained themselves to only glance at the painting when they need to. When they are saying generic words like "a" or "the" or "is," they just focus on their own sentence. When they need to mention something specific — like "bicycle" or "standing" — they look up at the painting to confirm what they see.

**Why this is smart:** Most words in a sentence are structural, not visual. There is no need to look at the image to say "the" or "in front of." BLIP learns when to look and when not to, which prevents it from getting confused by too much visual information.

**Size:** 224 million parameters  
**Best CIDEr score:** **0.62** (with optimized settings)

---

###  Model 2: ViT-GPT2 — The Constant Starer

**How it works in plain English:** ViT-GPT2 takes the opposite approach — for every single word, it stares at the entire painting. Writing "a"? Look at the whole image. Writing "dog"? Look at the whole image. Writing "the"? Still looking at the whole image.

**Why this still works:** Even though it is wasteful, staring at everything guarantees the model never misses any visual detail. The downside is that this constant stream of visual information can sometimes confuse the language part of the model.

**Size:** 239 million parameters  
**Typical CIDEr score:** ~0.55

---

###  Model 3: GIT — The Memorizer

**How it works in plain English:** GIT does something clever — instead of switching between looking at the painting and writing words, it first memorizes the entire painting and then writes the caption purely from memory.

In technical terms, GIT converts the image into a set of structured "memory notes" and places them at the beginning of its sentence. Then it processes everything — image memories and text — in one continuous stream. There is no separate "looking at the painting" step.

**Why this is elegant:** It is simpler and faster because it does not need the extra machinery for looking back and forth between image and text. The entire intelligence is in one unified processing step.

**Size:** 177 million parameters (smallest of the four)  
**Typical CIDEr score:** ~0.54

---

###  Model 4: Custom VLM — The Shakespeare Bot Learning Modern English

**How it works in plain English:** This is the most experimental model, and the one **I built entirely from scratch.** Imagine a narrator who grew up reading only Shakespeare and has never seen a photograph before. You give them a pair of glasses (a visual encoder — something that can look at images) and a translator (a small bridging network) and ask them to describe modern photographs.

The "Shakespeare bot" is a text generator I had previously trained on the complete works of Shakespeare. It knows English grammar and sentence structure — but in Elizabethan English. The challenge was teaching it to (a) understand images through the "glasses" and (b) speak in modern, descriptive English instead of iambic pentameter.

**Why I built this:** To understand what minimum set of components you need to make a functioning vision-language model. Instead of downloading a ready-made model with billions of parameters, I wanted to see if I could glue together a vision model and a text model with just a small trainable "bridge" in between.

**Size:** 103 million parameters total, but only **16.2 million are trainable** (the rest are frozen)  
**Best CIDEr score:** **0.2863** (still learning, but it works!)

---

## What Is CIDEr? (The Score We Use to Measure Quality)

Throughout this summary, I mention "CIDEr scores." Here is what they mean:

**CIDEr** stands for "Consensus-based Image Description Evaluation." In simple terms, it compares the caption our model generates to **five human-written descriptions** of the same image.

- It counts how many meaningful words overlap between the model's caption and the human captions
- It gives more weight to descriptive words (like "bicycle" or "stadium") than common words (like "the" or "is")
- **A higher score means the computer's description sounds more like what a human would write**

| CIDEr Score | What It Means |
|---|---|
| 0.00 | Completely wrong — no overlap with human descriptions |
| 0.20–0.30 | Early stage — some correct words, but the sentence may be awkward |
| 0.50–0.60 | Good — clearly related to the image, mostly sensible |
| 0.80–1.00 | Excellent — almost indistinguishable from a human caption |

---

## The Custom Model Story: A Journey of Debugging and Discovery

This is the part of the project I am most proud of, because it taught me the most about how machine learning actually works in practice — not just in theory, but when things go wrong.

### Chapter 1: "Why Is It Speaking Gibberish?"

My first attempt at the Custom VLM produced output like this:

> *"iGiiiiiGiviqiGqiFliqiGidlidiliGilFGilqiiiqiiiiGii"*

That is not English. That is not even Shakespeare. It is random noise.

**The problem:** The connection between the "glasses" (the image encoder) and the "brain" (the Shakespeare text generator) was too weak. I was using a single mathematical transformation to convert visual information into text information. Think of it like trying to translate a painting into a poem by only measuring the canvas size — you are missing all the important details.

**CIDEr score at this stage: 0.0000 — literally zero.**

### Chapter 2: "Better Connection, But Still Broken"

I upgraded the connection to a more powerful two-layer network. This is like upgrading from a basic dictionary to a bilingual tutor who understands context. The training measurements started improving — the numbers were going down, which normally means the model is learning.

But the output was still gibberish.

After days of investigation, I found the real problem — and it was a doozy:

> **When I loaded the Shakespeare brain into the model, 97% of the brain weights failed to load. Silently. No error message. No warning. The software just said "everything is fine" and moved on.**

My model had been running on a **randomly initialized brain** — essentially trying to learn language from scratch while simultaneously trying to learn to describe images. Imagine asking someone with amnesia to write poetry about something they've never seen. That's what my model was trying to do.

**Why did this happen?** The two models (Shakespeare and my VLM) stored their internal knowledge in slightly different formats. It is like trying to load a Word document into Excel — both are files, but the internal structure is completely different. The software saw the mismatched formats and just... skipped everything. Without telling me.

### Chapter 3: "It Finally Speaks!"

The fix required three things:
1. **Match the formats** — Make the new model structure identical to the Shakespeare model's structure (8 layers, 8 attention heads, matching dimensions)
2. **Translate the weights** — Write custom code to convert the Shakespeare data from one format to another
3. **Let the brain learn** — Instead of freezing the Shakespeare knowledge, let the model slowly adapt from old English to modern descriptions

**The result was immediate.** From the very first training session after the fix, the improvement was dramatic:

> Before fix: *"iGiiiiiGiviqiGqiFliqiGidlidiliGilFGilqiiiqiiiiGii"* (CIDEr: 0.0000)  
> After fix: *"man in the bluess and white play with and a pizza"* (CIDEr: 0.2863)

Not perfect. Not even grammatically correct. But it is **clearly English**, it is **clearly attempting to describe an image**, and it went from zero to something meaningful. The word "man" appeared because the image showed a man. The model learned real English words and connected them to visual concepts.

---

## What We Tested: The Three Experiments

### Experiment 1: "Can We Cover Part of the Image?"

I blocked parts of the image from the model and measured whether the captions got worse. The results were genuinely surprising:

| What We Did | Effect on Caption Quality |
|---|---|
| Showed the **full image** | Baseline quality (CIDEr: 0.5371) |
| **Hid 50%** of the image randomly | **No change at all** (CIDEr: 0.5371) |
| Showed **only the center** (removed background) | **No change at all** (CIDEr: 0.5371) |
| **Compressed everything** into one tiny summary | **Complete failure** (CIDEr: 0.0008 — a 99.8% drop) |

**What this teaches us:** Images contain a lot of redundant information. You can throw away half the visual data and still get perfectly good captions. But if you compress everything into a single summary, you lose the information about **where things are** relative to each other — and that spatial information turns out to be essential for describing a scene.

### Experiment 2: "What Settings Produce the Best Captions?"

When a model generates a caption, it uses a search algorithm that considers multiple possible sentences and picks the best one. I tested **18 different combinations** of settings and found:

- **Considering more candidate sentences (10 instead of 3) helped significantly** — about 13% improvement
- **Slightly encouraging shorter captions helped** — models tend to ramble when given too much freedom
- **Best combination found: CIDEr score of 0.6199** (up from 0.48 with the worst settings)

### Experiment 3: "Does Caption Quality During Training Matter?"

I compared different strategies for selecting which human captions to show the model during training:

| Strategy | CIDEr Score |
|---|---|
| Use any random caption | **0.6359** ← best for this clean dataset |
| Use only short captions (≤ 9 words) | 0.6016 |
| Use only medium-length captions (5–25 words) | 0.5877 |
| Use only long captions (≥ 12 words) | 0.5389 |

**Bottom line:** For this particular dataset (which is already well-curated), using raw unfiltered captions works best. But filtering is recommended for noisier datasets.

---

## The Interactive Demo

I built a web application where anyone can try the models themselves:

- **Upload any photo** and get a caption from any of the four models
- **Compare all four models** side by side on the same image — see how each one describes the same picture differently
- **Switch between pre-trained and fine-tuned** versions of each model
- **Adjust generation settings** — control how the model searches for the best caption
- **View experiment results** — browse all the findings from the three experiments

Every generated caption goes through a **safety filter** before being shown, because AI models can occasionally produce inappropriate descriptions. The filter uses a toxicity detection model to catch and block offensive content.

---

## Summary of Results

| Model | Approach | CIDEr Score | Key Strength |
|---|---|---|---|
| **BLIP** | Selective looking | **0.62** (best settings) | Best quality — knows when to look vs. when to focus on grammar |
| **ViT-GPT2** | Constant looking | ~0.55 | Strong baseline — full visual access at all times |
| **GIT** | Memory-based | ~0.54 | Elegant and efficient — no cross-attention needed at all |
| **Custom VLM** | Built from scratch | **0.29** | Proof of concept — works despite tiny vocabulary and Shakespeare origins |

---

## What I Actually Learned

1. **There is no single best way to connect vision and language.** BLIP's selective attention works best overall, but GIT's simpler approach is surprisingly competitive — proving that you do not always need complex mechanisms to solve complex problems.

2. **Silent failures are the most dangerous bugs in machine learning.** The most time-consuming problem in this project was a weight-loading failure that produced zero error messages. The model ran, the loss decreased, everything looked normal — but 97% of the model was running on random noise. I now always verify that weights loaded correctly.

3. **The number your model optimizes during training is not necessarily the number that tells you if it is doing a good job.** Training loss went down steadily, but the captions were still gibberish. Only when I started measuring CIDEr (actual caption quality) did I understand what was really happening.

4. **Small models can learn big tasks with the right approach.** The Custom VLM has only 16.2 million trainable parameters — roughly 1/15th the size of BLIP — yet it learned to produce recognizable English descriptions of images by building on existing Shakespeare knowledge.

5. **Images are surprisingly redundant.** You can literally hide half the image and the model generates identical captions. But structure matters — where objects are relative to each other is more important than being able to see every pixel.

---

## What Could Be Improved Next

If I continue this project, the highest-impact improvements would be:

- **Better vocabulary:** The Custom VLM currently spells everything letter-by-letter (65 characters). Switching to a word-piece vocabulary (thousands of tokens) would dramatically reduce the difficulty.
- **Stronger language foundation:** Replacing the Shakespeare decoder with a modern language model like GPT-2 would give the model native modern English instead of having to translate from Elizabethan.
- **More training data:** We currently use only 18% of the available dataset images.

---

*Project by Manoj Kumar, March 2026*
