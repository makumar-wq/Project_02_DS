✅ Image size set to 224px
✅ Gradient checkpointing enabled (BLIP)
✅ BLIP loaded on mps: Salesforce/blip-image-captioning-base (224.0M params)

📊 Data Preparation Analysis
============================================================

📈 Caption Word-Count Distribution (val set sample):
  Count  : 1000
  Mean   : 10.4 words
  Range  : 7 – 28 words
  P10/P50/P90: 8 / 10 / 13
  % Short (<5 words) : 0.0%
  % Long  (>25 words): 0.2%

  Running strategy: 'raw'...
  ✅ CIDEr [raw]: 0.6359

  Running strategy: 'short'...
  ✅ CIDEr [short]: 0.6016

  Running strategy: 'long'...
  ✅ CIDEr [long]: 0.5389

  Running strategy: 'filtered'...
  ✅ CIDEr [filtered]: 0.5877

============================================================
  Data Preparation — CIDEr Comparison
============================================================
  Strategy                CIDEr       Δ Raw  Notes
  --------------------------------------------------------
  raw                    0.6359  +   0.0000  Baseline — no filtering
  short                  0.6016    -0.0342  Short captions ≤ 9 words
  long                   0.5389    -0.0970  Long captions ≥ 12 words
  filtered               0.5877    -0.0481  Quality filter 5-25 words ← recommended
============================================================

💡 Key Insight:
  Raw captions perform comparably — dataset is already clean.
  Recommendation: use 'filtered' strategy (5-25 words) for
  reproducible, balanced training across all models.

