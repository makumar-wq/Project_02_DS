✅ Image size set to 224px
✅ Gradient checkpointing enabled (BLIP)
✅ BLIP loaded on mps: Salesforce/blip-image-captioning-base (224.0M params)

============================================================
  Ablation Mode : BASELINE
  Beams=4  MaxTokens=32  LenPenalty=1.0
============================================================
  ✅ CIDEr [baseline]: 0.5371

============================================================
  Ablation Mode : RANDOM_DROPOUT
  Beams=4  MaxTokens=32  LenPenalty=1.0
============================================================
  ✅ CIDEr [random_dropout]: 0.5371

============================================================
  Ablation Mode : CENTER_FOCUS
  Beams=4  MaxTokens=32  LenPenalty=1.0
============================================================
  ✅ CIDEr [center_focus]: 0.5371

============================================================
  Ablation Mode : SQUINT
  Beams=4  MaxTokens=32  LenPenalty=1.0
============================================================
  ✅ CIDEr [squint]: 0.0008


============================================================
  Cross-Attention Ablation Results (CIDEr)
  Beams=4  MaxTokens=32  LenPenalty=1.0
============================================================
  Mode                           CIDEr    Δ Baseline
------------------------------------------------------------
  baseline                      0.5371  +     0.0000
  random_dropout                0.5371  +     0.0000
  center_focus                  0.5371  +     0.0000
  squint                        0.0008      -0.5363
============================================================
============================================================
