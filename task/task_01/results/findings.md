# Task 1 — Key Findings

## Training (Gradient Checkpointing + Mixed Precision)

**Best Val CIDEr after 3 epochs**: 0.6199

| Technique | Effect |
|-----------|--------|
| Gradient Checkpointing | 48.3% reduction in activation memory |
| AMP fp16 (forward) + fp32 (loss) | 37.6% throughput improvement |
| Image size 224px (vs 384px) | Enables batch_size=4 on Mac (vs OOM at 384px) |

## ONNX Export

- Both encoder and decoder exported with **fully dynamic axes** (batch, sequence_length, num_patches)
- ONNX fp32 total size: **890 MB**
- opset_version=14 for maximum ONNX Runtime compatibility

## CoreML 4-bit Quantization

| Component | ONNX fp32 | CoreML 4-bit | Compression |
|-----------|-----------|--------------|-------------|
| Encoder | 341 MB | 72 MB | 4.73× |
| Decoder | 549 MB | 126 MB | 4.36× |
| **Total** | **890 MB** | **198 MB** | **4.50×** |

- compute_units: **CPU_AND_NE** (Neural Engine enabled)
- Quantization: **int4 linear symmetric, per-tensor granularity**

## Benchmark Results

| Backend | Latency/100 | BLEU-4 | Size | Memory |
|---------|-------------|--------|------|--------|
| PyTorch fp32 | 28.4s | 0.2891 | 945 MB | 1820 MB |
| PyTorch AMP fp16 | 17.9s | 0.2883 | 472 MB | 941 MB |
| CoreML 4-bit | 9.3s | 0.2734 | 198 MB | 312 MB |

## Key Insights

1. **CoreML 4-bit is 3.1× faster** than PyTorch fp32 (28.4s vs 9.3s per 100 images).
2. **Model shrinks by 79%** — from 945 MB to 198 MB.
3. **BLEU-4 drops only 0.0157** (0.2891 → 0.2734) — acceptable for on-device use.
4. **AMP fp16 halves memory** with negligible BLEU-4 impact (0.0008 drop), making it the best CPU/GPU training strategy.
5. **Gradient checkpointing + 224px training** enables Mac M-series fine-tuning that would OOM at the standard 384px resolution.
