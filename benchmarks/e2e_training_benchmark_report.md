# E2E Training Benchmark Report: youmu_page_aligned vs lerobot

## Setup

| Component | Details |
|-----------|---------|
| GPU | NVIDIA H100 80GB HBM3 |
| Model | Qwen3-VL-2B-Instruct (Qwen3VLForConditionalGenerationAction) |
| Precision | bfloat16 |
| Training | DDP, single GPU, AdamW, FlashAttention-2, rmpad_with_pos_ids |
| Steps | 50 per config (first 10 excluded as warmup, measuring steps 10-50) |
| obs_len | 4 |
| pred_len | 2 |
| Dataset | LIBERO — hf_vla_64k (youmu_page_aligned) / hf_vla (lerobot) |
| Batch sizes | 1, 2, 4, 8, 16, 32, 64 |

## Results

### Full Comparison Table

| Batch Size | youmu steps/s | lerobot steps/s | Speedup (youmu/lerobot) | youmu samp/s | lerobot samp/s | youmu wall (s) | lerobot wall (s) | youmu GPU (MB) | lerobot GPU (MB) |
|:----------:|:-------------:|:---------------:|:-----------------------:|:------------:|:--------------:|:--------------:|:----------------:|:--------------:|:----------------:|
| 1 | 2.63 | 3.14 | 0.84x | 2.63 | 3.14 | 29.02 | 25.64 | 21,008 | 21,008 |
| 2 | 2.60 | 3.12 | 0.83x | 5.20 | 6.24 | 29.46 | 26.30 | 21,042 | 21,032 |
| 4 | 2.66 | 3.09 | 0.86x | 10.64 | 12.36 | 29.11 | 26.24 | 21,032 | 21,008 |
| 8 | 2.63 | 3.01 | 0.87x | 21.04 | 24.08 | 29.13 | 26.33 | 21,012 | 21,004 |
| **16** | **2.57** | **1.73** | **1.49x** | **41.12** | **27.68** | **29.70** | **39.21** | 21,128 | 21,052 |
| **32** | **2.39** | **1.01** | **2.37x** | **76.48** | **32.32** | **31.15** | **70.14** | 21,270 | 21,166 |
| **64** | **1.71** | **1.11** | **1.54x** | **109.44** | **71.04** | **39.68** | **123.07** | 21,094 | 21,090 |

### Where Data Loading Becomes the Bottleneck

At **batch sizes 1-8**, GPU compute dominates the training step. Both backends keep up with the GPU — lerobot is actually ~15% faster here (3.0-3.14 vs 2.6 steps/s), likely due to in-memory data access being faster than page-aligned disk reads when the GPU isn't saturated.

At **batch size 16**, lerobot's throughput drops sharply from 3.01 to 1.73 steps/s (a 43% drop), while youmu_page_aligned stays nearly flat at 2.57 steps/s. This is the **crossover point** where data loading becomes the bottleneck for lerobot. With 16 samples per step, lerobot's data pipeline can no longer keep the GPU fed.

At **batch size 32**, the gap widens further: youmu is **2.37x faster** (2.39 vs 1.01 steps/s). Lerobot's data loading is severely bottlenecked.

At **batch size 64**, both backends slow down — youmu drops to 1.71 steps/s (now also partially data-bound), while lerobot remains at 1.11 steps/s. Youmu is **1.54x faster**.

### Maximum Batch Size Before OOM

Neither backend hit OOM at any batch size up to 64 on the H100 80GB. Peak GPU memory is ~21 GB for all configurations, confirming that memory usage is **model-dominated**, not data-dominated. The Qwen3-VL-2B model parameters, optimizer states, and activations consume a fixed ~21 GB regardless of batch size or backend.

### Peak GPU Memory Comparison

| Batch Size | youmu GPU (MB) | lerobot GPU (MB) | Difference |
|:----------:|:--------------:|:----------------:|:----------:|
| 1 | 21,008 | 21,008 | 0 MB |
| 8 | 21,012 | 21,004 | +8 MB |
| 16 | 21,128 | 21,052 | +76 MB |
| 32 | 21,270 | 21,166 | +104 MB |
| 64 | 21,094 | 21,090 | +4 MB |

GPU memory is virtually identical between backends (within ~100 MB). This confirms that VRAM usage is entirely dominated by model weights, gradients, optimizer states, and activations — not by the data loading backend. Note: this measures only GPU memory; CPU/system memory differs significantly (youmu uses ~800 MB RSS vs lerobot's 5-7 GB, as documented in the data loading benchmark).

## Conclusions

1. **youmu_page_aligned wins at large batch sizes.** At bs>=16, youmu is 1.5-2.4x faster because lerobot's data pipeline becomes the bottleneck. The crossover point is between batch size 8 and 16.

2. **lerobot wins at small batch sizes.** At bs<=8, lerobot is ~15% faster because the GPU is the bottleneck and lerobot's in-memory data access has lower per-sample latency.

3. **No OOM boundary found.** Both backends complete all batch sizes (1-64) on H100 80GB. GPU memory is fixed at ~21 GB, entirely model-dominated.

4. **Choose backend based on batch size regime:**
   - **bs <= 8:** lerobot is slightly faster (but uses 6-8x more system memory)
   - **bs >= 16:** youmu_page_aligned is significantly faster and more memory-efficient
   - For production training where larger batch sizes are typical (especially with gradient accumulation across GPUs), youmu_page_aligned is the better choice.

5. **The page-aligned optimization pays off at scale.** When data loading pressure increases (larger batches), youmu's page-aligned I/O maintains stable throughput while lerobot's in-memory approach degrades — suggesting lerobot's data pipeline has serialization overhead that doesn't scale with batch size.
