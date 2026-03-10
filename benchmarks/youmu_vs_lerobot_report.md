# Youmu vs LeRobot Benchmark Report

## System Information

| Component | Details |
|-----------|---------|
| GPU | NVIDIA H100 80GB HBM3 |
| CPU | Intel Xeon Platinum 8468 |
| RAM | 196 GB |
| Dataset | LIBERO (33GB, 377 parquet files, ~267K frames) |
| Model | Qwen3-VL-2B-Instruct + action head (Qwen3VLForConditionalGenerationAction) |
| Precision | bfloat16 |
| Framework | PyTorch + VeOmni |

## 1. Data Loading Throughput

Pure data loading benchmark via `torch.utils.data.DataLoader` — no GPU, no model forward/backward.

**Configuration:** 200 iterations, 10 warmup, pred_len=4

### Best Configurations

| Backend | Best samples/sec | Config | Peak RSS |
|---------|-----------------|--------|----------|
| LeRobot | **222.74** | obs=1, workers=8, bs=8 | 5,209 MB |
| Youmu | **102.64** | obs=1, workers=8, bs=16 | 808 MB |

### Data Loading: obs_len=1

| workers | bs | Youmu samp/s | LeRobot samp/s | Ratio (L/Y) | Youmu RSS | LeRobot RSS |
|---------|-----|-------------|----------------|-------------|-----------|-------------|
| 0 | 1 | 17.60 | 34.83 | 1.98x | 757 MB | 1,722 MB |
| 0 | 4 | 16.02 | 34.99 | 2.18x | 791 MB | 2,175 MB |
| 0 | 8 | 15.23 | 34.99 | 2.30x | 797 MB | 3,242 MB |
| 0 | 16 | 15.24 | 34.89 | 2.29x | 808 MB | 5,206 MB |
| 2 | 4 | 29.76 | 62.46 | 2.10x | 808 MB | 5,209 MB |
| 4 | 4 | 56.57 | 118.11 | 2.09x | 808 MB | 5,209 MB |
| 8 | 4 | 96.34 | 218.11 | 2.26x | 808 MB | 5,209 MB |
| 8 | 8 | 102.52 | 222.74 | 2.17x | 808 MB | 5,209 MB |
| 8 | 16 | 102.64 | 222.17 | 2.16x | 808 MB | 5,209 MB |

### Data Loading: obs_len=4

| workers | bs | Youmu samp/s | LeRobot samp/s | Ratio (L/Y) | Youmu RSS | LeRobot RSS |
|---------|-----|-------------|----------------|-------------|-----------|-------------|
| 0 | 4 | 14.09 | 19.73 | 1.40x | 819 MB | 5,903 MB |
| 4 | 4 | 51.08 | 66.24 | 1.30x | 842 MB | 7,200 MB |
| 8 | 4 | 94.22 | 122.27 | 1.30x | 842 MB | 7,200 MB |
| 8 | 16 | 95.11 | 125.39 | 1.32x | 842 MB | 7,200 MB |

### Scaling Analysis: Data Loading

**num_workers scaling (obs=1, bs=4):**

| workers | Youmu samp/s | LeRobot samp/s |
|---------|-------------|----------------|
| 0 | 16.02 | 34.99 |
| 1 | 15.74 | 31.50 |
| 2 | 29.76 | 62.46 |
| 4 | 56.57 | 118.11 |
| 8 | 96.34 | 218.11 |

Both backends scale near-linearly with num_workers. LeRobot maintains a ~2x advantage at every worker count.

**obs_len impact (workers=8, bs=4):**

| obs_len | Youmu samp/s | LeRobot samp/s | Ratio |
|---------|-------------|----------------|-------|
| 1 | 96.34 | 218.11 | 2.26x |
| 2 | 100.72 | 172.33 | 1.71x |
| 4 | 94.22 | 122.27 | 1.30x |

LeRobot's advantage narrows as obs_len increases — the per-sample cost grows, and LeRobot's in-memory advantage matters less.

**batch_size impact (workers=8, obs=1):**

| bs | Youmu samp/s | LeRobot samp/s |
|----|-------------|----------------|
| 1 | 61.54 | 196.36 |
| 4 | 96.34 | 218.11 |
| 8 | 102.52 | 222.74 |
| 16 | 102.64 | 222.17 |

Throughput saturates at bs=4-8 for both backends.

### Memory Comparison

| Config | Youmu RSS | LeRobot RSS | Ratio |
|--------|-----------|-------------|-------|
| obs=1, bs=1 | 757 MB | 1,722 MB | 2.3x |
| obs=1, bs=16 | 808 MB | 5,209 MB | 6.4x |
| obs=4, bs=16 | 842 MB | 7,200 MB | 8.6x |

Youmu memory is nearly flat (~800-842 MB) regardless of batch_size or obs_len. LeRobot grows significantly with larger batches because it loads entire dataset columns into memory.

---

## 2. End-to-End Training Throughput

Full training loop: data loading + transform + forward + backward + optimizer step on a single H100.

**Configuration:** 200 steps, 10 warmup, pred_len=4, lr=1e-4, AdamW, bf16

### Best Configurations

| Backend | Best steps/sec | Config | Best samp/sec | Config |
|---------|---------------|--------|---------------|--------|
| LeRobot | **6.33** | obs=1, bs=2, w=4 | **47.48** | obs=1, bs=8, w=2 |
| Youmu | **5.77** | obs=1, bs=2, w=2 | **42.71** | obs=1, bs=8, w=4 |

### E2E Training: Full Results

| obs | bs | workers | Youmu step/s | LeRobot step/s | Ratio | Youmu samp/s | LeRobot samp/s | GPU Peak |
|-----|-----|---------|-------------|----------------|-------|-------------|----------------|----------|
| 1 | 2 | 0 | 4.14 | 5.17 | 1.25x | 8.28 | 10.33 | ~20.4 GB |
| 1 | 2 | 2 | 5.77 | 6.28 | 1.09x | 11.54 | 12.56 | ~20.4 GB |
| 1 | 2 | 4 | 5.77 | 6.33 | 1.10x | 11.54 | 12.65 | ~20.4 GB |
| 1 | 4 | 0 | 3.97 | 3.88 | 0.98x | 15.86 | 15.50 | ~20.4 GB |
| 1 | 4 | 2 | 5.50 | 6.19 | 1.12x | 22.02 | 24.74 | ~20.4 GB |
| 1 | 4 | 4 | 5.56 | 6.17 | 1.11x | 22.23 | 24.69 | ~20.4 GB |
| 1 | 8 | 0 | 2.88 | 2.59 | 0.90x | 23.05 | 20.71 | ~20.4 GB |
| 1 | 8 | 2 | 5.30 | 5.94 | 1.12x | 42.37 | 47.48 | ~20.4 GB |
| 1 | 8 | 4 | 5.34 | 5.90 | 1.10x | 42.71 | 47.17 | ~20.4 GB |
| 2 | 2 | 0 | 4.95 | 4.74 | 0.96x | 9.91 | 9.47 | ~20.4 GB |
| 2 | 2 | 2 | 5.67 | 6.29 | 1.11x | 11.34 | 12.58 | ~20.4 GB |
| 2 | 4 | 0 | 3.85 | 3.45 | 0.90x | 15.40 | 13.82 | ~20.4 GB |
| 2 | 4 | 2 | 5.48 | 6.17 | 1.13x | 21.93 | 24.68 | ~20.4 GB |
| 2 | 8 | 0 | 2.72 | 2.23 | 0.82x | 21.73 | 17.86 | ~20.4 GB |
| 2 | 8 | 2 | 5.28 | 5.88 | 1.11x | 42.27 | 47.01 | ~20.4 GB |
| 2 | 8 | 4 | 5.26 | 5.92 | 1.13x | 42.04 | 47.33 | ~20.4 GB |

### Scaling Analysis: E2E Training

**num_workers impact (obs=1, bs=8):**

| workers | Youmu step/s | LeRobot step/s | Ratio |
|---------|-------------|----------------|-------|
| 0 | 2.88 | 2.59 | 0.90x |
| 2 | 5.30 | 5.94 | 1.12x |
| 4 | 5.34 | 5.90 | 1.10x |

At workers=0 (data-loading bound), Youmu is actually faster because its lower memory overhead leads to faster per-sample reads. With workers>=2, GPU compute dominates and LeRobot's faster data loading gives it a ~10-12% edge.

**batch_size impact (obs=1, workers=2):**

| bs | Youmu step/s | Youmu samp/s | LeRobot step/s | LeRobot samp/s |
|----|-------------|-------------|----------------|----------------|
| 2 | 5.77 | 11.54 | 6.28 | 12.56 |
| 4 | 5.50 | 22.02 | 6.19 | 24.74 |
| 8 | 5.30 | 42.37 | 5.94 | 47.48 |

Steps/sec decreases with larger batches (more compute per step), but samples/sec increases because each step processes more samples.

---

## 3. Conclusions

### Key Takeaways

1. **Data loading: LeRobot is ~2x faster but uses 6-8x more memory.** LeRobot loads dataset columns into memory (Arrow-backed), giving it faster random access at the cost of 5-7 GB RSS vs Youmu's ~800 MB. This makes Youmu better suited for memory-constrained environments or larger datasets.

2. **E2E training: the gap narrows to ~10-12%.** When GPU forward/backward dominates the step time, the 2x data loading difference shrinks to only 10-12% faster training for LeRobot. At workers=0, Youmu is actually competitive or faster.

3. **2 workers is the sweet spot.** Both backends gain significantly going from 0 to 2 workers. Going from 2 to 4 workers yields <1% additional throughput, confirming the bottleneck shifts to GPU compute.

4. **No OOM at bs=8 on H100.** The Qwen3-VL-2B model uses ~20.4 GB VRAM regardless of backend or batch size, well within the H100's 80 GB capacity.

5. **Youmu's memory efficiency is its main advantage.** Youmu uses a constant ~800 MB RSS regardless of configuration, while LeRobot can use up to 7.2 GB. For multi-GPU training or larger models where system memory is shared, this difference matters.

### Recommendations

- **Use Youmu** when memory is constrained, datasets are very large, or you need predictable memory usage.
- **Use LeRobot** when maximizing raw throughput is the priority and memory is abundant.
- **Always use at least 2 DataLoader workers** — the jump from 0 to 2 workers is the single biggest performance improvement for both backends.
- **Batch size 4-8** offers the best throughput-per-sample in e2e training without excessive step time.
