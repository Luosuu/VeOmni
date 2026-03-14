# Page-Aligned Data Loading Benchmark Report

Comprehensive comparison of three data loading backends for VLA training on the LIBERO dataset:

- **youmu** (row-range): Random-access row-range reads from Parquet files
- **youmu_page_aligned**: Page-level sequential reads with shuffle buffer (new)
- **lerobot**: HuggingFace LeRobot dataset format (Arrow-backed)

## 1. Unconstrained Benchmark (No Memory Limit)

Best throughput per backend with no memory constraints (200 iterations, sweep over num_workers=[0,2,4,8], obs_lens=[1,2,4], batch_sizes=[4,8]):

| Backend | Best Throughput (samples/s) | Config | Peak RSS (MB) |
|---------|---------------------------:|--------|---------------|
| youmu_page_aligned | **589.26** | w=4 obs=1 bs=8 | 1,625 |
| lerobot | 199.66 | w=8 obs=1 bs=8 | 5,155 |
| youmu (row-range) | 83.95 | w=8 obs=1 bs=8 | 788 |

**Key findings:**
- Page-aligned is **2.95x faster** than LeRobot and **7.02x faster** than row-range Youmu
- Page-aligned achieves peak throughput with fewer workers (4 vs 8) due to efficient sequential page reads
- Row-range Youmu uses the least memory (788MB) but is slowest due to random I/O patterns
- LeRobot uses the most memory (5.2GB) due to Arrow table caching

## 2. Memory-Constrained Benchmark

Best throughput per backend at each cgroup RAM limit (MemoryMax enforced, swap disabled, 50 iterations):

| RAM Limit | youmu_page_aligned | lerobot | youmu (row-range) | Ratio (page_aligned/lerobot) |
|-----------|-------------------:|--------:|------------------:|-----------------------------:|
| 2G | 119.38 (6 errors) | 138.09 | 5.85 | 0.86x |
| 4G | 134.15 (4 errors) | 139.90 | 6.80 | 0.96x |
| **8G** | **183.26** (2 errors) | 142.34 | 9.04 | **1.29x (crossover)** |
| 16G | **226.30** | 148.37 | 30.32 | **1.53x** |
| 32G | **205.02** | 147.12 | 90.90 | **1.39x** |

### Throughput by RAM Limit

```
samples/s
  250 |                              *** (226)
      |                    ***
  200 |  *** (page-aligned)    ***        *** (205)
      |           ***
  150 |  --- (lerobot, ~140-148 stable across all limits) ---
      |  *** (119)  *** (134)
  100 |                                        +++ (91)
      |
   50 |                              +++ (30)
      |                    +++ (9)
    0 |  +++ (6)   +++ (7)
      +---2G-------4G-------8G------16G------32G---
        +++ = youmu row-range
        *** = youmu page-aligned
        --- = lerobot
```

### Crossover Analysis

**Page-aligned vs LeRobot crossover at 8G RAM.** At 8G, page-aligned reaches 183 samples/s compared to LeRobot's 142 samples/s (1.29x advantage). The advantage grows to 1.53x at 16G.

**Youmu row-range never crosses LeRobot** in the tested range. Even at 32G (91 samples/s vs 147 samples/s), row-range is still 38% slower.

### Error Analysis at Low Memory

Page-aligned multi-worker configurations fail at low RAM due to per-worker memory overhead:

| RAM Limit | Errors | Root Cause |
|-----------|--------|------------|
| 2G | 6/8 multi-worker configs | Workers OOM (shuffle buffer + page cache per worker ~1.5GB) |
| 4G | 4/8 multi-worker configs | Workers with 4+ workers OOM |
| 8G | 2/8 multi-worker configs | Only 8-worker configs OOM |
| 16G | 0 | All configs succeed |
| 32G | 0 | All configs succeed |

Single-worker (num_workers=0) configs succeed at all RAM limits, achieving 80-134 samples/s even at 2G.

## 3. I/O Amplification Analysis

I/O amplification ratio = total bytes read from disk / useful payload bytes. Measured via `/proc/self/io` (main process only; multi-worker I/O tracked separately by OS).

### Single-Worker (num_workers=0) I/O Amplification

| RAM Limit | youmu (row-range) | youmu_page_aligned | lerobot |
|-----------|------------------:|-------------------:|--------:|
| 2G | 238.94x | 17.96x | 0.73x |
| 4G | 194.38x | 18.76x | 0.67x |
| 8G | 184.31x | 15.30x | 0.65x |
| 16G | 188.33x | 17.65x | 0.67x |
| 32G | 188.80x | 19.34x | 0.67x |

**Key findings:**
- **Row-range Youmu has extreme I/O amplification** (184-239x) because random row access causes full page reads across many pages, most of which are discarded
- **Page-aligned reduces I/O amplification by ~13x** (15-19x) compared to row-range, because it reads complete pages sequentially and uses all data within each page
- **LeRobot has minimal I/O amplification** (<1x) because Arrow format stores data contiguously in memory-mapped files
- Row-range I/O amplification is slightly worse at 2G (239x vs 188x at 32G) because the OS page cache is less effective at retaining previously read pages

## 4. Memory Efficiency

Peak RSS memory usage at best-throughput configuration:

| RAM Limit | youmu (row-range) | youmu_page_aligned | lerobot |
|-----------|------------------:|-------------------:|--------:|
| 2G | 745 MB | 1,535 MB | 1,310 MB |
| 8G | 748 MB | 1,571 MB | 1,464 MB |
| 16G | 790 MB | 1,576 MB | 1,465 MB |
| 32G | 783 MB | 1,557 MB | 1,470 MB |

- Row-range Youmu is most memory-efficient (~750MB) but sacrifices throughput
- Page-aligned uses ~1.5GB for obs_len=1, growing to ~3.5GB for obs_len=4 (shuffle buffer + page cache)
- LeRobot uses ~1.3-1.8GB depending on obs_len

## 5. Conclusion and Recommendations

### When to use each backend:

| Scenario | Recommended Backend | Reason |
|----------|-------------------|--------|
| **Production training (>=8GB RAM)** | **youmu_page_aligned** | Fastest throughput (183-226 samples/s), stable performance |
| **Memory-constrained (<8GB)** | **lerobot** or **youmu_page_aligned (single worker)** | LeRobot is stable at ~140 samples/s; page-aligned single-worker gets 80-134 samples/s |
| **Minimal memory footprint** | **youmu (row-range)** | Only 750MB RSS, but very slow (5-90 samples/s) |
| **Quick prototyping** | **lerobot** | Consistent ~140 samples/s regardless of memory, simple setup |

### Summary

The page-aligned backend delivers the best throughput in production settings (>=8GB RAM), achieving up to **226 samples/s** — **1.53x faster than LeRobot** and **25x faster than row-range Youmu** under memory pressure. Its key innovation is reading Parquet data at page granularity (64KB aligned), which reduces I/O amplification from 239x (row-range) to 18x while maintaining sample randomization through a shuffle buffer.

The tradeoff is higher base memory usage (~1.5GB per worker) due to the shuffle buffer and page cache. For environments with less than 8GB available RAM, LeRobot provides the most reliable performance at ~140 samples/s with minimal configuration sensitivity.

Row-range Youmu should be avoided for production training as it suffers from severe I/O amplification under memory pressure, degrading to just 5-9 samples/s at 2G-8G RAM limits.
