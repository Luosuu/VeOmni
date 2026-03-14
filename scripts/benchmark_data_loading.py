"""Data loading throughput benchmark: Youmu vs LeRobot.

Measures pure data loading throughput (samples/sec, time-to-first-sample,
peak RSS memory) across multiple configurations of num_workers, obs_len,
and batch_size.

Usage:
    . .venv/bin/activate
    python scripts/benchmark_data_loading.py              # full sweep (120 configs)
    python scripts/benchmark_data_loading.py --quick      # smoke test (1 config per backend)
"""

import argparse
import csv
import gc
import json
import os
import resource
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Default dataset paths
YOUMU_DATA_DIR = "/home/ubuntu/dataset/hf_vla_64k"
LEROBOT_DATA_DIR = "/home/ubuntu/dataset/hf_vla"

# Sweep parameters
ALL_NUM_WORKERS = [0, 1, 2, 4, 8]
ALL_OBS_LENS = [1, 2, 4]
ALL_BATCH_SIZES = [1, 4, 8, 16]

# Quick mode subset
QUICK_NUM_WORKERS = [0]
QUICK_OBS_LENS = [1]
QUICK_BATCH_SIZES = [4]

PRED_LEN = 4
NUM_ITERATIONS = 200
WARMUP_ITERATIONS = 10


def create_youmu_dataset(data_dir: str, obs_len: int, pred_len: int):
    """Create a LiberoYoumuDataset instance.

    Args:
        data_dir: Path to 64KB page-size parquet dataset.
        obs_len: Number of observation frames.
        pred_len: Number of prediction frames.

    Returns:
        LiberoYoumuDataset instance.
    """
    from youmu.libero_dataset import LiberoYoumuDataset

    return LiberoYoumuDataset(
        data_dir=data_dir,
        obs_len=obs_len,
        pred_len=pred_len,
        state_column="observation.state",
        action_column="action",
        image_column="observation.images.image",
    )


def create_lerobot_dataset(data_dir: str, obs_len: int, pred_len: int):
    """Create a LeRobotDataset instance with delta_timestamps.

    Args:
        data_dir: Path to LeRobot-format dataset.
        obs_len: Number of observation frames.
        pred_len: Number of prediction frames.

    Returns:
        LeRobotDataset instance.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    info_path = Path(data_dir) / "meta" / "info.json"
    with open(info_path) as f:
        fps = json.load(f)["fps"]

    obs_timestamps = [-(obs_len - 1 - i) / fps for i in range(obs_len)]
    pred_timestamps = [i / fps for i in range(pred_len)]

    return LeRobotDataset(
        repo_id="HuggingFaceVLA/libero",
        root=str(Path(data_dir).resolve()),
        delta_timestamps={
            "observation.state": obs_timestamps,
            "action": pred_timestamps,
            "observation.images.image": obs_timestamps,
        },
        download_videos=False,
    )


def get_peak_rss_mb() -> float:
    """Get peak RSS memory usage in MB via resource.getrusage.

    Returns:
        Peak RSS in megabytes.
    """
    # ru_maxrss is in KB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def read_io_bytes() -> int:
    """Read cumulative bytes read from disk via /proc/self/io.

    Parses the 'read_bytes' line from /proc/self/io, which counts
    bytes fetched from the storage layer (not just page cache).

    Returns:
        Cumulative read_bytes value.
    """
    with open("/proc/self/io") as f:
        for line in f:
            if line.startswith("read_bytes:"):
                return int(line.split(":")[1].strip())
    raise RuntimeError("Could not find read_bytes in /proc/self/io")


def compute_payload_bytes(batch: dict) -> int:
    """Compute the useful payload size in bytes for a batch of tensors.

    Sums element_count * element_byte_size for every tensor in the batch.

    Args:
        batch: Dict of batch outputs from the DataLoader.

    Returns:
        Total payload bytes across all tensors in the batch.
    """
    total = 0
    for v in batch.values():
        if isinstance(v, torch.Tensor):
            total += v.nelement() * v.element_size()
    return total


def benchmark_one_config(
    dataset,
    num_workers: int,
    batch_size: int,
    num_iterations: int,
    warmup_iterations: int,
) -> dict:
    """Benchmark a single DataLoader configuration.

    Runs warmup_iterations batches (excluded from timing), then times
    num_iterations batches. Measures samples/sec, time-to-first-sample,
    and peak RSS memory.

    Args:
        dataset: PyTorch Dataset instance.
        num_workers: Number of DataLoader worker processes.
        batch_size: Batch size.
        num_iterations: Number of timed iterations (after warmup).
        warmup_iterations: Number of warmup iterations.

    Returns:
        Dict with benchmark metrics.
    """
    gc.collect()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    # Warmup phase
    warmup_done = 0
    for batch in loader:
        warmup_done += 1
        if warmup_done >= warmup_iterations:
            break

    # Timed phase
    rss_before = get_peak_rss_mb()
    io_before = read_io_bytes()
    total_samples = 0
    total_payload_bytes = 0
    time_to_first = None
    start = time.perf_counter()

    batch_count = 0
    for batch in loader:
        if time_to_first is None:
            time_to_first = time.perf_counter() - start

        # Count batch size from first tensor
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                total_samples += v.shape[0]
                break
        else:
            total_samples += batch_size

        # Accumulate payload bytes
        total_payload_bytes += compute_payload_bytes(batch)

        batch_count += 1
        if batch_count >= num_iterations:
            break

    total_time = time.perf_counter() - start
    io_after = read_io_bytes()
    rss_after = get_peak_rss_mb()

    io_bytes_read = io_after - io_before
    io_amplification = (
        round(io_bytes_read / total_payload_bytes, 4)
        if total_payload_bytes > 0
        else None
    )

    # Cleanup persistent workers
    del loader
    gc.collect()

    return {
        "total_samples": total_samples,
        "total_time_sec": round(total_time, 4),
        "samples_per_sec": round(total_samples / total_time, 2) if total_time > 0 else 0,
        "time_to_first_sample_sec": round(time_to_first, 4) if time_to_first is not None else None,
        "peak_rss_mb": round(max(rss_before, rss_after), 2),
        "batches_completed": batch_count,
        "io_bytes_read": io_bytes_read,
        "payload_bytes": total_payload_bytes,
        "io_amplification_ratio": io_amplification,
    }


def run_sweep(
    backends: list[str],
    youmu_data_dir: str,
    lerobot_data_dir: str,
    num_workers_list: list[int],
    obs_lens: list[int],
    batch_sizes: list[int],
    pred_len: int,
    num_iterations: int,
    warmup_iterations: int,
) -> list[dict]:
    """Run the full benchmark sweep across all configurations.

    Args:
        backends: List of backend names to benchmark ("youmu", "lerobot").
        youmu_data_dir: Path to Youmu-format dataset.
        lerobot_data_dir: Path to LeRobot-format dataset.
        num_workers_list: List of num_workers values to sweep.
        obs_lens: List of obs_len values to sweep.
        batch_sizes: List of batch_size values to sweep.
        pred_len: Prediction length (fixed).
        num_iterations: Number of timed iterations per config.
        warmup_iterations: Number of warmup iterations per config.

    Returns:
        List of result dicts, one per configuration.
    """
    total_configs = len(backends) * len(obs_lens) * len(num_workers_list) * len(batch_sizes)
    print(f"Running {total_configs} configurations...")
    print()

    results = []
    config_idx = 0

    for backend in backends:
        for obs_len in obs_lens:
            # Create dataset once per (backend, obs_len) pair
            print(f"Initializing {backend} dataset (obs_len={obs_len})... ", end="", flush=True)
            t0 = time.perf_counter()
            try:
                if backend == "youmu":
                    dataset = create_youmu_dataset(youmu_data_dir, obs_len, pred_len)
                else:
                    dataset = create_lerobot_dataset(lerobot_data_dir, obs_len, pred_len)
                init_time = time.perf_counter() - t0
                dataset_len = len(dataset)
                print(f"done ({init_time:.2f}s, {dataset_len} samples)")
            except Exception as e:
                init_time = time.perf_counter() - t0
                print(f"FAILED: {e}")
                # Log error for all configs with this backend/obs_len
                for nw in num_workers_list:
                    for bs in batch_sizes:
                        config_idx += 1
                        results.append({
                            "backend": backend,
                            "obs_len": obs_len,
                            "pred_len": pred_len,
                            "num_workers": nw,
                            "batch_size": bs,
                            "error": str(e),
                        })
                continue

            for nw in num_workers_list:
                for bs in batch_sizes:
                    config_idx += 1
                    label = (
                        f"[{config_idx}/{total_configs}] {backend} "
                        f"obs={obs_len} workers={nw} bs={bs}"
                    )
                    print(f"  {label} ... ", end="", flush=True)

                    try:
                        metrics = benchmark_one_config(
                            dataset, nw, bs, num_iterations, warmup_iterations
                        )
                        row = {
                            "backend": backend,
                            "obs_len": obs_len,
                            "pred_len": pred_len,
                            "num_workers": nw,
                            "batch_size": bs,
                            "samples_per_sec": metrics["samples_per_sec"],
                            "time_to_first_sample_sec": metrics["time_to_first_sample_sec"],
                            "peak_rss_mb": metrics["peak_rss_mb"],
                            "total_samples": metrics["total_samples"],
                            "total_time_sec": metrics["total_time_sec"],
                            "batches_completed": metrics["batches_completed"],
                            "io_bytes_read": metrics["io_bytes_read"],
                            "payload_bytes": metrics["payload_bytes"],
                            "io_amplification_ratio": metrics["io_amplification_ratio"],
                            "error": "",
                        }
                        io_amp_str = (
                            f"{metrics['io_amplification_ratio']:.2f}x"
                            if metrics["io_amplification_ratio"] is not None
                            else "N/A"
                        )
                        print(
                            f"{metrics['samples_per_sec']:.2f} samples/s, "
                            f"first={metrics['time_to_first_sample_sec']:.4f}s, "
                            f"rss={metrics['peak_rss_mb']:.0f}MB, "
                            f"io_amp={io_amp_str}"
                        )
                    except Exception as e:
                        row = {
                            "backend": backend,
                            "obs_len": obs_len,
                            "pred_len": pred_len,
                            "num_workers": nw,
                            "batch_size": bs,
                            "error": str(e),
                        }
                        print(f"ERROR: {e}")

                    results.append(row)

            # Free dataset between obs_len changes
            del dataset
            gc.collect()
            print()

    return results


def save_csv(results: list[dict], output_path: str):
    """Save benchmark results to CSV.

    Args:
        results: List of result dicts.
        output_path: Path to output CSV file.
    """
    fieldnames = [
        "backend",
        "obs_len",
        "pred_len",
        "num_workers",
        "batch_size",
        "samples_per_sec",
        "time_to_first_sample_sec",
        "peak_rss_mb",
        "total_samples",
        "total_time_sec",
        "batches_completed",
        "io_bytes_read",
        "payload_bytes",
        "io_amplification_ratio",
        "error",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_path}")


def print_comparison_table(results: list[dict]):
    """Print a grouped comparison table to stdout.

    Groups results by (obs_len, num_workers, batch_size) and shows
    youmu vs lerobot side by side.

    Args:
        results: List of result dicts.
    """
    # Build lookup: (backend, obs_len, num_workers, batch_size) -> row
    lookup = {}
    for r in results:
        key = (r["backend"], r.get("obs_len"), r.get("num_workers"), r.get("batch_size"))
        lookup[key] = r

    # Get unique config combos
    configs = set()
    for r in results:
        if not r.get("error"):
            configs.add((r.get("obs_len"), r.get("num_workers"), r.get("batch_size")))
    configs = sorted(configs)

    backends = sorted({r["backend"] for r in results})

    print("\n" + "=" * 100)
    print("DATA LOADING BENCHMARK: GROUPED COMPARISON")
    print("=" * 100)

    header = f"{'obs_len':>7} {'workers':>7} {'bs':>4}"
    for b in backends:
        header += f"  |  {b + ' samp/s':>14} {b + ' first(s)':>14} {b + ' rss(MB)':>12}"
    print(header)
    print("-" * 100)

    for obs_len, nw, bs in configs:
        line = f"{obs_len:>7} {nw:>7} {bs:>4}"
        for b in backends:
            r = lookup.get((b, obs_len, nw, bs))
            if r and not r.get("error"):
                sps = r.get("samples_per_sec", 0)
                ttf = r.get("time_to_first_sample_sec")
                rss = r.get("peak_rss_mb", 0)
                ttf_str = f"{ttf:.4f}" if ttf is not None else "N/A"
                line += f"  |  {sps:>14.2f} {ttf_str:>14} {rss:>12.0f}"
            elif r and r.get("error"):
                line += f"  |  {'ERROR':>14} {'':>14} {'':>12}"
            else:
                line += f"  |  {'N/A':>14} {'':>14} {'':>12}"
        print(line)

    print("=" * 100)


def main():
    """Entry point: parse args and run benchmark sweep."""
    parser = argparse.ArgumentParser(
        description="Benchmark data loading throughput: Youmu vs LeRobot"
    )
    parser.add_argument(
        "--youmu-data-dir",
        type=str,
        default=YOUMU_DATA_DIR,
        help="Path to Youmu (64KB page-size) dataset",
    )
    parser.add_argument(
        "--lerobot-data-dir",
        type=str,
        default=LEROBOT_DATA_DIR,
        help="Path to LeRobot dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/data_loading_results.csv",
        help="Path to output CSV file",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick smoke test (num_workers=0, obs_len=1, batch_size=4)",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["youmu", "lerobot"],
        choices=["youmu", "lerobot"],
        help="Backends to benchmark",
    )
    args = parser.parse_args()

    if args.quick:
        num_workers_list = QUICK_NUM_WORKERS
        obs_lens = QUICK_OBS_LENS
        batch_sizes = QUICK_BATCH_SIZES
        num_iterations = 20
        warmup_iterations = 2
    else:
        num_workers_list = ALL_NUM_WORKERS
        obs_lens = ALL_OBS_LENS
        batch_sizes = ALL_BATCH_SIZES
        num_iterations = NUM_ITERATIONS
        warmup_iterations = WARMUP_ITERATIONS

    print("=" * 60)
    print("Data Loading Benchmark: Youmu vs LeRobot")
    print("=" * 60)
    print(f"  Mode:         {'quick' if args.quick else 'full'}")
    print(f"  Backends:     {args.backends}")
    print(f"  num_workers:  {num_workers_list}")
    print(f"  obs_lens:     {obs_lens}")
    print(f"  batch_sizes:  {batch_sizes}")
    print(f"  pred_len:     {PRED_LEN}")
    print(f"  iterations:   {num_iterations} (warmup={warmup_iterations})")
    print(f"  Youmu data:   {args.youmu_data_dir}")
    print(f"  LeRobot data: {args.lerobot_data_dir}")
    print(f"  Output CSV:   {args.output}")
    print()

    results = run_sweep(
        backends=args.backends,
        youmu_data_dir=args.youmu_data_dir,
        lerobot_data_dir=args.lerobot_data_dir,
        num_workers_list=num_workers_list,
        obs_lens=obs_lens,
        batch_sizes=batch_sizes,
        pred_len=PRED_LEN,
        num_iterations=num_iterations,
        warmup_iterations=warmup_iterations,
    )

    save_csv(results, args.output)
    print_comparison_table(results)


if __name__ == "__main__":
    main()
