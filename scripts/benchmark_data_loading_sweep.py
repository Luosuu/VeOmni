"""Data loading throughput benchmark sweep: Youmu vs Youmu Page-Aligned vs LeRobot.

Measures pure data loading throughput (samples/sec) across a comprehensive
parameter sweep including pred_len. No GPU required.

Usage:
    . .venv/bin/activate
    python scripts/benchmark_data_loading_sweep.py              # full sweep
    python scripts/benchmark_data_loading_sweep.py --quick      # smoke test (1 config per backend)
    python scripts/benchmark_data_loading_sweep.py --backends youmu youmu_page_aligned lerobot
"""

import argparse
import gc
import json
import os
import platform
import time

import psutil
import torch
from torch.utils.data import DataLoader, IterableDataset

# Default dataset paths
YOUMU_DATA_DIR = "/mnt/local/localcache00/libero_64KB"
LEROBOT_DATA_DIR = "/mnt/local/localcache00/hf_libero"

# Full sweep parameters
ALL_NUM_WORKERS = [0, 1, 2, 4, 8]
ALL_OBS_LENS = [1, 2, 4]
ALL_PRED_LENS = [1, 2, 4]
ALL_BATCH_SIZES = [1, 4, 8, 16]

# Quick mode subset
QUICK_NUM_WORKERS = [0]
QUICK_OBS_LENS = [1]
QUICK_PRED_LENS = [1]
QUICK_BATCH_SIZES = [4]

NUM_ITERATIONS = 200
WARMUP_ITERATIONS = 10


def get_system_info(youmu_data_dir: str, lerobot_data_dir: str) -> dict:
    """Collect system information (CPU, RAM, dataset paths).

    Args:
        youmu_data_dir: Path to Youmu dataset.
        lerobot_data_dir: Path to LeRobot dataset.

    Returns:
        Dict with system info fields.
    """
    return {
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "youmu_data_dir": youmu_data_dir,
        "lerobot_data_dir": lerobot_data_dir,
    }


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


def create_youmu_page_aligned_dataset(data_dir: str, obs_len: int, pred_len: int):
    """Create a LiberoYoumuPageAlignedDataset instance.

    Args:
        data_dir: Path to 64KB page-size parquet dataset.
        obs_len: Number of observation frames.
        pred_len: Number of prediction frames.

    Returns:
        LiberoYoumuPageAlignedDataset instance (IterableDataset).
    """
    from youmu.libero_dataset import LiberoYoumuPageAlignedDataset

    return LiberoYoumuPageAlignedDataset(
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
    from pathlib import Path

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


def benchmark_one_config(
    dataset,
    num_workers: int,
    batch_size: int,
    num_iterations: int,
    warmup_iterations: int,
) -> dict:
    """Benchmark a single DataLoader configuration.

    Runs warmup_iterations batches (excluded from timing), then times
    num_iterations batches. Measures samples/sec.

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

    # IterableDataset handles shuffling internally; map-style datasets use DataLoader shuffle
    is_iterable = isinstance(dataset, IterableDataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=not is_iterable,
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
    total_samples = 0
    start = time.perf_counter()

    batch_count = 0
    for batch in loader:
        # Count batch size from first tensor
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                total_samples += v.shape[0]
                break
        else:
            total_samples += batch_size

        batch_count += 1
        if batch_count >= num_iterations:
            break

    total_time = time.perf_counter() - start

    # Cleanup persistent workers
    del loader
    gc.collect()

    return {
        "total_samples": total_samples,
        "total_time_sec": round(total_time, 4),
        "samples_per_sec": round(total_samples / total_time, 2) if total_time > 0 else 0,
        "batches_completed": batch_count,
    }


def run_sweep(
    backends: list[str],
    youmu_data_dir: str,
    lerobot_data_dir: str,
    num_workers_list: list[int],
    obs_lens: list[int],
    pred_lens: list[int],
    batch_sizes: list[int],
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
        pred_lens: List of pred_len values to sweep.
        batch_sizes: List of batch_size values to sweep.
        num_iterations: Number of timed iterations per config.
        warmup_iterations: Number of warmup iterations per config.

    Returns:
        List of result dicts, one per configuration.
    """
    total_configs = (
        len(backends) * len(num_workers_list) * len(obs_lens) * len(pred_lens) * len(batch_sizes)
    )
    print(f"Running {total_configs} configurations...")
    print()

    results = []
    config_idx = 0

    for backend in backends:
        for obs_len in obs_lens:
            for pred_len in pred_lens:
                # Create dataset once per (backend, obs_len, pred_len) triple
                print(
                    f"Initializing {backend} dataset (obs_len={obs_len}, pred_len={pred_len})... ",
                    end="",
                    flush=True,
                )
                t0 = time.perf_counter()
                try:
                    if backend == "youmu":
                        dataset = create_youmu_dataset(youmu_data_dir, obs_len, pred_len)
                    elif backend == "youmu_page_aligned":
                        dataset = create_youmu_page_aligned_dataset(youmu_data_dir, obs_len, pred_len)
                    else:
                        dataset = create_lerobot_dataset(lerobot_data_dir, obs_len, pred_len)
                    init_time = time.perf_counter() - t0
                    dataset_len = len(dataset)
                    print(f"done ({init_time:.2f}s, {dataset_len} samples)")
                except Exception as e:
                    init_time = time.perf_counter() - t0
                    print(f"FAILED: {e}")
                    # Log error for all configs with this backend/obs_len/pred_len
                    for nw in num_workers_list:
                        for bs in batch_sizes:
                            config_idx += 1
                            results.append(
                                {
                                    "backend": backend,
                                    "obs_len": obs_len,
                                    "pred_len": pred_len,
                                    "num_workers": nw,
                                    "batch_size": bs,
                                    "error": str(e),
                                }
                            )
                    continue

                for nw in num_workers_list:
                    for bs in batch_sizes:
                        config_idx += 1
                        label = (
                            f"[{config_idx}/{total_configs}] {backend} "
                            f"obs={obs_len} pred={pred_len} workers={nw} bs={bs}"
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
                                "total_samples": metrics["total_samples"],
                                "total_time_sec": metrics["total_time_sec"],
                                "batches_completed": metrics["batches_completed"],
                            }
                            print(f"{metrics['samples_per_sec']:.2f} samples/s")
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

                # Free dataset between config changes
                del dataset
                gc.collect()
                print()

    return results


def save_json(results: list[dict], system_info: dict, output_path: str):
    """Save benchmark results to JSON.

    Args:
        results: List of result dicts.
        system_info: System information dict.
        output_path: Path to output JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output = {
        "system_info": system_info,
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")


def main():
    """Entry point: parse args and run benchmark sweep."""
    parser = argparse.ArgumentParser(
        description="Benchmark data loading throughput sweep: Youmu vs Youmu Page-Aligned vs LeRobot"
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
        default="benchmarks/data_loading_sweep_results.json",
        help="Path to output JSON file",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick smoke test (num_workers=0, obs_len=1, pred_len=1, batch_size=4)",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["youmu", "youmu_page_aligned", "lerobot"],
        choices=["youmu", "youmu_page_aligned", "lerobot"],
        help="Backends to benchmark",
    )
    args = parser.parse_args()

    if args.quick:
        num_workers_list = QUICK_NUM_WORKERS
        obs_lens = QUICK_OBS_LENS
        pred_lens = QUICK_PRED_LENS
        batch_sizes = QUICK_BATCH_SIZES
        num_iterations = 20
        warmup_iterations = 2
    else:
        num_workers_list = ALL_NUM_WORKERS
        obs_lens = ALL_OBS_LENS
        pred_lens = ALL_PRED_LENS
        batch_sizes = ALL_BATCH_SIZES
        num_iterations = NUM_ITERATIONS
        warmup_iterations = WARMUP_ITERATIONS

    system_info = get_system_info(args.youmu_data_dir, args.lerobot_data_dir)

    print("=" * 60)
    print("Data Loading Benchmark Sweep: Youmu vs Youmu Page-Aligned vs LeRobot")
    print("=" * 60)
    print(f"  Mode:         {'quick' if args.quick else 'full'}")
    print(f"  Backends:     {args.backends}")
    print(f"  num_workers:  {num_workers_list}")
    print(f"  obs_lens:     {obs_lens}")
    print(f"  pred_lens:    {pred_lens}")
    print(f"  batch_sizes:  {batch_sizes}")
    print(f"  iterations:   {num_iterations} (warmup={warmup_iterations})")
    print(f"  Youmu data:   {args.youmu_data_dir}")
    print(f"  LeRobot data: {args.lerobot_data_dir}")
    print(f"  Output JSON:  {args.output}")
    print()

    results = run_sweep(
        backends=args.backends,
        youmu_data_dir=args.youmu_data_dir,
        lerobot_data_dir=args.lerobot_data_dir,
        num_workers_list=num_workers_list,
        obs_lens=obs_lens,
        pred_lens=pred_lens,
        batch_sizes=batch_sizes,
        num_iterations=num_iterations,
        warmup_iterations=warmup_iterations,
    )

    save_json(results, system_info, args.output)


if __name__ == "__main__":
    main()
