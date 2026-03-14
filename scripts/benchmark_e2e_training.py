#!/usr/bin/env python3
"""E2E training benchmark: youmu_page_aligned vs lerobot.

Iterates over generated YAML configs in configs/multimodal/qwen3_vl/bench/,
launches each as a torchrun training job, collects timing and memory metrics,
and saves results to CSV.

For each (backend, batch_size) configuration:
- Launches training via torchrun --nproc_per_node=1
- Measures wall-clock time
- Parses tqdm output for steps/sec
- Monitors peak GPU memory via nvidia-smi in a background thread
- Detects OOM and skips larger batch sizes for that backend

Usage:
    python scripts/benchmark_e2e_training.py
    python scripts/benchmark_e2e_training.py --config-dir configs/multimodal/qwen3_vl/bench
    python scripts/benchmark_e2e_training.py --warmup-steps 10 --max-steps 50
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Defaults
DEFAULT_CONFIG_DIR = "configs/multimodal/qwen3_vl/bench"
DEFAULT_OUTPUT_CSV = "benchmarks/e2e_training_results.csv"
DEFAULT_WARMUP_STEPS = 10
DEFAULT_MAX_STEPS = 50
BACKENDS = ["youmu_page_aligned", "lerobot"]
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
GPU_MEMORY_POLL_INTERVAL_SEC = 0.5
SUBPROCESS_TIMEOUT_SEC = 1800  # 30 min per config


@dataclass
class BenchmarkResult:
    """Stores the result of a single benchmark run."""

    backend: str
    micro_batch_size: int
    steps_per_sec: float = 0.0
    samples_per_sec: float = 0.0
    wall_clock_sec: float = 0.0
    peak_gpu_mem_mb: float = 0.0
    status: str = "pending"


def config_name_for(backend: str, batch_size: int) -> str:
    """Return the YAML config filename for a given backend and batch size.

    Args:
        backend: Dataset backend name (youmu_page_aligned or lerobot).
        batch_size: Micro batch size.

    Returns:
        YAML filename like bench_page_aligned_bs4.yaml.
    """
    # Config naming: youmu_page_aligned -> page_aligned, lerobot -> lerobot
    backend_short = backend.replace("youmu_", "")
    return f"bench_{backend_short}_bs{batch_size}.yaml"


def query_gpu_memory_mb() -> float:
    """Query current GPU memory usage via nvidia-smi.

    Returns:
        GPU memory used in MB for the first GPU.
    """
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    lines = result.stdout.strip().split("\n")
    return float(lines[0].strip())


class GpuMemoryMonitor:
    """Background thread that polls nvidia-smi for peak GPU memory.

    Polls GPU memory at a fixed interval and tracks the maximum observed value.
    """

    def __init__(self, poll_interval: float = GPU_MEMORY_POLL_INTERVAL_SEC):
        """Initialize the GPU memory monitor.

        Args:
            poll_interval: Seconds between nvidia-smi polls.
        """
        self._poll_interval = poll_interval
        self._peak_mb: float = 0.0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start polling GPU memory in a background thread."""
        self._peak_mb = 0.0
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        """Stop polling and return peak GPU memory in MB.

        Returns:
            Peak GPU memory observed during monitoring, in MB.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        return self._peak_mb

    def _poll_loop(self) -> None:
        """Poll GPU memory until stop is signaled."""
        while not self._stop_event.is_set():
            mem_mb = query_gpu_memory_mb()
            if mem_mb > self._peak_mb:
                self._peak_mb = mem_mb
            self._stop_event.wait(self._poll_interval)


def parse_tqdm_steps_per_sec(output: str) -> Optional[float]:
    """Parse tqdm progress bar output to extract iterations per second.

    Looks for patterns like '1.45it/s' or '2.30s/it' in tqdm output.
    Returns the rate from the last tqdm update (most accurate overall rate).

    Args:
        output: Combined stdout+stderr from training process.

    Returns:
        Steps per second, or None if not parseable.
    """
    # tqdm formats: "1.45it/s" or "2.30s/it"
    it_per_sec_matches = re.findall(r"(\d+\.?\d*)\s*it/s", output)
    if it_per_sec_matches:
        return float(it_per_sec_matches[-1])

    sec_per_it_matches = re.findall(r"(\d+\.?\d*)\s*s/it", output)
    if sec_per_it_matches:
        sec_per_it = float(sec_per_it_matches[-1])
        if sec_per_it > 0:
            return 1.0 / sec_per_it

    return None


def parse_steps_completed(output: str) -> int:
    """Parse how many steps were completed from tqdm output.

    Looks for tqdm-style "N/M" progress patterns and returns the largest
    step number seen.

    Args:
        output: Combined stdout+stderr from training process.

    Returns:
        Number of steps completed, or 0 if not parseable.
    """
    matches = re.findall(r"(\d+)/(\d+)", output)
    if matches:
        return max(int(m[0]) for m in matches)
    return 0


def is_oom(exit_code: int, output: str) -> bool:
    """Check if a training run failed due to CUDA out-of-memory.

    Args:
        exit_code: Process exit code.
        output: Combined stdout+stderr.

    Returns:
        True if the run OOMed.
    """
    if exit_code == 0:
        return False
    oom_patterns = [
        "CUDA out of memory",
        "OutOfMemoryError",
        "torch.OutOfMemoryError",
    ]
    return any(pattern in output for pattern in oom_patterns)


def run_single_benchmark(
    config_path: str,
    backend: str,
    batch_size: int,
    warmup_steps: int,
    max_steps: int,
    timeout: int = SUBPROCESS_TIMEOUT_SEC,
) -> BenchmarkResult:
    """Run a single training benchmark configuration via torchrun subprocess.

    Launches torchrun, monitors GPU memory in a background thread, and
    parses the training output for timing metrics.

    Args:
        config_path: Path to the YAML config file.
        backend: Dataset backend name.
        batch_size: Micro batch size.
        warmup_steps: Number of warmup steps to exclude from timing.
        max_steps: Total training steps.
        timeout: Subprocess timeout in seconds.

    Returns:
        BenchmarkResult with collected metrics.
    """
    result = BenchmarkResult(backend=backend, micro_batch_size=batch_size)
    print(f"\n{'='*60}")
    print(f"Running: backend={backend}, batch_size={batch_size}")
    print(f"Config:  {config_path}")
    print(f"{'='*60}")

    # Start GPU memory monitor
    gpu_monitor = GpuMemoryMonitor()
    gpu_monitor.start()

    # Launch training via torchrun
    cmd = [
        "torchrun",
        "--nproc_per_node=1",
        "tasks/omni/train_qwen_vl_libero.py",
        config_path,
    ]

    start_time = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    wall_clock = time.time() - start_time

    # Stop GPU monitor and get peak
    peak_mem_mb = gpu_monitor.stop()

    # Combine output for parsing (tqdm writes to stderr)
    combined_output = proc.stdout + "\n" + proc.stderr

    # Check for OOM
    if is_oom(proc.returncode, combined_output):
        result.status = "OOM"
        result.wall_clock_sec = round(wall_clock, 2)
        result.peak_gpu_mem_mb = round(peak_mem_mb, 1)
        print(f"  OOM at batch_size={batch_size}")
        return result

    # Check for other failures
    if proc.returncode != 0:
        result.status = f"FAILED(exit={proc.returncode})"
        result.wall_clock_sec = round(wall_clock, 2)
        result.peak_gpu_mem_mb = round(peak_mem_mb, 1)
        print(f"  FAILED with exit code {proc.returncode}")
        # Print last 20 lines for debugging
        output_lines = combined_output.strip().split("\n")
        for line in output_lines[-20:]:
            print(f"    {line}")
        return result

    # Parse timing from tqdm output
    steps_per_sec = parse_tqdm_steps_per_sec(combined_output)
    steps_completed = parse_steps_completed(combined_output)

    if steps_per_sec is None:
        # Fallback: compute from wall clock (includes warmup, less accurate)
        steps_per_sec = steps_completed / wall_clock if wall_clock > 0 else 0.0

    samples_per_sec = steps_per_sec * batch_size

    result.steps_per_sec = round(steps_per_sec, 4)
    result.samples_per_sec = round(samples_per_sec, 4)
    result.wall_clock_sec = round(wall_clock, 2)
    result.peak_gpu_mem_mb = round(peak_mem_mb, 1)
    result.status = "OK"

    print(f"  Steps completed: {steps_completed}")
    print(f"  Steps/sec:       {result.steps_per_sec}")
    print(f"  Samples/sec:     {result.samples_per_sec}")
    print(f"  Wall clock:      {result.wall_clock_sec}s")
    print(f"  Peak GPU mem:    {result.peak_gpu_mem_mb} MB")

    return result


def cleanup_output_dir(backend: str, batch_size: int) -> None:
    """Clean up the training output directory to avoid disk pressure.

    Args:
        backend: Dataset backend name.
        batch_size: Micro batch size.
    """
    output_dir = f"/tmp/bench_{backend}_bs{batch_size}"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
        print(f"  Cleaned up {output_dir}")


def save_results_csv(results: List[BenchmarkResult], output_path: str) -> None:
    """Save benchmark results to a CSV file.

    Creates parent directories if needed. CSV columns:
    backend, micro_batch_size, steps_per_sec, samples_per_sec,
    wall_clock_sec, peak_gpu_mem_mb, status.

    Args:
        results: List of benchmark results.
        output_path: Path to write the CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "backend",
        "micro_batch_size",
        "steps_per_sec",
        "samples_per_sec",
        "wall_clock_sec",
        "peak_gpu_mem_mb",
        "status",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for r in results:
            writer.writerow([
                r.backend,
                r.micro_batch_size,
                r.steps_per_sec,
                r.samples_per_sec,
                r.wall_clock_sec,
                r.peak_gpu_mem_mb,
                r.status,
            ])
    print(f"\nResults saved to {output_path}")


def run_benchmark(
    config_dir: str = DEFAULT_CONFIG_DIR,
    output_csv: str = DEFAULT_OUTPUT_CSV,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> List[BenchmarkResult]:
    """Run the full e2e training benchmark across all configurations.

    Iterates over backends and batch sizes in ascending order.
    Stops increasing batch size for a backend after OOM.

    Args:
        config_dir: Directory containing benchmark YAML configs.
        output_csv: Path to save results CSV.
        warmup_steps: Number of warmup steps to exclude.
        max_steps: Total training steps per config.

    Returns:
        List of all benchmark results.
    """
    all_results: List[BenchmarkResult] = []

    for backend in BACKENDS:
        print(f"\n{'#'*60}")
        print(f"# Backend: {backend}")
        print(f"{'#'*60}")

        oom_hit = False
        for batch_size in BATCH_SIZES:
            if oom_hit:
                # Skip larger batch sizes after OOM
                result = BenchmarkResult(
                    backend=backend,
                    micro_batch_size=batch_size,
                    status="SKIPPED(prior OOM)",
                )
                all_results.append(result)
                print(f"\n  Skipping batch_size={batch_size} (prior OOM)")
                continue

            config_name = config_name_for(backend, batch_size)
            config_path = os.path.join(config_dir, config_name)

            if not os.path.exists(config_path):
                print(f"\n  Config not found: {config_path}, skipping")
                result = BenchmarkResult(
                    backend=backend,
                    micro_batch_size=batch_size,
                    status="SKIPPED(no config)",
                )
                all_results.append(result)
                continue

            result = run_single_benchmark(
                config_path=config_path,
                backend=backend,
                batch_size=batch_size,
                warmup_steps=warmup_steps,
                max_steps=max_steps,
            )
            all_results.append(result)

            if result.status == "OOM":
                oom_hit = True

            # Clean up output dir to save disk space
            cleanup_output_dir(backend, batch_size)

    # Save results
    save_results_csv(all_results, output_csv)
    return all_results


def main() -> None:
    """Entry point: parse args and run the full benchmark suite."""
    parser = argparse.ArgumentParser(description="E2E training benchmark")
    parser.add_argument(
        "--config-dir",
        default=DEFAULT_CONFIG_DIR,
        help=f"Directory with benchmark YAML configs (default: {DEFAULT_CONFIG_DIR})",
    )
    parser.add_argument(
        "--output-csv",
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=DEFAULT_WARMUP_STEPS,
        help=f"Warmup steps to exclude from timing (default: {DEFAULT_WARMUP_STEPS})",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Total training steps (default: {DEFAULT_MAX_STEPS})",
    )
    args = parser.parse_args()

    print("E2E Training Benchmark")
    print(f"Config dir:    {args.config_dir}")
    print(f"Output CSV:    {args.output_csv}")
    print(f"Warmup steps:  {args.warmup_steps}")
    print(f"Max steps:     {args.max_steps}")

    run_benchmark(
        config_dir=args.config_dir,
        output_csv=args.output_csv,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
