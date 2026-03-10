"""End-to-end training throughput benchmark: Youmu vs LeRobot.

Measures training throughput (steps/sec, samples/sec, average step time,
GPU memory peak) on a single H100 GPU across multiple configurations of
batch_size, num_workers, and obs_len.

Usage:
    . .venv/bin/activate
    python scripts/benchmark_e2e_training.py              # full sweep (36 configs)
    python scripts/benchmark_e2e_training.py --quick      # smoke test (1 config per backend)
"""

import argparse
import csv
import gc
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Default paths
YOUMU_DATA_DIR = "/home/ubuntu/dataset/hf_vla_64k"
LEROBOT_DATA_DIR = "/home/ubuntu/dataset/hf_vla"
MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-VL-2B-Instruct")

# Sweep parameters
ALL_BATCH_SIZES = [2, 4, 8]
ALL_NUM_WORKERS = [0, 2, 4]
ALL_OBS_LENS = [1, 2]

# Quick mode subset
QUICK_BATCH_SIZES = [2]
QUICK_NUM_WORKERS = [0]
QUICK_OBS_LENS = [1]

PRED_LEN = 4
NUM_STEPS = 200
WARMUP_STEPS = 10
LR = 1e-4


def build_dataset(backend, data_dir, obs_len, pred_len):
    """Build a LIBERO dataset using the specified backend.

    Args:
        backend: One of "youmu" or "lerobot".
        data_dir: Root directory of the LIBERO dataset.
        obs_len: Number of observation frames.
        pred_len: Number of prediction frames.

    Returns:
        A PyTorch Dataset instance.
    """
    if backend == "youmu":
        from youmu.libero_dataset import LiberoYoumuDataset

        return LiberoYoumuDataset(
            data_dir=data_dir,
            obs_len=obs_len,
            pred_len=pred_len,
            state_column="observation.state",
            action_column="action",
            image_column="observation.images.image",
        )
    elif backend == "lerobot":
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        info_path = os.path.join(data_dir, "meta", "info.json")
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
    else:
        raise ValueError(f"Unknown backend: {backend}")


def build_model_and_processor(model_path):
    """Build Qwen3VLForConditionalGenerationAction and processor.

    Loads pretrained VLM weights then initializes action-prediction heads.
    The entire model is cast to bf16 for single-GPU training.

    Args:
        model_path: Path to the pretrained Qwen3-VL model.

    Returns:
        Tuple of (model, processor, position_id_func).
    """
    from accelerate import init_empty_weights
    from transformers import AutoConfig

    from veomni.models import build_processor, load_model_weights
    from veomni.models.transformers.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGenerationAction,
        apply_veomni_qwen3vl_patch,
    )

    apply_veomni_qwen3vl_patch()

    model_config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model_config.action_dim = 7  # LIBERO action dim
    model_config.pred_len = PRED_LEN

    with init_empty_weights():
        model = Qwen3VLForConditionalGenerationAction(model_config)

    load_model_weights(model, model_path, "cuda")
    model.to(torch.bfloat16)
    model.to("cuda")
    model.train()

    processor = build_processor(model_path)
    position_id_func = model.get_position_id_func()

    return model, processor, position_id_func


def build_transform(processor, position_id_func, meta_path, prompt_template):
    """Build the LIBERO sample transform function.

    Args:
        processor: Qwen3-VL processor.
        position_id_func: Model's 3D position ID function.
        meta_path: Path to episode metadata.
        prompt_template: Prompt template with {task} placeholder.

    Returns:
        Transform function for MappingDataset.
    """
    from veomni.data.multimodal.data_transform import (
        load_libero_task_descriptions,
        process_libero_sample_qwen3_vl,
    )

    task_descriptions = load_libero_task_descriptions(meta_path)

    return partial(
        process_libero_sample_qwen3_vl,
        processor=processor,
        position_id_func=position_id_func,
        task_descriptions=task_descriptions,
        prompt_template=prompt_template,
    )


def find_meta_path(data_dir):
    """Find the episode metadata file in a LIBERO dataset directory.

    Args:
        data_dir: Root directory of the LIBERO dataset.

    Returns:
        Path to the metadata file.
    """
    candidates = [
        os.path.join(data_dir, "meta", "episodes.jsonl"),
        os.path.join(data_dir, "meta", "episodes", "episodes.parquet"),
        os.path.join(data_dir, "meta", "episodes", "chunk-000", "file-000.parquet"),
    ]
    return next((p for p in candidates if os.path.exists(p)), candidates[-1])


def libero_collate_fn(features):
    """Collate function that unwraps the list-wrapped transform output.

    The process_libero_sample_qwen3_vl transform returns a list of one dict
    per sample. MappingDataset passes that through, so each feature in the
    batch is a list of one dict. We unwrap before passing to LiberoActionCollator.

    Args:
        features: List of list-wrapped dicts from the DataLoader.

    Returns:
        Collated batch dict.
    """
    from veomni.data import LiberoActionCollator

    collator = LiberoActionCollator()
    # Unwrap: each feature is [dict] -> dict
    unwrapped = [f[0] if isinstance(f, list) else f for f in features]
    return collator(unwrapped)


def benchmark_one_config(
    model,
    optimizer,
    dataset,
    batch_size,
    num_workers,
    num_steps,
    warmup_steps,
):
    """Benchmark a single training configuration.

    Runs warmup_steps steps (excluded from timing), then times num_steps
    steps. Measures steps/sec, samples/sec, average step time, and GPU
    memory peak.

    Args:
        model: The model to train.
        optimizer: The optimizer.
        dataset: The wrapped MappingDataset.
        collate_fn: Data collator function.
        batch_size: Batch size.
        num_workers: Number of DataLoader worker processes.
        num_steps: Number of timed steps.
        warmup_steps: Number of warmup steps.

    Returns:
        Dict with benchmark metrics.
    """
    from veomni.utils.seqlen_pos_transform_utils import prepare_fa_kwargs_from_position_ids

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        collate_fn=libero_collate_fn,
    )

    torch.cuda.reset_peak_memory_stats()

    data_iter = iter(loader)
    step = 0
    total_steps = warmup_steps + num_steps

    # Warmup + timed loop
    timing_start = None
    total_samples = 0

    while step < total_steps:
        # Get next batch, cycling dataloader if needed
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        # Compute flash attention kwargs
        (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = (
            prepare_fa_kwargs_from_position_ids(batch["position_ids"][:, 0, :])
        )
        batch.update(
            dict(
                cu_seq_lens_q=cu_seq_lens_q,
                cu_seq_lens_k=cu_seq_lens_k,
                max_length_q=max_length_q,
                max_length_k=max_length_k,
            )
        )

        # Move to GPU
        batch = {
            k: v.to("cuda", non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Forward
        output = model(**batch, use_cache=False)
        loss = output.loss

        # Backward
        loss.backward()

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        step += 1

        if step == warmup_steps:
            # Start timing after warmup
            torch.cuda.synchronize()
            timing_start = time.perf_counter()

        if step > warmup_steps:
            # Count samples in timed region
            for v in batch.values():
                if isinstance(v, torch.Tensor):
                    total_samples += v.shape[0]
                    break
            else:
                total_samples += batch_size

    torch.cuda.synchronize()
    total_time = time.perf_counter() - timing_start
    gpu_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Cleanup
    del loader, data_iter
    gc.collect()

    return {
        "timed_steps": num_steps,
        "total_samples": total_samples,
        "total_time_sec": round(total_time, 4),
        "steps_per_sec": round(num_steps / total_time, 4),
        "samples_per_sec": round(total_samples / total_time, 2),
        "avg_step_time_sec": round(total_time / num_steps, 4),
        "gpu_peak_mb": round(gpu_peak_mb, 2),
    }


def run_sweep(
    backends,
    youmu_data_dir,
    lerobot_data_dir,
    model_path,
    batch_sizes,
    num_workers_list,
    obs_lens,
    pred_len,
    num_steps,
    warmup_steps,
):
    """Run the full benchmark sweep across all configurations.

    Args:
        backends: List of backend names to benchmark.
        youmu_data_dir: Path to Youmu-format dataset.
        lerobot_data_dir: Path to LeRobot-format dataset.
        model_path: Path to the pretrained model.
        batch_sizes: List of batch_size values to sweep.
        num_workers_list: List of num_workers values to sweep.
        obs_lens: List of obs_len values to sweep.
        pred_len: Prediction length (fixed).
        num_steps: Number of timed steps per config.
        warmup_steps: Number of warmup steps per config.

    Returns:
        List of result dicts, one per configuration.
    """
    from veomni.data.dataset import MappingDataset

    total_configs = len(backends) * len(batch_sizes) * len(num_workers_list) * len(obs_lens)
    print(f"Running {total_configs} configurations...")
    print()

    results = []
    config_idx = 0

    # Build model once — reuse across all configs
    print("Building model... ", end="", flush=True)
    t0 = time.perf_counter()
    model, processor, position_id_func = build_model_and_processor(model_path)
    print(f"done ({time.perf_counter() - t0:.2f}s)")

    prompt_template = "Predict the next actions for the robot task: {task}"

    for backend in backends:
        data_dir = youmu_data_dir if backend == "youmu" else lerobot_data_dir

        for obs_len in obs_lens:
            # Build dataset + transform once per (backend, obs_len) pair
            print(f"Initializing {backend} dataset (obs_len={obs_len})... ", end="", flush=True)
            t0 = time.perf_counter()
            try:
                raw_dataset = build_dataset(backend, data_dir, obs_len, pred_len)
                meta_path = find_meta_path(data_dir)
                transform = build_transform(processor, position_id_func, meta_path, prompt_template)
                dataset = MappingDataset(data=raw_dataset, transform=transform)
                init_time = time.perf_counter() - t0
                print(f"done ({init_time:.2f}s, {len(dataset)} samples)")
            except Exception as e:
                init_time = time.perf_counter() - t0
                print(f"FAILED: {e}")
                for bs in batch_sizes:
                    for nw in num_workers_list:
                        config_idx += 1
                        results.append({
                            "backend": backend,
                            "obs_len": obs_len,
                            "pred_len": pred_len,
                            "batch_size": bs,
                            "num_workers": nw,
                            "error": str(e),
                        })
                continue

            for bs in batch_sizes:
                for nw in num_workers_list:
                    config_idx += 1
                    label = (
                        f"[{config_idx}/{total_configs}] {backend} "
                        f"obs={obs_len} bs={bs} workers={nw}"
                    )
                    print(f"  {label} ... ", end="", flush=True)

                    # Fresh optimizer per config to avoid state accumulation
                    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

                    try:
                        metrics = benchmark_one_config(
                            model=model,
                            optimizer=optimizer,
                            dataset=dataset,
                            batch_size=bs,
                            num_workers=nw,
                            num_steps=num_steps,
                            warmup_steps=warmup_steps,
                        )
                        row = {
                            "backend": backend,
                            "obs_len": obs_len,
                            "pred_len": pred_len,
                            "batch_size": bs,
                            "num_workers": nw,
                            "steps_per_sec": metrics["steps_per_sec"],
                            "samples_per_sec": metrics["samples_per_sec"],
                            "avg_step_time_sec": metrics["avg_step_time_sec"],
                            "gpu_peak_mb": metrics["gpu_peak_mb"],
                            "total_samples": metrics["total_samples"],
                            "total_time_sec": metrics["total_time_sec"],
                            "timed_steps": metrics["timed_steps"],
                            "error": "",
                        }
                        print(
                            f"{metrics['steps_per_sec']:.4f} steps/s, "
                            f"{metrics['samples_per_sec']:.2f} samp/s, "
                            f"gpu={metrics['gpu_peak_mb']:.0f}MB"
                        )
                    except torch.cuda.OutOfMemoryError:
                        row = {
                            "backend": backend,
                            "obs_len": obs_len,
                            "pred_len": pred_len,
                            "batch_size": bs,
                            "num_workers": nw,
                            "error": "OOM",
                        }
                        print("OOM")
                        torch.cuda.empty_cache()
                        gc.collect()
                    except Exception as e:
                        row = {
                            "backend": backend,
                            "obs_len": obs_len,
                            "pred_len": pred_len,
                            "batch_size": bs,
                            "num_workers": nw,
                            "error": str(e),
                        }
                        print(f"ERROR: {e}")

                    results.append(row)

                    # Clear GPU cache between configs
                    del optimizer
                    torch.cuda.empty_cache()
                    gc.collect()

            # Free dataset between obs_len changes
            del raw_dataset, dataset
            gc.collect()
            print()

    return results


def save_csv(results, output_path):
    """Save benchmark results to CSV.

    Args:
        results: List of result dicts.
        output_path: Path to output CSV file.
    """
    fieldnames = [
        "backend",
        "obs_len",
        "pred_len",
        "batch_size",
        "num_workers",
        "steps_per_sec",
        "samples_per_sec",
        "avg_step_time_sec",
        "gpu_peak_mb",
        "total_samples",
        "total_time_sec",
        "timed_steps",
        "error",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_path}")


def print_comparison_table(results):
    """Print a grouped comparison table to stdout.

    Groups results by (obs_len, batch_size, num_workers) and shows
    youmu vs lerobot side by side.

    Args:
        results: List of result dicts.
    """
    # Build lookup
    lookup = {}
    for r in results:
        key = (r["backend"], r.get("obs_len"), r.get("batch_size"), r.get("num_workers"))
        lookup[key] = r

    # Get unique config combos
    configs = set()
    for r in results:
        if not r.get("error"):
            configs.add((r.get("obs_len"), r.get("batch_size"), r.get("num_workers")))
    configs = sorted(configs)

    backends = sorted({r["backend"] for r in results})

    print("\n" + "=" * 120)
    print("E2E TRAINING BENCHMARK: GROUPED COMPARISON")
    print("=" * 120)

    header = f"{'obs_len':>7} {'bs':>4} {'workers':>7}"
    for b in backends:
        header += f"  |  {b + ' step/s':>14} {b + ' samp/s':>14} {b + ' gpu(MB)':>12}"
    print(header)
    print("-" * 120)

    for obs_len, bs, nw in configs:
        line = f"{obs_len:>7} {bs:>4} {nw:>7}"
        for b in backends:
            r = lookup.get((b, obs_len, bs, nw))
            if r and not r.get("error"):
                sps = r.get("steps_per_sec", 0)
                samp = r.get("samples_per_sec", 0)
                gpu = r.get("gpu_peak_mb", 0)
                line += f"  |  {sps:>14.4f} {samp:>14.2f} {gpu:>12.0f}"
            elif r and r.get("error") == "OOM":
                line += f"  |  {'OOM':>14} {'':>14} {'':>12}"
            elif r and r.get("error"):
                line += f"  |  {'ERROR':>14} {'':>14} {'':>12}"
            else:
                line += f"  |  {'N/A':>14} {'':>14} {'':>12}"
        print(line)

    print("=" * 120)


def main():
    """Entry point: parse args and run benchmark sweep."""
    parser = argparse.ArgumentParser(
        description="Benchmark e2e training throughput: Youmu vs LeRobot"
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
        "--model-path",
        type=str,
        default=MODEL_PATH,
        help="Path to pretrained Qwen3-VL model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/e2e_training_results.csv",
        help="Path to output CSV file",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick smoke test (batch_size=2, num_workers=0, obs_len=1)",
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
        batch_sizes = QUICK_BATCH_SIZES
        num_workers_list = QUICK_NUM_WORKERS
        obs_lens = QUICK_OBS_LENS
        num_steps = 20
        warmup_steps = 2
    else:
        batch_sizes = ALL_BATCH_SIZES
        num_workers_list = ALL_NUM_WORKERS
        obs_lens = ALL_OBS_LENS
        num_steps = NUM_STEPS
        warmup_steps = WARMUP_STEPS

    print("=" * 60)
    print("E2E Training Benchmark: Youmu vs LeRobot")
    print("=" * 60)
    print(f"  Mode:         {'quick' if args.quick else 'full'}")
    print(f"  Backends:     {args.backends}")
    print(f"  batch_sizes:  {batch_sizes}")
    print(f"  num_workers:  {num_workers_list}")
    print(f"  obs_lens:     {obs_lens}")
    print(f"  pred_len:     {PRED_LEN}")
    print(f"  steps:        {num_steps} (warmup={warmup_steps})")
    print(f"  lr:           {LR}")
    print(f"  Model:        {args.model_path}")
    print(f"  Youmu data:   {args.youmu_data_dir}")
    print(f"  LeRobot data: {args.lerobot_data_dir}")
    print(f"  Output CSV:   {args.output}")
    print()

    results = run_sweep(
        backends=args.backends,
        youmu_data_dir=args.youmu_data_dir,
        lerobot_data_dir=args.lerobot_data_dir,
        model_path=args.model_path,
        batch_sizes=batch_sizes,
        num_workers_list=num_workers_list,
        obs_lens=obs_lens,
        pred_len=PRED_LEN,
        num_steps=num_steps,
        warmup_steps=warmup_steps,
    )

    save_csv(results, args.output)
    print_comparison_table(results)


if __name__ == "__main__":
    main()
