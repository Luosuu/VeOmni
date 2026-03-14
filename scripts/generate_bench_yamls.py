"""Generate benchmark YAML configs for e2e training benchmark.

Reads the base benchmark YAML and generates per-(backend, batch_size) variants
for comparing youmu_page_aligned vs lerobot training throughput.
"""

import argparse
from pathlib import Path

import yaml


# Backend-specific settings
BACKEND_CONFIGS = {
    "youmu_page_aligned": {
        "libero_data_dir": "/home/ubuntu/dataset/hf_vla_64k",
        "libero_dataset_backend": "youmu_page_aligned",
        "filename_prefix": "bench_page_aligned",
    },
    "lerobot": {
        "libero_data_dir": "/home/ubuntu/dataset/hf_vla",
        "libero_dataset_backend": "lerobot",
        "filename_prefix": "bench_lerobot",
    },
}

# Batch sizes to sweep
MICRO_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]


def load_base_yaml(base_yaml_path: Path) -> dict:
    """Load the base benchmark YAML config."""
    with open(base_yaml_path) as f:
        return yaml.safe_load(f)


def generate_variant(base_config: dict, backend: str, micro_batch_size: int) -> dict:
    """Generate a YAML config variant for a specific backend and batch size.

    Args:
        base_config: The base YAML config dict.
        backend: Backend name ('youmu_page_aligned' or 'lerobot').
        micro_batch_size: Micro batch size for this variant.

    Returns:
        Modified config dict for this variant.
    """
    config = yaml.safe_load(yaml.dump(base_config))  # deep copy
    backend_cfg = BACKEND_CONFIGS[backend]

    # Update data section
    config["data"]["libero_data_dir"] = backend_cfg["libero_data_dir"]
    config["data"]["libero_dataset_backend"] = backend_cfg["libero_dataset_backend"]

    # Update train section
    config["train"]["micro_batch_size"] = micro_batch_size
    config["train"]["global_batch_size"] = micro_batch_size  # single GPU
    config["train"]["output_dir"] = f"/tmp/bench_{backend}_bs{micro_batch_size}"

    return config


def generate_all_yamls(base_yaml_path: Path, output_dir: Path) -> list[Path]:
    """Generate all benchmark YAML variants.

    Args:
        base_yaml_path: Path to the base benchmark YAML.
        output_dir: Directory to write generated YAMLs.

    Returns:
        List of paths to generated YAML files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = load_base_yaml(base_yaml_path)
    generated = []

    for backend, backend_cfg in BACKEND_CONFIGS.items():
        for bs in MICRO_BATCH_SIZES:
            variant = generate_variant(base_config, backend, bs)
            filename = f"{backend_cfg['filename_prefix']}_bs{bs}.yaml"
            output_path = output_dir / filename
            with open(output_path, "w") as f:
                yaml.dump(variant, f, default_flow_style=False, sort_keys=False)
            generated.append(output_path)
            print(f"Generated: {output_path}")

    return generated


def main():
    """Entry point for generating benchmark YAML configs."""
    parser = argparse.ArgumentParser(
        description="Generate benchmark YAML configs for e2e training comparison"
    )
    parser.add_argument(
        "--base-yaml",
        type=Path,
        default=Path("configs/multimodal/qwen3_vl/qwen3_vl_libero_bench.yaml"),
        help="Path to base benchmark YAML config",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("configs/multimodal/qwen3_vl/bench"),
        help="Directory for generated YAML configs",
    )
    args = parser.parse_args()

    generated = generate_all_yamls(args.base_yaml, args.output_dir)
    print(f"\nGenerated {len(generated)} YAML configs in {args.output_dir}")


if __name__ == "__main__":
    main()
