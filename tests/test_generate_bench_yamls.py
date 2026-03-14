"""Tests for scripts/generate_bench_yamls.py."""

import tempfile
from pathlib import Path

import yaml

from scripts.generate_bench_yamls import (
    BACKEND_CONFIGS,
    MICRO_BATCH_SIZES,
    generate_all_yamls,
    generate_variant,
    load_base_yaml,
)

BASE_YAML_PATH = Path("configs/multimodal/qwen3_vl/qwen3_vl_libero_bench.yaml")


class TestLoadBaseYaml:
    """Tests for load_base_yaml."""

    def test_loads_valid_yaml(self):
        """Base YAML loads without error and has expected sections."""
        config = load_base_yaml(BASE_YAML_PATH)
        assert "model" in config
        assert "data" in config
        assert "train" in config

    def test_base_yaml_has_required_fields(self):
        """Base YAML contains all required fields for benchmark."""
        config = load_base_yaml(BASE_YAML_PATH)
        assert config["model"]["model_path"] == "/home/ubuntu/huggingface/Qwen3-VL-2B-Instruct"
        assert config["model"]["attn_implementation"] == "flash_attention_2"
        assert config["data"]["obs_len"] == 4
        assert config["data"]["pred_len"] == 2
        assert config["data"]["max_seq_len"] == 2048
        assert config["train"]["max_steps"] == 50
        assert config["train"]["use_wandb"] is False
        assert config["train"]["rmpad_with_pos_ids"] is True
        assert config["train"]["data_parallel_mode"] == "fsdp2"
        assert config["train"]["init_device"] == "meta"
        assert config["train"]["freeze_vit"] is False


class TestGenerateVariant:
    """Tests for generate_variant."""

    def test_youmu_page_aligned_variant(self):
        """youmu_page_aligned variant has correct data dir and backend."""
        base = load_base_yaml(BASE_YAML_PATH)
        variant = generate_variant(base, "youmu_page_aligned", 4)
        assert variant["data"]["libero_data_dir"] == "/home/ubuntu/dataset/hf_vla_64k"
        assert variant["data"]["libero_dataset_backend"] == "youmu_page_aligned"
        assert variant["train"]["micro_batch_size"] == 4
        assert variant["train"]["global_batch_size"] == 4
        assert variant["train"]["output_dir"] == "/tmp/bench_youmu_page_aligned_bs4"

    def test_lerobot_variant(self):
        """lerobot variant has correct data dir and backend."""
        base = load_base_yaml(BASE_YAML_PATH)
        variant = generate_variant(base, "lerobot", 16)
        assert variant["data"]["libero_data_dir"] == "/home/ubuntu/dataset/hf_vla"
        assert variant["data"]["libero_dataset_backend"] == "lerobot"
        assert variant["train"]["micro_batch_size"] == 16
        assert variant["train"]["global_batch_size"] == 16
        assert variant["train"]["output_dir"] == "/tmp/bench_lerobot_bs16"

    def test_does_not_mutate_base_config(self):
        """Generating a variant does not modify the base config."""
        base = load_base_yaml(BASE_YAML_PATH)
        original_bs = base["train"]["micro_batch_size"]
        generate_variant(base, "lerobot", 32)
        assert base["train"]["micro_batch_size"] == original_bs

    def test_preserves_non_modified_fields(self):
        """Variant preserves fields not modified by the generation."""
        base = load_base_yaml(BASE_YAML_PATH)
        variant = generate_variant(base, "youmu_page_aligned", 2)
        assert variant["train"]["max_steps"] == 50
        assert variant["train"]["freeze_vit"] is False
        assert variant["model"]["attn_implementation"] == "flash_attention_2"


class TestGenerateAllYamls:
    """Tests for generate_all_yamls."""

    def test_generates_correct_number_of_files(self):
        """Should generate len(backends) * len(batch_sizes) YAML files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            generated = generate_all_yamls(BASE_YAML_PATH, output_dir)
            expected = len(BACKEND_CONFIGS) * len(MICRO_BATCH_SIZES)
            assert len(generated) == expected

    def test_generated_filenames(self):
        """Generated files have correct naming convention."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            generated = generate_all_yamls(BASE_YAML_PATH, output_dir)
            filenames = {p.name for p in generated}
            # Spot check key filenames
            assert "bench_page_aligned_bs1.yaml" in filenames
            assert "bench_page_aligned_bs64.yaml" in filenames
            assert "bench_lerobot_bs1.yaml" in filenames
            assert "bench_lerobot_bs64.yaml" in filenames

    def test_generated_yamls_are_valid(self):
        """All generated YAML files are valid and parseable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            generated = generate_all_yamls(BASE_YAML_PATH, output_dir)
            for path in generated:
                with open(path) as f:
                    config = yaml.safe_load(f)
                assert "model" in config
                assert "data" in config
                assert "train" in config
                # global_batch_size == micro_batch_size (single GPU)
                assert config["train"]["global_batch_size"] == config["train"]["micro_batch_size"]

    def test_batch_sizes_match_constants(self):
        """Generated configs cover all specified batch sizes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            generate_all_yamls(BASE_YAML_PATH, output_dir)
            for backend_cfg in BACKEND_CONFIGS.values():
                for bs in MICRO_BATCH_SIZES:
                    filename = f"{backend_cfg['filename_prefix']}_bs{bs}.yaml"
                    assert (output_dir / filename).exists(), f"Missing: {filename}"


class TestGeneratedConfigsOnDisk:
    """Tests for the actual generated configs in configs/multimodal/qwen3_vl/bench/."""

    BENCH_DIR = Path("configs/multimodal/qwen3_vl/bench")

    def test_bench_dir_exists(self):
        """Bench directory exists with generated configs."""
        assert self.BENCH_DIR.is_dir()

    def test_expected_file_count(self):
        """Correct number of generated YAML files exist."""
        yamls = list(self.BENCH_DIR.glob("*.yaml"))
        expected = len(BACKEND_CONFIGS) * len(MICRO_BATCH_SIZES)
        assert len(yamls) == expected

    def test_page_aligned_configs_use_64k_dataset(self):
        """All page_aligned configs point to the 64k dataset."""
        for path in self.BENCH_DIR.glob("bench_page_aligned_*.yaml"):
            with open(path) as f:
                config = yaml.safe_load(f)
            assert config["data"]["libero_data_dir"] == "/home/ubuntu/dataset/hf_vla_64k"
            assert config["data"]["libero_dataset_backend"] == "youmu_page_aligned"

    def test_lerobot_configs_use_hf_vla_dataset(self):
        """All lerobot configs point to the hf_vla dataset."""
        for path in self.BENCH_DIR.glob("bench_lerobot_*.yaml"):
            with open(path) as f:
                config = yaml.safe_load(f)
            assert config["data"]["libero_data_dir"] == "/home/ubuntu/dataset/hf_vla"
            assert config["data"]["libero_dataset_backend"] == "lerobot"
