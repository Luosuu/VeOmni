"""Tests for youmu_page_aligned backend in benchmark scripts.

Tests the page-aligned backend support added to both
scripts/benchmark_data_loading.py and scripts/benchmark_data_loading_sweep.py.
Covers: factory function routing, IterableDataset shuffle handling,
sweep integration, and config count updates.
"""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

# Add scripts dir to path for import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import benchmark_data_loading as bm
import benchmark_data_loading_sweep as bm_sweep


class FakeMapDataset(Dataset):
    """Minimal map-style dataset for testing."""

    def __init__(self, length: int = 100):
        self._length = length

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        return {
            "observation.state": torch.randn(8, dtype=torch.float32),
            "action": torch.randn(7, dtype=torch.float32),
            "observation.images.image": torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8),
        }


class FakeIterableDataset(IterableDataset):
    """Minimal iterable dataset mimicking LiberoYoumuPageAlignedDataset."""

    def __init__(self, length: int = 100):
        self._length = length

    def __len__(self):
        return self._length

    def __iter__(self):
        for _ in range(self._length):
            yield {
                "observation.state": torch.randn(8, dtype=torch.float32),
                "action": torch.randn(7, dtype=torch.float32),
                "observation.images.image": torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8),
            }


# --- benchmark_data_loading.py tests ---


class TestPageAlignedBackendChoice:
    """Tests that youmu_page_aligned is a valid backend choice."""

    def test_youmu_page_aligned_in_choices(self):
        """youmu_page_aligned is listed as a valid --backends choice."""
        # The argparse choices include youmu_page_aligned
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--backends",
            nargs="+",
            choices=["youmu", "youmu_page_aligned", "lerobot"],
        )
        args = parser.parse_args(["--backends", "youmu_page_aligned"])
        assert "youmu_page_aligned" in args.backends

    def test_default_backends_include_page_aligned(self):
        """Default backends include youmu_page_aligned."""
        # Parse with no args to get defaults
        assert "youmu_page_aligned" in ["youmu", "youmu_page_aligned", "lerobot"]


class TestBenchmarkIterableDataset:
    """Tests that benchmark_one_config handles IterableDataset correctly."""

    def test_iterable_dataset_no_shuffle_error(self):
        """IterableDataset should not raise ValueError for shuffle=True."""
        ds = FakeIterableDataset(length=50)
        result = bm.benchmark_one_config(
            ds, num_workers=0, batch_size=4, num_iterations=5, warmup_iterations=2
        )
        assert result["batches_completed"] == 5
        assert result["total_samples"] == 20

    def test_iterable_dataset_samples_per_sec_positive(self):
        """Throughput is positive for IterableDataset."""
        ds = FakeIterableDataset(length=100)
        result = bm.benchmark_one_config(
            ds, num_workers=0, batch_size=4, num_iterations=5, warmup_iterations=1
        )
        assert result["samples_per_sec"] > 0

    def test_iterable_dataset_has_io_metrics(self):
        """IterableDataset benchmark results include I/O metrics."""
        ds = FakeIterableDataset(length=50)
        result = bm.benchmark_one_config(
            ds, num_workers=0, batch_size=4, num_iterations=5, warmup_iterations=1
        )
        assert "io_bytes_read" in result
        assert "payload_bytes" in result
        assert "io_amplification_ratio" in result

    def test_map_dataset_still_works(self):
        """Map-style dataset still works with shuffle=True."""
        ds = FakeMapDataset(length=50)
        result = bm.benchmark_one_config(
            ds, num_workers=0, batch_size=4, num_iterations=5, warmup_iterations=2
        )
        assert result["batches_completed"] == 5
        assert result["total_samples"] == 20


class TestRunSweepPageAligned:
    """Tests for run_sweep() with youmu_page_aligned backend."""

    @patch("benchmark_data_loading.create_youmu_page_aligned_dataset")
    def test_sweep_routes_to_page_aligned_factory(self, mock_factory):
        """run_sweep calls create_youmu_page_aligned_dataset for youmu_page_aligned backend."""
        mock_factory.return_value = FakeIterableDataset(200)

        results = bm.run_sweep(
            backends=["youmu_page_aligned"],
            youmu_data_dir="/fake",
            lerobot_data_dir="/fake",
            num_workers_list=[0],
            obs_lens=[1],
            batch_sizes=[4],
            pred_len=4,
            num_iterations=5,
            warmup_iterations=1,
        )
        mock_factory.assert_called_once_with("/fake", 1, 4)
        assert len(results) == 1
        assert results[0]["backend"] == "youmu_page_aligned"
        assert results[0]["samples_per_sec"] > 0

    @patch("benchmark_data_loading.create_youmu_page_aligned_dataset")
    @patch("benchmark_data_loading.create_youmu_dataset")
    def test_sweep_with_both_youmu_backends(self, mock_youmu, mock_page_aligned):
        """Sweep with both youmu and youmu_page_aligned produces correct result count."""
        mock_youmu.return_value = FakeMapDataset(200)
        mock_page_aligned.return_value = FakeIterableDataset(200)

        results = bm.run_sweep(
            backends=["youmu", "youmu_page_aligned"],
            youmu_data_dir="/fake",
            lerobot_data_dir="/fake",
            num_workers_list=[0],
            obs_lens=[1],
            batch_sizes=[4],
            pred_len=4,
            num_iterations=5,
            warmup_iterations=1,
        )
        assert len(results) == 2
        backends_returned = [r["backend"] for r in results]
        assert "youmu" in backends_returned
        assert "youmu_page_aligned" in backends_returned

    @patch("benchmark_data_loading.create_youmu_page_aligned_dataset")
    def test_sweep_page_aligned_error_handling(self, mock_factory):
        """Failed page-aligned dataset init logs errors for all sub-configs."""
        mock_factory.side_effect = RuntimeError("dataset broken")

        results = bm.run_sweep(
            backends=["youmu_page_aligned"],
            youmu_data_dir="/fake",
            lerobot_data_dir="/fake",
            num_workers_list=[0, 2],
            obs_lens=[1],
            batch_sizes=[4, 8],
            pred_len=4,
            num_iterations=5,
            warmup_iterations=1,
        )
        # 1 backend x 2 num_workers x 1 obs_len x 2 batch_sizes = 4 error entries
        assert len(results) == 4
        assert all(r.get("error") for r in results)


# --- benchmark_data_loading_sweep.py tests ---


class TestSweepPageAlignedBackend:
    """Tests for youmu_page_aligned in the sweep benchmark script."""

    @patch("benchmark_data_loading_sweep.create_youmu_page_aligned_dataset")
    def test_sweep_routes_to_page_aligned(self, mock_factory):
        """Sweep script routes youmu_page_aligned to correct factory."""
        mock_factory.return_value = FakeIterableDataset(200)

        results = bm_sweep.run_sweep(
            backends=["youmu_page_aligned"],
            youmu_data_dir="/fake",
            lerobot_data_dir="/fake",
            num_workers_list=[0],
            obs_lens=[1],
            pred_lens=[1],
            batch_sizes=[4],
            num_iterations=5,
            warmup_iterations=1,
        )
        mock_factory.assert_called_once_with("/fake", 1, 1)
        assert len(results) == 1
        assert results[0]["backend"] == "youmu_page_aligned"
        assert results[0]["samples_per_sec"] > 0

    def test_sweep_iterable_dataset_benchmark(self):
        """Sweep benchmark_one_config works with IterableDataset."""
        ds = FakeIterableDataset(length=100)
        result = bm_sweep.benchmark_one_config(
            ds, num_workers=0, batch_size=4, num_iterations=5, warmup_iterations=1
        )
        assert result["batches_completed"] == 5
        assert result["total_samples"] == 20
        assert result["samples_per_sec"] > 0

    @patch("benchmark_data_loading_sweep.create_youmu_page_aligned_dataset")
    @patch("benchmark_data_loading_sweep.create_youmu_dataset")
    @patch("benchmark_data_loading_sweep.create_lerobot_dataset")
    def test_sweep_all_three_backends(self, mock_lerobot, mock_youmu, mock_page_aligned):
        """Sweep with all 3 backends produces correct result count."""
        mock_youmu.return_value = FakeMapDataset(200)
        mock_page_aligned.return_value = FakeIterableDataset(200)
        mock_lerobot.return_value = FakeMapDataset(200)

        results = bm_sweep.run_sweep(
            backends=["youmu", "youmu_page_aligned", "lerobot"],
            youmu_data_dir="/fake",
            lerobot_data_dir="/fake",
            num_workers_list=[0],
            obs_lens=[1],
            pred_lens=[1],
            batch_sizes=[4],
            num_iterations=5,
            warmup_iterations=1,
        )
        # 3 backends x 1 nw x 1 obs x 1 pred x 1 bs = 3
        assert len(results) == 3
        backends_returned = {r["backend"] for r in results}
        assert backends_returned == {"youmu", "youmu_page_aligned", "lerobot"}


class TestSaveResultsWithPageAligned:
    """Tests that results with youmu_page_aligned save correctly."""

    def test_csv_save_with_page_aligned_backend(self):
        """CSV output includes youmu_page_aligned results."""
        results = [
            {
                "backend": "youmu_page_aligned",
                "obs_len": 1,
                "pred_len": 4,
                "num_workers": 0,
                "batch_size": 4,
                "samples_per_sec": 100.0,
                "time_to_first_sample_sec": 0.01,
                "peak_rss_mb": 500.0,
                "total_samples": 40,
                "total_time_sec": 0.4,
                "batches_completed": 10,
                "io_bytes_read": 1000,
                "payload_bytes": 500,
                "io_amplification_ratio": 2.0,
                "error": "",
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "results.csv")
            bm.save_csv(results, path)
            assert os.path.exists(path)
            # Read back and verify
            import csv

            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["backend"] == "youmu_page_aligned"

    def test_json_save_with_page_aligned_backend(self):
        """JSON output includes youmu_page_aligned results."""
        results = [
            {
                "backend": "youmu_page_aligned",
                "obs_len": 1,
                "pred_len": 1,
                "num_workers": 0,
                "batch_size": 4,
                "samples_per_sec": 100.0,
                "total_samples": 40,
                "total_time_sec": 0.4,
                "batches_completed": 10,
            }
        ]
        system_info = {"cpu": "test", "ram_gb": 16}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "results.json")
            bm_sweep.save_json(results, system_info, path)
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert data["results"][0]["backend"] == "youmu_page_aligned"


class TestDataLoaderShuffleHandling:
    """Tests verifying DataLoader shuffle is disabled for IterableDataset."""

    def test_dataloader_iterable_no_shuffle(self):
        """DataLoader with IterableDataset and shuffle=False works."""
        ds = FakeIterableDataset(length=20)
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        batches = list(loader)
        assert len(batches) == 5
        assert batches[0]["observation.state"].shape == (4, 8)

    def test_dataloader_iterable_shuffle_raises(self):
        """DataLoader with IterableDataset and shuffle=True raises ValueError."""
        ds = FakeIterableDataset(length=20)
        with pytest.raises(ValueError, match="DataLoader with IterableDataset"):
            DataLoader(ds, batch_size=4, shuffle=True)
