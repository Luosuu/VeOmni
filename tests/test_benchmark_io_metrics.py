"""Tests for I/O efficiency metrics in scripts/benchmark_data_loading.py.

Unit tests for read_io_bytes, compute_payload_bytes, and integration
tests verifying the new I/O metrics appear in benchmark_one_config output.
"""

import os
import sys

import pytest
import torch
from torch.utils.data import Dataset

# Add scripts dir to path for import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import benchmark_data_loading as bm


class FakeDataset(Dataset):
    """Minimal dataset returning fixed-size tensors for benchmarking."""

    def __init__(self, length: int = 100):
        self._length = length

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        return {
            "observation.state": torch.randn(8, dtype=torch.float32),
            "action": torch.randn(7, dtype=torch.float32),
            "observation.images.image": torch.randint(
                0, 256, (3, 64, 64), dtype=torch.uint8
            ),
        }


# --- Unit tests for read_io_bytes ---


class TestReadIoBytes:
    """Tests for read_io_bytes()."""

    def test_returns_non_negative_int(self):
        """read_io_bytes should return a non-negative integer."""
        value = bm.read_io_bytes()
        assert isinstance(value, int)
        assert value >= 0

    def test_monotonically_increases_after_read(self):
        """Reading data from disk should increase the counter."""
        before = bm.read_io_bytes()
        # Force some I/O by reading a file
        with open("/proc/self/io") as f:
            _ = f.read()
        after = bm.read_io_bytes()
        assert after >= before


# --- Unit tests for compute_payload_bytes ---


class TestComputePayloadBytes:
    """Tests for compute_payload_bytes()."""

    def test_single_float32_tensor(self):
        """A single float32 tensor of 10 elements = 40 bytes."""
        batch = {"x": torch.zeros(10, dtype=torch.float32)}
        assert bm.compute_payload_bytes(batch) == 10 * 4

    def test_single_uint8_tensor(self):
        """A uint8 tensor of shape (3, 64, 64) = 12288 bytes."""
        batch = {"img": torch.zeros(3, 64, 64, dtype=torch.uint8)}
        assert bm.compute_payload_bytes(batch) == 3 * 64 * 64 * 1

    def test_multiple_tensors(self):
        """Payload sums across all tensors in batch."""
        batch = {
            "a": torch.zeros(8, dtype=torch.float32),  # 32 bytes
            "b": torch.zeros(7, dtype=torch.float32),  # 28 bytes
        }
        assert bm.compute_payload_bytes(batch) == 32 + 28

    def test_empty_batch(self):
        """Empty dict returns 0 bytes."""
        assert bm.compute_payload_bytes({}) == 0

    def test_non_tensor_values_ignored(self):
        """Non-tensor values in batch dict are ignored."""
        batch = {
            "tensor": torch.zeros(5, dtype=torch.float32),  # 20 bytes
            "metadata": "not a tensor",
            "index": 42,
        }
        assert bm.compute_payload_bytes(batch) == 20

    def test_mixed_dtypes(self):
        """Correctly handles mixed dtypes (float32 + uint8 + float64)."""
        batch = {
            "f32": torch.zeros(10, dtype=torch.float32),  # 40 bytes
            "u8": torch.zeros(10, dtype=torch.uint8),  # 10 bytes
            "f64": torch.zeros(10, dtype=torch.float64),  # 80 bytes
        }
        assert bm.compute_payload_bytes(batch) == 40 + 10 + 80


# --- Integration: benchmark_one_config returns I/O metrics ---


class TestBenchmarkOneConfigIoMetrics:
    """Integration tests for I/O metrics in benchmark_one_config output."""

    def test_io_metrics_present_in_result(self):
        """Result dict includes io_bytes_read, payload_bytes, io_amplification_ratio."""
        ds = FakeDataset(length=50)
        result = bm.benchmark_one_config(
            ds, num_workers=0, batch_size=4, num_iterations=5, warmup_iterations=1
        )
        assert "io_bytes_read" in result
        assert "payload_bytes" in result
        assert "io_amplification_ratio" in result

    def test_io_bytes_read_non_negative(self):
        """io_bytes_read should be non-negative."""
        ds = FakeDataset(length=50)
        result = bm.benchmark_one_config(
            ds, num_workers=0, batch_size=4, num_iterations=5, warmup_iterations=1
        )
        assert result["io_bytes_read"] >= 0

    def test_payload_bytes_positive(self):
        """payload_bytes should be positive when samples are loaded."""
        ds = FakeDataset(length=50)
        result = bm.benchmark_one_config(
            ds, num_workers=0, batch_size=4, num_iterations=5, warmup_iterations=1
        )
        assert result["payload_bytes"] > 0

    def test_payload_bytes_scales_with_batches(self):
        """More batches should produce proportionally more payload bytes."""
        ds = FakeDataset(length=200)
        result_5 = bm.benchmark_one_config(
            ds, num_workers=0, batch_size=4, num_iterations=5, warmup_iterations=1
        )
        result_10 = bm.benchmark_one_config(
            ds, num_workers=0, batch_size=4, num_iterations=10, warmup_iterations=1
        )
        assert result_10["payload_bytes"] == 2 * result_5["payload_bytes"]

    def test_io_amplification_ratio_is_ratio(self):
        """io_amplification_ratio = io_bytes_read / payload_bytes."""
        ds = FakeDataset(length=50)
        result = bm.benchmark_one_config(
            ds, num_workers=0, batch_size=4, num_iterations=5, warmup_iterations=1
        )
        if result["io_amplification_ratio"] is not None:
            expected = round(
                result["io_bytes_read"] / result["payload_bytes"], 4
            )
            assert result["io_amplification_ratio"] == expected


# --- CSV field coverage ---


class TestCsvFieldnames:
    """Verify save_csv includes the new I/O columns."""

    def test_fieldnames_include_io_columns(self):
        """The CSV fieldnames list in save_csv includes io_bytes_read,
        payload_bytes, and io_amplification_ratio."""
        import inspect

        source = inspect.getsource(bm.save_csv)
        assert "io_bytes_read" in source
        assert "payload_bytes" in source
        assert "io_amplification_ratio" in source
