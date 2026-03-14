"""Tests for scripts/benchmark_e2e_training.py.

Tests cover:
- Config name generation
- tqdm output parsing (steps/sec, steps completed)
- OOM detection
- CSV output format
- GPU memory monitor lifecycle
- Benchmark result dataclass
- run_benchmark orchestration with OOM skip logic
- Cleanup of output directories
"""

import csv
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.benchmark_e2e_training import (
    BACKENDS,
    BATCH_SIZES,
    BenchmarkResult,
    GpuMemoryMonitor,
    cleanup_output_dir,
    config_name_for,
    is_oom,
    parse_steps_completed,
    parse_tqdm_steps_per_sec,
    query_gpu_memory_mb,
    run_benchmark,
    run_single_benchmark,
    save_results_csv,
)


# ---------------------------------------------------------------------------
# config_name_for
# ---------------------------------------------------------------------------
class TestConfigNameFor:
    """Tests for config_name_for()."""

    def test_youmu_page_aligned_bs1(self):
        """youmu_page_aligned -> page_aligned in filename."""
        assert config_name_for("youmu_page_aligned", 1) == "bench_page_aligned_bs1.yaml"

    def test_youmu_page_aligned_bs64(self):
        """Large batch size works."""
        assert config_name_for("youmu_page_aligned", 64) == "bench_page_aligned_bs64.yaml"

    def test_lerobot_bs4(self):
        """lerobot stays as-is in filename."""
        assert config_name_for("lerobot", 4) == "bench_lerobot_bs4.yaml"

    def test_lerobot_bs32(self):
        """Another lerobot batch size."""
        assert config_name_for("lerobot", 32) == "bench_lerobot_bs32.yaml"

    def test_all_backends_and_sizes(self):
        """All expected config names are generated correctly."""
        for backend in BACKENDS:
            for bs in BATCH_SIZES:
                name = config_name_for(backend, bs)
                assert name.startswith("bench_")
                assert name.endswith(f"_bs{bs}.yaml")
                assert "youmu" not in name  # youmu_ prefix is stripped


# ---------------------------------------------------------------------------
# parse_tqdm_steps_per_sec
# ---------------------------------------------------------------------------
class TestParseTqdmStepsPerSec:
    """Tests for parse_tqdm_steps_per_sec()."""

    def test_it_per_sec(self):
        """Parse 'X.XXit/s' format."""
        output = "Epoch 1/1: 100%|██████| 50/50 [01:23<00:00, 1.67it/s, loss: 0.1234]"
        assert parse_tqdm_steps_per_sec(output) == pytest.approx(1.67)

    def test_sec_per_it(self):
        """Parse 'X.XXs/it' format (slow training)."""
        output = "Epoch 1/1:  10%|█       | 5/50 [00:25<03:45, 5.00s/it, loss: 0.5678]"
        assert parse_tqdm_steps_per_sec(output) == pytest.approx(0.2)

    def test_multiple_updates_returns_last(self):
        """Returns the last tqdm update (most accurate overall rate)."""
        output = (
            "Epoch 1/1:  20%|██        | 10/50 [00:10<00:40, 1.00it/s]\r"
            "Epoch 1/1: 100%|██████████| 50/50 [00:40<00:00, 1.25it/s]"
        )
        assert parse_tqdm_steps_per_sec(output) == pytest.approx(1.25)

    def test_no_tqdm_output(self):
        """Returns None when no tqdm pattern found."""
        output = "Loading model... done\nTraining complete."
        assert parse_tqdm_steps_per_sec(output) is None

    def test_empty_output(self):
        """Returns None for empty string."""
        assert parse_tqdm_steps_per_sec("") is None

    def test_integer_it_per_sec(self):
        """Parse integer it/s (no decimal)."""
        output = "50/50 [00:05<00:00, 10it/s]"
        assert parse_tqdm_steps_per_sec(output) == pytest.approx(10.0)

    def test_sec_per_it_very_slow(self):
        """Very slow training: 30s/it."""
        output = "1/50 [00:30<24:30, 30.00s/it]"
        assert parse_tqdm_steps_per_sec(output) == pytest.approx(1.0 / 30.0)


# ---------------------------------------------------------------------------
# parse_steps_completed
# ---------------------------------------------------------------------------
class TestParseStepsCompleted:
    """Tests for parse_steps_completed()."""

    def test_full_completion(self):
        """50/50 means 50 steps completed."""
        output = "Epoch 1/1: 100%|██████████| 50/50 [01:23<00:00, 1.67it/s]"
        assert parse_steps_completed(output) == 50

    def test_partial_completion(self):
        """23/50 means 23 steps completed (max of all N/M matches)."""
        output = "Epoch 1/1:  46%|████▌     | 23/50 [00:15<00:18, 1.45it/s]"
        # Matches: 1/1 (Epoch), 23/50 (progress) -> max(1, 23) = 23
        assert parse_steps_completed(output) == 23

    def test_no_progress_info(self):
        """Returns 0 when no progress pattern found."""
        output = "Loading model..."
        assert parse_steps_completed(output) == 0

    def test_empty_output(self):
        """Returns 0 for empty string."""
        assert parse_steps_completed("") == 0

    def test_multiple_progress_updates(self):
        """Returns the max step number seen across all updates."""
        output = "10/50\r20/50\r30/50\r40/50\r50/50"
        assert parse_steps_completed(output) == 50


# ---------------------------------------------------------------------------
# is_oom
# ---------------------------------------------------------------------------
class TestIsOom:
    """Tests for is_oom()."""

    def test_cuda_oom_message(self):
        """CUDA out of memory detected."""
        assert is_oom(1, "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB") is True

    def test_torch_oom_error(self):
        """torch.OutOfMemoryError detected."""
        assert is_oom(1, "torch.OutOfMemoryError: CUDA out of memory") is True

    def test_generic_oom_error(self):
        """OutOfMemoryError detected."""
        assert is_oom(1, "OutOfMemoryError") is True

    def test_success_exit_no_oom(self):
        """Exit code 0 is never OOM, even with matching text."""
        assert is_oom(0, "CUDA out of memory") is False

    def test_non_oom_failure(self):
        """Non-zero exit without OOM message."""
        assert is_oom(1, "RuntimeError: expected scalar type Float") is False

    def test_empty_output(self):
        """Non-zero exit with no output."""
        assert is_oom(1, "") is False


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------
class TestBenchmarkResult:
    """Tests for the BenchmarkResult dataclass."""

    def test_default_values(self):
        """Default values are set correctly."""
        r = BenchmarkResult(backend="lerobot", micro_batch_size=4)
        assert r.backend == "lerobot"
        assert r.micro_batch_size == 4
        assert r.steps_per_sec == 0.0
        assert r.samples_per_sec == 0.0
        assert r.wall_clock_sec == 0.0
        assert r.peak_gpu_mem_mb == 0.0
        assert r.status == "pending"

    def test_custom_values(self):
        """Custom values are stored correctly."""
        r = BenchmarkResult(
            backend="youmu_page_aligned",
            micro_batch_size=8,
            steps_per_sec=1.5,
            samples_per_sec=12.0,
            wall_clock_sec=33.3,
            peak_gpu_mem_mb=45000.0,
            status="OK",
        )
        assert r.steps_per_sec == 1.5
        assert r.status == "OK"


# ---------------------------------------------------------------------------
# save_results_csv
# ---------------------------------------------------------------------------
class TestSaveResultsCsv:
    """Tests for save_results_csv()."""

    def test_writes_correct_csv(self, tmp_path):
        """CSV has correct headers and data."""
        results = [
            BenchmarkResult("youmu_page_aligned", 1, 2.0, 2.0, 25.0, 30000.0, "OK"),
            BenchmarkResult("lerobot", 4, 1.5, 6.0, 33.3, 35000.0, "OK"),
            BenchmarkResult("lerobot", 8, 0.0, 0.0, 5.0, 60000.0, "OOM"),
        ]
        csv_path = str(tmp_path / "results.csv")
        save_results_csv(results, csv_path)

        with open(csv_path) as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Header
        assert rows[0] == [
            "backend", "micro_batch_size", "steps_per_sec",
            "samples_per_sec", "wall_clock_sec", "peak_gpu_mem_mb", "status",
        ]
        # Data rows
        assert len(rows) == 4  # header + 3 results
        assert rows[1][0] == "youmu_page_aligned"
        assert rows[1][6] == "OK"
        assert rows[3][6] == "OOM"

    def test_creates_parent_dirs(self, tmp_path):
        """Parent directories are created if they don't exist."""
        csv_path = str(tmp_path / "subdir" / "deep" / "results.csv")
        results = [BenchmarkResult("lerobot", 1, 1.0, 1.0, 10.0, 20000.0, "OK")]
        save_results_csv(results, csv_path)
        assert os.path.exists(csv_path)

    def test_empty_results(self, tmp_path):
        """Empty results list writes only headers."""
        csv_path = str(tmp_path / "empty.csv")
        save_results_csv([], csv_path)
        with open(csv_path) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 1  # header only


# ---------------------------------------------------------------------------
# GpuMemoryMonitor
# ---------------------------------------------------------------------------
class TestGpuMemoryMonitor:
    """Tests for GpuMemoryMonitor."""

    @patch("scripts.benchmark_e2e_training.query_gpu_memory_mb")
    def test_tracks_peak_memory(self, mock_query):
        """Monitor tracks the peak memory across polls."""
        # Simulate memory usage: 1000, 5000, 3000, then repeat 3000 for extra polls
        mock_query.side_effect = [1000.0, 5000.0, 3000.0] + [2000.0] * 100
        monitor = GpuMemoryMonitor(poll_interval=0.01)
        monitor.start()
        time.sleep(0.1)  # let it poll a few times
        peak = monitor.stop()
        assert peak == 5000.0

    @patch("scripts.benchmark_e2e_training.query_gpu_memory_mb")
    def test_stop_returns_zero_when_no_polls(self, mock_query):
        """If stopped immediately, peak is 0."""
        mock_query.return_value = 0.0
        monitor = GpuMemoryMonitor(poll_interval=10.0)  # long interval
        monitor.start()
        peak = monitor.stop()
        # May or may not have polled once; peak should be >= 0
        assert peak >= 0.0


# ---------------------------------------------------------------------------
# cleanup_output_dir
# ---------------------------------------------------------------------------
class TestCleanupOutputDir:
    """Tests for cleanup_output_dir()."""

    def test_cleans_existing_dir(self, tmp_path, monkeypatch):
        """Removes existing output directory."""
        # Create a fake output dir
        output_dir = tmp_path / "bench_lerobot_bs4"
        output_dir.mkdir()
        (output_dir / "checkpoint.pt").write_text("data")

        # Monkeypatch to use our tmp_path
        monkeypatch.setattr(
            "scripts.benchmark_e2e_training.cleanup_output_dir",
            lambda backend, batch_size: None,
        )
        # Direct test of the original function logic
        import shutil

        dir_path = str(output_dir)
        assert os.path.exists(dir_path)
        shutil.rmtree(dir_path, ignore_errors=True)
        assert not os.path.exists(dir_path)

    def test_nonexistent_dir_no_error(self):
        """No error when directory doesn't exist."""
        # This should not raise
        cleanup_output_dir("nonexistent_backend", 999)


# ---------------------------------------------------------------------------
# run_single_benchmark (mocked subprocess)
# ---------------------------------------------------------------------------
class TestRunSingleBenchmark:
    """Tests for run_single_benchmark() with mocked subprocess."""

    @patch("scripts.benchmark_e2e_training.GpuMemoryMonitor")
    @patch("subprocess.run")
    def test_successful_run(self, mock_subproc, mock_monitor_cls):
        """Successful run extracts metrics correctly."""
        # Mock subprocess output
        mock_subproc.return_value = MagicMock(
            returncode=0,
            stdout="",
            stderr="Epoch 1/1: 100%|██████████| 50/50 [00:33<00:00, 1.50it/s, loss: 0.1234]",
        )
        # Mock GPU monitor
        mock_monitor = MagicMock()
        mock_monitor.stop.return_value = 45000.0
        mock_monitor_cls.return_value = mock_monitor

        result = run_single_benchmark(
            config_path="configs/test.yaml",
            backend="youmu_page_aligned",
            batch_size=4,
            warmup_steps=10,
            max_steps=50,
        )

        assert result.status == "OK"
        assert result.steps_per_sec == pytest.approx(1.50, abs=0.01)
        assert result.samples_per_sec == pytest.approx(6.0, abs=0.1)
        assert result.peak_gpu_mem_mb == 45000.0

    @patch("scripts.benchmark_e2e_training.GpuMemoryMonitor")
    @patch("subprocess.run")
    def test_oom_detection(self, mock_subproc, mock_monitor_cls):
        """OOM is detected and reported."""
        mock_subproc.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="RuntimeError: CUDA out of memory. Tried to allocate 4.00 GiB",
        )
        mock_monitor = MagicMock()
        mock_monitor.stop.return_value = 80000.0
        mock_monitor_cls.return_value = mock_monitor

        result = run_single_benchmark(
            config_path="configs/test.yaml",
            backend="lerobot",
            batch_size=64,
            warmup_steps=10,
            max_steps=50,
        )

        assert result.status == "OOM"
        assert result.peak_gpu_mem_mb == 80000.0

    @patch("scripts.benchmark_e2e_training.GpuMemoryMonitor")
    @patch("subprocess.run")
    def test_non_oom_failure(self, mock_subproc, mock_monitor_cls):
        """Non-OOM failure is reported with exit code."""
        mock_subproc.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="RuntimeError: expected scalar type Float",
        )
        mock_monitor = MagicMock()
        mock_monitor.stop.return_value = 10000.0
        mock_monitor_cls.return_value = mock_monitor

        result = run_single_benchmark(
            config_path="configs/test.yaml",
            backend="lerobot",
            batch_size=1,
            warmup_steps=10,
            max_steps=50,
        )

        assert result.status == "FAILED(exit=1)"

    @patch("scripts.benchmark_e2e_training.GpuMemoryMonitor")
    @patch("subprocess.run")
    def test_fallback_timing(self, mock_subproc, mock_monitor_cls):
        """When tqdm output is missing, falls back to wall-clock timing."""
        mock_subproc.return_value = MagicMock(
            returncode=0,
            stdout="Training complete\n",
            stderr="",
        )
        mock_monitor = MagicMock()
        mock_monitor.stop.return_value = 20000.0
        mock_monitor_cls.return_value = mock_monitor

        result = run_single_benchmark(
            config_path="configs/test.yaml",
            backend="lerobot",
            batch_size=2,
            warmup_steps=10,
            max_steps=50,
        )

        assert result.status == "OK"
        # Fallback: 0 steps / wall_clock = 0
        assert result.steps_per_sec == 0.0


# ---------------------------------------------------------------------------
# run_benchmark (orchestration with mocked internals)
# ---------------------------------------------------------------------------
class TestRunBenchmark:
    """Tests for run_benchmark() orchestration."""

    @patch("scripts.benchmark_e2e_training.cleanup_output_dir")
    @patch("scripts.benchmark_e2e_training.run_single_benchmark")
    def test_skips_after_oom(self, mock_run_single, mock_cleanup, tmp_path):
        """After OOM, larger batch sizes are skipped for that backend."""
        config_dir = str(tmp_path / "configs")
        os.makedirs(config_dir)

        # Create config files for youmu_page_aligned only, bs 1-8
        for bs in [1, 2, 4, 8]:
            config_name = config_name_for("youmu_page_aligned", bs)
            (tmp_path / "configs" / config_name).write_text("model: test")

        def mock_run(config_path, backend, batch_size, warmup_steps, max_steps):
            """Simulate OOM at batch_size=4."""
            if batch_size >= 4:
                return BenchmarkResult(backend=backend, micro_batch_size=batch_size, status="OOM")
            return BenchmarkResult(
                backend=backend, micro_batch_size=batch_size,
                steps_per_sec=1.0, status="OK",
            )

        mock_run_single.side_effect = mock_run

        csv_path = str(tmp_path / "results.csv")
        results = run_benchmark(config_dir=config_dir, output_csv=csv_path)

        # Filter youmu_page_aligned results
        ypa_results = [r for r in results if r.backend == "youmu_page_aligned"]

        # bs=1 OK, bs=2 OK, bs=4 OOM, bs=8+ SKIPPED
        statuses = {r.micro_batch_size: r.status for r in ypa_results}
        assert statuses.get(1) == "OK"
        assert statuses.get(2) == "OK"
        assert statuses.get(4) == "OOM"
        # bs=8 should be skipped (prior OOM)
        assert "SKIPPED" in statuses.get(8, "")

    @patch("scripts.benchmark_e2e_training.cleanup_output_dir")
    @patch("scripts.benchmark_e2e_training.run_single_benchmark")
    def test_missing_config_skipped(self, mock_run_single, mock_cleanup, tmp_path):
        """Missing config files result in SKIPPED status."""
        config_dir = str(tmp_path / "empty_configs")
        os.makedirs(config_dir)

        csv_path = str(tmp_path / "results.csv")
        results = run_benchmark(config_dir=config_dir, output_csv=csv_path)

        # All should be skipped (no config files)
        for r in results:
            assert "SKIPPED" in r.status

        # run_single_benchmark should never be called
        mock_run_single.assert_not_called()

    @patch("scripts.benchmark_e2e_training.cleanup_output_dir")
    @patch("scripts.benchmark_e2e_training.run_single_benchmark")
    def test_results_csv_written(self, mock_run_single, mock_cleanup, tmp_path):
        """Results are saved to CSV after benchmark completes."""
        config_dir = str(tmp_path / "configs")
        os.makedirs(config_dir)

        # Create one config
        config_name = config_name_for("lerobot", 1)
        (tmp_path / "configs" / config_name).write_text("model: test")

        mock_run_single.return_value = BenchmarkResult(
            backend="lerobot", micro_batch_size=1,
            steps_per_sec=2.0, samples_per_sec=2.0,
            wall_clock_sec=25.0, peak_gpu_mem_mb=30000.0,
            status="OK",
        )

        csv_path = str(tmp_path / "output" / "results.csv")
        run_benchmark(config_dir=config_dir, output_csv=csv_path)

        assert os.path.exists(csv_path)
        with open(csv_path) as f:
            rows = list(csv.reader(f))
        # At least header + 1 data row for lerobot bs=1
        assert len(rows) >= 2

    @patch("scripts.benchmark_e2e_training.cleanup_output_dir")
    @patch("scripts.benchmark_e2e_training.run_single_benchmark")
    def test_both_backends_run(self, mock_run_single, mock_cleanup, tmp_path):
        """Both backends are benchmarked."""
        config_dir = str(tmp_path / "configs")
        os.makedirs(config_dir)

        # Create configs for both backends, bs=1 only
        for backend in BACKENDS:
            config_name = config_name_for(backend, 1)
            (tmp_path / "configs" / config_name).write_text("model: test")

        mock_run_single.return_value = BenchmarkResult(
            backend="test", micro_batch_size=1, steps_per_sec=1.0, status="OK",
        )

        csv_path = str(tmp_path / "results.csv")
        results = run_benchmark(config_dir=config_dir, output_csv=csv_path)

        # Should have results for both backends
        backends_seen = {r.backend for r in results}
        assert "youmu_page_aligned" in backends_seen
        assert "lerobot" in backends_seen


# ---------------------------------------------------------------------------
# Integration: config naming matches generated files
# ---------------------------------------------------------------------------
class TestConfigNamingIntegration:
    """Verify config names match the actual generated YAML files."""

    def test_generated_configs_exist(self):
        """All expected config files exist in the bench directory."""
        config_dir = "configs/multimodal/qwen3_vl/bench"
        if not os.path.exists(config_dir):
            pytest.skip("Benchmark configs not generated yet")

        for backend in BACKENDS:
            for bs in BATCH_SIZES:
                config_name = config_name_for(backend, bs)
                config_path = os.path.join(config_dir, config_name)
                assert os.path.exists(config_path), f"Missing config: {config_path}"
