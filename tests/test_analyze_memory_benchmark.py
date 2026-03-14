"""Tests for scripts/analyze_memory_benchmark.py.

Tests CSV aggregation, report generation, and CLI behavior.
"""

import csv
import os
import tempfile

import pytest

from scripts.analyze_memory_benchmark import (
    COMBINED_FIELDNAMES,
    aggregate_csvs,
    find_crossover_point,
    generate_report,
    get_best_throughput_per_backend,
    get_oom_failures,
    memory_limit_sort_key,
    parse_memory_limit_from_filename,
    save_combined_csv,
)


# --- Fixtures ---


def _make_csv(directory: str, filename: str, rows: list[dict]) -> str:
    """Helper to write a CSV file with standard benchmark columns.

    Args:
        directory: Directory to write the file in.
        filename: Name of the CSV file.
        rows: List of row dicts.

    Returns:
        Path to the created CSV file.
    """
    fieldnames = [
        "backend", "obs_len", "pred_len", "num_workers", "batch_size",
        "samples_per_sec", "time_to_first_sample_sec", "peak_rss_mb",
        "total_samples", "total_time_sec", "batches_completed",
        "io_bytes_read", "payload_bytes", "io_amplification_ratio", "error",
    ]
    filepath = os.path.join(directory, filename)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return filepath


@pytest.fixture
def sample_csv_dir(tmp_path):
    """Create a temporary directory with sample per-RAM-limit CSV files."""
    # mem_4G.csv: youmu and lerobot both succeed
    _make_csv(str(tmp_path), "mem_4G.csv", [
        {
            "backend": "youmu", "obs_len": "1", "pred_len": "4",
            "num_workers": "2", "batch_size": "8",
            "samples_per_sec": "120.50", "time_to_first_sample_sec": "0.0012",
            "peak_rss_mb": "512.0", "total_samples": "400",
            "total_time_sec": "3.32", "batches_completed": "50",
            "io_bytes_read": "100000", "payload_bytes": "80000",
            "io_amplification_ratio": "1.25", "error": "",
        },
        {
            "backend": "lerobot", "obs_len": "1", "pred_len": "4",
            "num_workers": "2", "batch_size": "8",
            "samples_per_sec": "95.30", "time_to_first_sample_sec": "0.0050",
            "peak_rss_mb": "1024.0", "total_samples": "400",
            "total_time_sec": "4.20", "batches_completed": "50",
            "io_bytes_read": "200000", "payload_bytes": "80000",
            "io_amplification_ratio": "2.50", "error": "",
        },
    ])

    # mem_2G.csv: youmu succeeds, lerobot has error
    _make_csv(str(tmp_path), "mem_2G.csv", [
        {
            "backend": "youmu", "obs_len": "1", "pred_len": "4",
            "num_workers": "0", "batch_size": "4",
            "samples_per_sec": "80.00", "time_to_first_sample_sec": "0.0020",
            "peak_rss_mb": "400.0", "total_samples": "200",
            "total_time_sec": "2.50", "batches_completed": "50",
            "io_bytes_read": "90000", "payload_bytes": "70000",
            "io_amplification_ratio": "1.29", "error": "",
        },
        {
            "backend": "lerobot", "obs_len": "1", "pred_len": "4",
            "num_workers": "0", "batch_size": "4",
            "samples_per_sec": "", "time_to_first_sample_sec": "",
            "peak_rss_mb": "", "total_samples": "",
            "total_time_sec": "", "batches_completed": "",
            "io_bytes_read": "", "payload_bytes": "",
            "io_amplification_ratio": "", "error": "OOM killed",
        },
    ])

    # mem_8G.csv: both succeed, lerobot is faster
    _make_csv(str(tmp_path), "mem_8G.csv", [
        {
            "backend": "youmu", "obs_len": "2", "pred_len": "4",
            "num_workers": "4", "batch_size": "8",
            "samples_per_sec": "150.00", "time_to_first_sample_sec": "0.0010",
            "peak_rss_mb": "600.0", "total_samples": "400",
            "total_time_sec": "2.67", "batches_completed": "50",
            "io_bytes_read": "110000", "payload_bytes": "90000",
            "io_amplification_ratio": "1.22", "error": "",
        },
        {
            "backend": "lerobot", "obs_len": "2", "pred_len": "4",
            "num_workers": "4", "batch_size": "8",
            "samples_per_sec": "200.00", "time_to_first_sample_sec": "0.0030",
            "peak_rss_mb": "1500.0", "total_samples": "400",
            "total_time_sec": "2.00", "batches_completed": "50",
            "io_bytes_read": "250000", "payload_bytes": "90000",
            "io_amplification_ratio": "2.78", "error": "",
        },
    ])

    return str(tmp_path)


# --- Unit tests: parse_memory_limit_from_filename ---


class TestParseMemoryLimitFromFilename:
    """Tests for parse_memory_limit_from_filename."""

    def test_valid_filenames(self):
        """Parses standard mem_<N>G.csv filenames."""
        assert parse_memory_limit_from_filename("mem_1G.csv") == "1G"
        assert parse_memory_limit_from_filename("mem_4G.csv") == "4G"
        assert parse_memory_limit_from_filename("mem_32G.csv") == "32G"

    def test_invalid_filename_raises(self):
        """Raises ValueError for non-matching filenames."""
        with pytest.raises(ValueError):
            parse_memory_limit_from_filename("results.csv")
        with pytest.raises(ValueError):
            parse_memory_limit_from_filename("mem_4M.csv")
        with pytest.raises(ValueError):
            parse_memory_limit_from_filename("mem_.csv")


# --- Unit tests: memory_limit_sort_key ---


class TestMemoryLimitSortKey:
    """Tests for memory_limit_sort_key."""

    def test_sort_order(self):
        """Returns numeric values for sorting."""
        limits = ["8G", "1G", "32G", "2G", "4G", "16G"]
        sorted_limits = sorted(limits, key=memory_limit_sort_key)
        assert sorted_limits == ["1G", "2G", "4G", "8G", "16G", "32G"]

    def test_invalid_format_returns_zero(self):
        """Returns 0 for unrecognized formats."""
        assert memory_limit_sort_key("invalid") == 0


# --- Unit tests: aggregate_csvs ---


class TestAggregateCsvs:
    """Tests for aggregate_csvs."""

    def test_combines_all_files(self, sample_csv_dir):
        """Reads all mem_*G.csv files and adds memory_limit column."""
        combined = aggregate_csvs(sample_csv_dir)
        # 2 rows in mem_2G + 2 in mem_4G + 2 in mem_8G = 6
        assert len(combined) == 6
        # All rows should have memory_limit
        for row in combined:
            assert "memory_limit" in row

    def test_sorted_by_memory_limit(self, sample_csv_dir):
        """Results are sorted by memory limit ascending."""
        combined = aggregate_csvs(sample_csv_dir)
        limits = [r["memory_limit"] for r in combined]
        # Should be 2G, 2G, 4G, 4G, 8G, 8G
        assert limits == ["2G", "2G", "4G", "4G", "8G", "8G"]

    def test_empty_dir_raises(self, tmp_path):
        """Raises FileNotFoundError when no matching CSVs exist."""
        with pytest.raises(FileNotFoundError):
            aggregate_csvs(str(tmp_path))

    def test_preserves_original_columns(self, sample_csv_dir):
        """Original CSV columns are preserved in combined output."""
        combined = aggregate_csvs(sample_csv_dir)
        first = combined[0]
        assert "backend" in first
        assert "samples_per_sec" in first
        assert "error" in first


# --- Unit tests: save_combined_csv ---


class TestSaveCombinedCsv:
    """Tests for save_combined_csv."""

    def test_writes_csv_with_header(self, sample_csv_dir, tmp_path):
        """Writes CSV with correct fieldnames and all rows."""
        combined = aggregate_csvs(sample_csv_dir)
        output = str(tmp_path / "out" / "combined.csv")
        save_combined_csv(combined, output)

        assert os.path.exists(output)
        with open(output, newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == COMBINED_FIELDNAMES
            rows = list(reader)
            assert len(rows) == 6

    def test_creates_directories(self, tmp_path):
        """Creates parent directories if they don't exist."""
        output = str(tmp_path / "deep" / "nested" / "combined.csv")
        save_combined_csv([], output)
        assert os.path.exists(output)


# --- Unit tests: get_best_throughput_per_backend ---


class TestGetBestThroughputPerBackend:
    """Tests for get_best_throughput_per_backend."""

    def test_finds_best_per_group(self, sample_csv_dir):
        """Returns highest samples/sec for each (memory_limit, backend)."""
        combined = aggregate_csvs(sample_csv_dir)
        best = get_best_throughput_per_backend(combined)
        assert best[("4G", "youmu")]["samples_per_sec"] == 120.50
        assert best[("4G", "lerobot")]["samples_per_sec"] == 95.30
        assert best[("8G", "youmu")]["samples_per_sec"] == 150.00

    def test_skips_error_rows(self, sample_csv_dir):
        """Rows with errors are excluded from best throughput."""
        combined = aggregate_csvs(sample_csv_dir)
        best = get_best_throughput_per_backend(combined)
        # 2G lerobot had OOM error, should not appear
        assert ("2G", "lerobot") not in best

    def test_includes_config_details(self, sample_csv_dir):
        """Best result includes configuration parameters."""
        combined = aggregate_csvs(sample_csv_dir)
        best = get_best_throughput_per_backend(combined)
        entry = best[("4G", "youmu")]
        assert entry["num_workers"] == "2"
        assert entry["batch_size"] == "8"


# --- Unit tests: get_oom_failures ---


class TestGetOomFailures:
    """Tests for get_oom_failures."""

    def test_counts_error_rows(self, sample_csv_dir):
        """Counts rows with non-empty error field."""
        combined = aggregate_csvs(sample_csv_dir)
        failures = get_oom_failures(combined)
        assert failures[("2G", "lerobot")] == 1
        assert ("4G", "youmu") not in failures

    def test_no_failures(self):
        """Returns empty dict when no errors exist."""
        rows = [{"memory_limit": "4G", "backend": "youmu", "error": ""}]
        assert get_oom_failures(rows) == {}


# --- Unit tests: find_crossover_point ---


class TestFindCrossoverPoint:
    """Tests for find_crossover_point."""

    def test_finds_crossover(self):
        """Returns first memory limit where youmu >= lerobot."""
        best = {
            ("2G", "youmu"): {"samples_per_sec": 50},
            ("2G", "lerobot"): {"samples_per_sec": 60},
            ("4G", "youmu"): {"samples_per_sec": 100},
            ("4G", "lerobot"): {"samples_per_sec": 90},
        }
        assert find_crossover_point(best, ["2G", "4G"]) == "4G"

    def test_no_crossover(self):
        """Returns None when youmu never meets lerobot."""
        best = {
            ("4G", "youmu"): {"samples_per_sec": 50},
            ("4G", "lerobot"): {"samples_per_sec": 100},
        }
        assert find_crossover_point(best, ["4G"]) is None

    def test_immediate_crossover(self):
        """Returns first limit if youmu already ahead."""
        best = {
            ("1G", "youmu"): {"samples_per_sec": 80},
            ("1G", "lerobot"): {"samples_per_sec": 70},
        }
        assert find_crossover_point(best, ["1G"]) == "1G"

    def test_missing_backend_data(self):
        """Skips limits where one backend has no data."""
        best = {
            ("2G", "youmu"): {"samples_per_sec": 100},
            # No lerobot at 2G
            ("4G", "youmu"): {"samples_per_sec": 100},
            ("4G", "lerobot"): {"samples_per_sec": 90},
        }
        assert find_crossover_point(best, ["2G", "4G"]) == "4G"


# --- Unit tests: generate_report ---


class TestGenerateReport:
    """Tests for generate_report."""

    def test_report_contains_title(self, sample_csv_dir):
        """Report starts with the expected title."""
        combined = aggregate_csvs(sample_csv_dir)
        report = generate_report(combined)
        assert "# Memory-Constrained Data Loading Benchmark Report" in report

    def test_report_contains_summary_table(self, sample_csv_dir):
        """Report contains summary table with RAM limits."""
        combined = aggregate_csvs(sample_csv_dir)
        report = generate_report(combined)
        assert "| RAM Limit |" in report
        assert "| 4G |" in report
        assert "120.50" in report

    def test_report_contains_analysis(self, sample_csv_dir):
        """Report has analysis section with crossover info."""
        combined = aggregate_csvs(sample_csv_dir)
        report = generate_report(combined)
        assert "## Analysis" in report
        # youmu beats lerobot at 2G (80 vs OOM) and 4G (120.50 vs 95.30)
        assert "crossover" in report.lower()

    def test_report_contains_oom_notes(self, sample_csv_dir):
        """Report notes OOM failures."""
        combined = aggregate_csvs(sample_csv_dir)
        report = generate_report(combined)
        assert "## OOM / Error Notes" in report
        assert "2G" in report
        assert "lerobot" in report

    def test_report_contains_full_results(self, sample_csv_dir):
        """Report has full results table with all rows."""
        combined = aggregate_csvs(sample_csv_dir)
        report = generate_report(combined)
        assert "## Full Results" in report
        # Check some data values appear
        assert "150.00" in report  # 8G youmu
        assert "200.00" in report  # 8G lerobot

    def test_report_no_oom_section_when_clean(self, tmp_path):
        """Report omits OOM section when there are no errors."""
        _make_csv(str(tmp_path), "mem_4G.csv", [
            {
                "backend": "youmu", "obs_len": "1", "pred_len": "4",
                "num_workers": "0", "batch_size": "4",
                "samples_per_sec": "100.0", "time_to_first_sample_sec": "0.001",
                "peak_rss_mb": "500.0", "total_samples": "200",
                "total_time_sec": "2.0", "batches_completed": "50",
                "io_bytes_read": "80000", "payload_bytes": "70000",
                "io_amplification_ratio": "1.14", "error": "",
            },
        ])
        combined = aggregate_csvs(str(tmp_path))
        report = generate_report(combined)
        assert "## OOM / Error Notes" not in report


# --- Integration test: CLI ---


class TestCLI:
    """Integration tests for the CLI entry point."""

    def test_end_to_end(self, sample_csv_dir, tmp_path):
        """Full pipeline: aggregate -> combined CSV -> report."""
        combined_out = str(tmp_path / "combined.csv")
        report_out = str(tmp_path / "report.md")

        combined = aggregate_csvs(sample_csv_dir)
        save_combined_csv(combined, combined_out)
        report = generate_report(combined)
        with open(report_out, "w") as f:
            f.write(report)

        # Verify combined CSV
        assert os.path.exists(combined_out)
        with open(combined_out, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 6
            assert all("memory_limit" in r for r in rows)

        # Verify report
        assert os.path.exists(report_out)
        with open(report_out) as f:
            content = f.read()
            assert len(content) > 100
            assert "# Memory-Constrained" in content
