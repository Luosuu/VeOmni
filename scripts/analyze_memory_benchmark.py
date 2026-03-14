"""Aggregate memory-constrained benchmark results and generate a report.

Reads per-RAM-limit CSV files from benchmarks/memory_constrained/,
combines them into a single CSV with a memory_limit column, and
generates a markdown report with summary tables and analysis.

Usage:
    . .venv/bin/activate
    python scripts/analyze_memory_benchmark.py
    python scripts/analyze_memory_benchmark.py --input-dir benchmarks/memory_constrained
"""

import argparse
import csv
import os
import re
from pathlib import Path


# Ordered RAM limits for consistent sorting
RAM_LIMIT_ORDER = ["1G", "2G", "4G", "8G", "16G", "32G"]

# CSV fieldnames from benchmark_data_loading.py plus memory_limit
COMBINED_FIELDNAMES = [
    "memory_limit",
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


def parse_memory_limit_from_filename(filename: str) -> str:
    """Extract memory limit string from a filename like mem_4G.csv.

    Args:
        filename: Basename of the CSV file (e.g. "mem_4G.csv").

    Returns:
        Memory limit string (e.g. "4G").

    Raises:
        ValueError: If filename doesn't match expected pattern.
    """
    match = re.match(r"^mem_(\d+G)\.csv$", filename)
    if not match:
        raise ValueError(f"Filename does not match expected pattern mem_<N>G.csv: {filename}")
    return match.group(1)


def memory_limit_sort_key(limit: str) -> int:
    """Return a numeric sort key for a memory limit string.

    Args:
        limit: Memory limit string like "4G".

    Returns:
        Integer value in GB for sorting.
    """
    match = re.match(r"^(\d+)G$", limit)
    if not match:
        return 0
    return int(match.group(1))


def read_csv_file(filepath: str) -> list[dict]:
    """Read a CSV file and return rows as list of dicts.

    Args:
        filepath: Path to the CSV file.

    Returns:
        List of row dicts.
    """
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def aggregate_csvs(input_dir: str) -> list[dict]:
    """Read all mem_*.csv files from input_dir, add memory_limit column.

    Args:
        input_dir: Directory containing per-RAM-limit CSV files.

    Returns:
        Combined list of row dicts with memory_limit column added,
        sorted by memory limit then backend.
    """
    input_path = Path(input_dir)
    csv_files = sorted(input_path.glob("mem_*G.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No mem_*G.csv files found in {input_dir}")

    combined = []
    for csv_file in csv_files:
        mem_limit = parse_memory_limit_from_filename(csv_file.name)
        rows = read_csv_file(str(csv_file))
        for row in rows:
            row["memory_limit"] = mem_limit
            combined.append(row)

    # Sort by memory limit (numeric), then backend
    combined.sort(key=lambda r: (memory_limit_sort_key(r["memory_limit"]), r.get("backend", "")))
    return combined


def save_combined_csv(rows: list[dict], output_path: str) -> None:
    """Save combined results to a single CSV.

    Args:
        rows: List of row dicts with memory_limit column.
        output_path: Path to output CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COMBINED_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def get_best_throughput_per_backend(rows: list[dict]) -> dict:
    """Find the best (highest) samples/sec for each (memory_limit, backend).

    Args:
        rows: Combined row dicts.

    Returns:
        Dict mapping (memory_limit, backend) -> dict with best row info.
    """
    best = {}
    for row in rows:
        # Skip rows with errors or missing throughput
        if row.get("error"):
            continue
        sps_str = row.get("samples_per_sec", "")
        if not sps_str:
            continue
        sps = float(sps_str)

        key = (row["memory_limit"], row["backend"])
        if key not in best or sps > best[key]["samples_per_sec"]:
            best[key] = {
                "samples_per_sec": sps,
                "num_workers": row.get("num_workers", ""),
                "obs_len": row.get("obs_len", ""),
                "batch_size": row.get("batch_size", ""),
                "peak_rss_mb": row.get("peak_rss_mb", ""),
            }
    return best


def get_oom_failures(rows: list[dict]) -> dict:
    """Identify which (memory_limit, backend) pairs had errors (likely OOM).

    Args:
        rows: Combined row dicts.

    Returns:
        Dict mapping (memory_limit, backend) -> count of error rows.
    """
    failures = {}
    for row in rows:
        if row.get("error"):
            key = (row["memory_limit"], row["backend"])
            failures[key] = failures.get(key, 0) + 1
    return failures


def find_crossover_point(best: dict, mem_limits: list[str]) -> str | None:
    """Find the lowest memory limit where youmu throughput >= lerobot.

    Args:
        best: Dict from get_best_throughput_per_backend.
        mem_limits: Ordered list of memory limits present in the data.

    Returns:
        Memory limit string where crossover occurs, or None.
    """
    for mem in mem_limits:
        youmu = best.get((mem, "youmu"))
        lerobot = best.get((mem, "lerobot"))
        if youmu and lerobot and youmu["samples_per_sec"] >= lerobot["samples_per_sec"]:
            return mem
    return None


def generate_report(rows: list[dict]) -> str:
    """Generate a markdown report from combined benchmark results.

    Args:
        rows: Combined row dicts with memory_limit column.

    Returns:
        Markdown report string.
    """
    best = get_best_throughput_per_backend(rows)
    failures = get_oom_failures(rows)

    # Get ordered memory limits present in data
    mem_limits_set = {r["memory_limit"] for r in rows}
    mem_limits = sorted(mem_limits_set, key=memory_limit_sort_key)

    backends = sorted({r["backend"] for r in rows})

    lines = []
    lines.append("# Memory-Constrained Data Loading Benchmark Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("Best throughput (samples/sec) per backend at each RAM limit:")
    lines.append("")

    # Summary table header
    header = "| RAM Limit |"
    separator = "|-----------|"
    for b in backends:
        header += f" {b} (samples/s) | {b} config |"
        separator += "---:|---|"
    header += " Ratio (youmu/lerobot) |"
    separator += "---:|"
    lines.append(header)
    lines.append(separator)

    # Summary table rows
    crossover_point = find_crossover_point(best, mem_limits)
    for mem in mem_limits:
        row_str = f"| {mem} |"
        youmu_sps = None
        lerobot_sps = None
        for b in backends:
            entry = best.get((mem, b))
            fail_count = failures.get((mem, b), 0)
            if entry:
                sps = entry["samples_per_sec"]
                config = f"w={entry['num_workers']} obs={entry['obs_len']} bs={entry['batch_size']}"
                if b == "youmu":
                    youmu_sps = sps
                elif b == "lerobot":
                    lerobot_sps = sps
                suffix = f" ({fail_count} errors)" if fail_count else ""
                row_str += f" {sps:.2f}{suffix} | {config} |"
            elif fail_count:
                row_str += f" OOM ({fail_count} errors) | - |"
            else:
                row_str += " N/A | - |"

        # Ratio column
        if youmu_sps and lerobot_sps and lerobot_sps > 0:
            ratio = youmu_sps / lerobot_sps
            marker = " **crossover**" if mem == crossover_point else ""
            row_str += f" {ratio:.2f}x{marker} |"
        else:
            row_str += " N/A |"

        lines.append(row_str)

    lines.append("")

    # Crossover analysis
    lines.append("## Analysis")
    lines.append("")
    if crossover_point:
        lines.append(
            f"**Crossover point:** Youmu throughput meets or exceeds LeRobot "
            f"at **{crossover_point}** RAM limit."
        )
    else:
        lines.append(
            "**Crossover point:** No crossover detected in the tested RAM range. "
            "Youmu did not meet or exceed LeRobot throughput at any tested limit."
        )
    lines.append("")

    # OOM failure notes
    if failures:
        lines.append("## OOM / Error Notes")
        lines.append("")
        for mem in mem_limits:
            for b in backends:
                count = failures.get((mem, b), 0)
                if count:
                    lines.append(f"- **{mem}** / {b}: {count} configuration(s) had errors")
        lines.append("")

    # Full results table
    lines.append("## Full Results")
    lines.append("")
    lines.append(
        "| RAM | Backend | obs_len | workers | batch_size | samples/s | "
        "first_sample(s) | peak_rss(MB) | io_amp | error |"
    )
    lines.append(
        "|-----|---------|---------|---------|------------|----------:|"
        "----------------:|-------------:|-------:|-------|"
    )

    for row in rows:
        sps = row.get("samples_per_sec", "")
        ttf = row.get("time_to_first_sample_sec", "")
        rss = row.get("peak_rss_mb", "")
        io_amp = row.get("io_amplification_ratio", "")
        error = row.get("error", "")

        # Format numeric values
        sps_str = f"{float(sps):.2f}" if sps else "-"
        ttf_str = f"{float(ttf):.4f}" if ttf else "-"
        rss_str = f"{float(rss):.0f}" if rss else "-"
        io_amp_str = f"{float(io_amp):.2f}" if io_amp else "-"

        lines.append(
            f"| {row['memory_limit']} | {row['backend']} | "
            f"{row.get('obs_len', '')} | {row.get('num_workers', '')} | "
            f"{row.get('batch_size', '')} | {sps_str} | {ttf_str} | "
            f"{rss_str} | {io_amp_str} | {error} |"
        )

    lines.append("")
    return "\n".join(lines)


def main():
    """Entry point: parse args, aggregate CSVs, generate report."""
    parser = argparse.ArgumentParser(
        description="Aggregate memory-constrained benchmark results and generate report"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="benchmarks/memory_constrained",
        help="Directory containing per-RAM-limit CSV files (mem_*G.csv)",
    )
    parser.add_argument(
        "--combined-output",
        type=str,
        default="benchmarks/memory_constrained/combined_results.csv",
        help="Path to combined CSV output",
    )
    parser.add_argument(
        "--report-output",
        type=str,
        default="benchmarks/memory_constrained_benchmark_report.md",
        help="Path to markdown report output",
    )
    args = parser.parse_args()

    print(f"Reading CSVs from {args.input_dir}...")
    combined = aggregate_csvs(args.input_dir)
    print(f"  Found {len(combined)} total rows")

    # Save combined CSV
    save_combined_csv(combined, args.combined_output)
    print(f"  Combined CSV saved to {args.combined_output}")

    # Generate and save report
    report = generate_report(combined)
    os.makedirs(os.path.dirname(args.report_output) or ".", exist_ok=True)
    with open(args.report_output, "w") as f:
        f.write(report)
    print(f"  Report saved to {args.report_output}")


if __name__ == "__main__":
    main()
