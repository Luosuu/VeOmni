#!/usr/bin/env bash
# benchmark_memory_constrained.sh - Run data loading benchmark under memory-limited cgroup.
#
# Uses systemd-run to enforce a hard RAM limit (no swap) via cgroup v2,
# drops page cache before the run, then delegates to benchmark_data_loading.py.
#
# Usage:
#   ./scripts/benchmark_memory_constrained.sh --memory-limit 4G [benchmark args...]
#
# Examples:
#   ./scripts/benchmark_memory_constrained.sh --memory-limit 4G --quick
#   ./scripts/benchmark_memory_constrained.sh --memory-limit 2G --backends youmu --output out.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_SCRIPT="${SCRIPT_DIR}/benchmark_data_loading.py"
VENV_ACTIVATE="${SCRIPT_DIR}/../.venv/bin/activate"

usage() {
    cat <<EOF
Usage: $(basename "$0") --memory-limit <LIMIT> [benchmark_data_loading.py args...]

Run the data loading benchmark inside a memory-limited cgroup.

Required:
  --memory-limit <LIMIT>   RAM limit (e.g. 1G, 2G, 4G, 8G, 16G, 32G)

Optional:
  --help                   Show this help message

All remaining arguments are passed through to benchmark_data_loading.py.

Examples:
  $(basename "$0") --memory-limit 4G --quick
  $(basename "$0") --memory-limit 2G --backends youmu lerobot --output results.csv
EOF
}

# --- Parse --memory-limit and --help from args, pass rest through ---
MEMORY_LIMIT=""
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help)
            usage
            exit 0
            ;;
        --memory-limit)
            if [[ $# -lt 2 ]]; then
                echo "Error: --memory-limit requires a value (e.g. 4G)" >&2
                exit 1
            fi
            MEMORY_LIMIT="$2"
            shift 2
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

# Require --memory-limit
if [[ -z "${MEMORY_LIMIT}" ]]; then
    echo "Error: --memory-limit is required." >&2
    echo "Run with --help for usage information." >&2
    exit 1
fi

# Check sudo access
if ! sudo -n true 2>/dev/null; then
    echo "Error: sudo access required (for systemd-run and page cache drop)." >&2
    exit 1
fi

# Check benchmark script exists
if [[ ! -f "${BENCHMARK_SCRIPT}" ]]; then
    echo "Error: benchmark script not found: ${BENCHMARK_SCRIPT}" >&2
    exit 1
fi

# Drop page cache
echo "Dropping page cache..."
sudo sync
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
echo "Page cache dropped."

# Print configuration
echo ""
echo "=========================================="
echo "Memory-Constrained Benchmark"
echo "=========================================="
echo "  Memory limit:  ${MEMORY_LIMIT}"
echo "  Swap:          disabled (MemorySwapMax=0)"
echo "  Benchmark:     ${BENCHMARK_SCRIPT}"
echo "  Extra args:    ${PASSTHROUGH_ARGS[*]:-<none>}"
echo "=========================================="
echo ""

# Run benchmark inside memory-limited cgroup
echo "Starting benchmark with MemoryMax=${MEMORY_LIMIT}, MemorySwapMax=0..."
sudo systemd-run --scope \
    -p "MemoryMax=${MEMORY_LIMIT}" \
    -p "MemorySwapMax=0" \
    -- bash -c ". '${VENV_ACTIVATE}' && python '${BENCHMARK_SCRIPT}' ${PASSTHROUGH_ARGS[*]:-}"

echo ""
echo "Benchmark completed successfully under ${MEMORY_LIMIT} memory limit."
