#!/usr/bin/env bash
# benchmark_memory_sweep.sh - Run data loading benchmark across multiple RAM limits.
#
# Iterates over RAM limits [1G, 2G, 4G, 8G, 16G, 32G], running the benchmark
# for each limit via benchmark_memory_constrained.sh. Each backend is run
# separately so an OOM in one doesn't skip the other.
#
# Results are saved to benchmarks/memory_constrained/mem_<limit>.csv.
#
# Usage:
#   ./scripts/benchmark_memory_sweep.sh
#   ./scripts/benchmark_memory_sweep.sh --help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONSTRAINED_SCRIPT="${SCRIPT_DIR}/benchmark_memory_constrained.sh"
OUTPUT_DIR="benchmarks/memory_constrained"

# Sweep parameters
RAM_LIMITS=(1G 2G 4G 8G 16G 32G)
BACKENDS=(youmu lerobot)
NUM_WORKERS="0 2 4 8"
OBS_LENS="1 2 4"
BATCH_SIZES="4 8"
NUM_ITERATIONS=50
WARMUP_ITERATIONS=5

usage() {
    cat <<EOF
Usage: $(basename "$0") [--help]

Run the data loading benchmark across all RAM limits and configurations.

RAM limits: ${RAM_LIMITS[*]}
Backends:   ${BACKENDS[*]}
Workers:    ${NUM_WORKERS}
Obs lens:   ${OBS_LENS}
Batch sizes: ${BATCH_SIZES}
Iterations: ${NUM_ITERATIONS} (warmup=${WARMUP_ITERATIONS})

Results are saved to ${OUTPUT_DIR}/mem_<limit>.csv.
EOF
}

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown argument '$1'" >&2
            echo "Run with --help for usage information." >&2
            exit 1
            ;;
    esac
done

# Check constrained script exists
if [[ ! -f "${CONSTRAINED_SCRIPT}" ]]; then
    echo "Error: constrained benchmark script not found: ${CONSTRAINED_SCRIPT}" >&2
    exit 1
fi

# Check sudo access
if ! sudo -n true 2>/dev/null; then
    echo "Error: sudo access required (for systemd-run and page cache drop)." >&2
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Track overall timing and results
SWEEP_START=$(date +%s)
TOTAL_RUNS=0
SUCCESSFUL_RUNS=0
OOM_RUNS=0
FAILED_RUNS=0

echo "=========================================="
echo "Memory-Constrained Benchmark Sweep"
echo "=========================================="
echo "  RAM limits:   ${RAM_LIMITS[*]}"
echo "  Backends:     ${BACKENDS[*]}"
echo "  Workers:      ${NUM_WORKERS}"
echo "  Obs lens:     ${OBS_LENS}"
echo "  Batch sizes:  ${BATCH_SIZES}"
echo "  Iterations:   ${NUM_ITERATIONS} (warmup=${WARMUP_ITERATIONS})"
echo "  Output dir:   ${OUTPUT_DIR}"
echo "=========================================="
echo ""

for mem_limit in "${RAM_LIMITS[@]}"; do
    echo ""
    echo "######################################################################"
    echo "# RAM Limit: ${mem_limit}"
    echo "######################################################################"
    echo ""

    output_file="${OUTPUT_DIR}/mem_${mem_limit}.csv"
    first_backend=true

    for backend in "${BACKENDS[@]}"; do
        echo "--- Backend: ${backend} (memory limit: ${mem_limit}) ---"

        TOTAL_RUNS=$((TOTAL_RUNS + 1))

        # Build the benchmark_data_loading.py args
        # Use a temporary output file per backend, then merge
        tmp_output="${OUTPUT_DIR}/.tmp_${mem_limit}_${backend}.csv"

        # Run the constrained benchmark for this backend.
        # benchmark_data_loading.py accepts --num-workers, --obs-lens,
        # --batch-sizes as nargs="+" (space-separated values).
        exit_code=0
        bash "${CONSTRAINED_SCRIPT}" \
            --memory-limit "${mem_limit}" \
            --backends "${backend}" \
            --output "${tmp_output}" \
            --num-workers ${NUM_WORKERS} \
            --obs-lens ${OBS_LENS} \
            --batch-sizes ${BATCH_SIZES} \
            --iterations ${NUM_ITERATIONS} \
            --warmup ${WARMUP_ITERATIONS} \
            || exit_code=$?

        if [[ ${exit_code} -eq 137 ]]; then
            echo ""
            echo "WARNING: ${backend} was OOM-killed at ${mem_limit} (exit code 137)"
            echo ""
            OOM_RUNS=$((OOM_RUNS + 1))
        elif [[ ${exit_code} -ne 0 ]]; then
            echo ""
            echo "WARNING: ${backend} failed at ${mem_limit} (exit code ${exit_code})"
            echo ""
            FAILED_RUNS=$((FAILED_RUNS + 1))
        else
            SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
        fi

        # Merge tmp output into the per-memory-limit CSV
        if [[ -f "${tmp_output}" ]]; then
            if [[ "${first_backend}" == true ]]; then
                # First backend: copy with header
                cp "${tmp_output}" "${output_file}"
                first_backend=false
            else
                # Subsequent backends: append without header
                tail -n +2 "${tmp_output}" >> "${output_file}"
            fi
            rm -f "${tmp_output}"
        fi
    done

    echo ""
    echo "Results for ${mem_limit} saved to ${output_file}"
done

# Print summary
SWEEP_END=$(date +%s)
ELAPSED=$((SWEEP_END - SWEEP_START))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo ""
echo "=========================================="
echo "Sweep Complete"
echo "=========================================="
echo "  Elapsed time:    ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "  Total runs:      ${TOTAL_RUNS}"
echo "  Successful:      ${SUCCESSFUL_RUNS}"
echo "  OOM killed:      ${OOM_RUNS}"
echo "  Other failures:  ${FAILED_RUNS}"
echo "  Results dir:     ${OUTPUT_DIR}/"
echo "=========================================="
