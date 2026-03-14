#!/usr/bin/env bash
# Tests for benchmark_memory_sweep.sh
#
# Verify CLI behavior: help output, unknown args, and script structure.
#
# Usage:
#   bash tests/test_benchmark_memory_sweep.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_UNDER_TEST="${SCRIPT_DIR}/../scripts/benchmark_memory_sweep.sh"

PASS=0
FAIL=0

# Helper: run a test that checks output contains a string
run_test_output_contains() {
    local name="$1"
    local expected_exit="$2"
    local expected_string="$3"
    shift 3
    local cmd=("$@")

    local output
    local actual_exit
    output=$("${cmd[@]}" 2>&1) && actual_exit=0 || actual_exit=$?

    if [[ "${actual_exit}" -ne "${expected_exit}" ]]; then
        echo "  FAIL: ${name} (expected exit=${expected_exit}, got exit=${actual_exit})"
        echo "        output: ${output}"
        FAIL=$((FAIL + 1))
        return
    fi

    if echo "${output}" | grep -qF -- "${expected_string}"; then
        echo "  PASS: ${name}"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: ${name} (output missing: '${expected_string}')"
        echo "        output: ${output}"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== Tests for benchmark_memory_sweep.sh ==="
echo ""

# --- Test: --help shows usage and exits 0 ---
run_test_output_contains \
    "--help shows usage" \
    0 \
    "Usage:" \
    bash "${SCRIPT_UNDER_TEST}" --help

# --- Test: --help mentions RAM limits ---
run_test_output_contains \
    "--help mentions RAM limits" \
    0 \
    "2G" \
    bash "${SCRIPT_UNDER_TEST}" --help

# --- Test: --help mentions backends ---
run_test_output_contains \
    "--help mentions backends" \
    0 \
    "youmu" \
    bash "${SCRIPT_UNDER_TEST}" --help

# --- Test: --help mentions youmu_page_aligned ---
run_test_output_contains \
    "--help mentions youmu_page_aligned" \
    0 \
    "youmu_page_aligned" \
    bash "${SCRIPT_UNDER_TEST}" --help

# --- Test: --help mentions iterations ---
run_test_output_contains \
    "--help mentions iterations" \
    0 \
    "50" \
    bash "${SCRIPT_UNDER_TEST}" --help

# --- Test: unknown argument exits with error ---
run_test_output_contains \
    "unknown argument exits with error" \
    1 \
    "unknown argument" \
    bash "${SCRIPT_UNDER_TEST}" --bogus-flag

# --- Test: script is executable ---
if [[ -x "${SCRIPT_UNDER_TEST}" ]]; then
    echo "  PASS: script is executable"
    PASS=$((PASS + 1))
else
    echo "  FAIL: script is not executable"
    FAIL=$((FAIL + 1))
fi

# --- Test: script references benchmark_memory_constrained.sh ---
if grep -qF "benchmark_memory_constrained.sh" "${SCRIPT_UNDER_TEST}"; then
    echo "  PASS: references benchmark_memory_constrained.sh"
    PASS=$((PASS + 1))
else
    echo "  FAIL: does not reference benchmark_memory_constrained.sh"
    FAIL=$((FAIL + 1))
fi

# --- Test: script handles OOM exit code 137 ---
if grep -qF "137" "${SCRIPT_UNDER_TEST}"; then
    echo "  PASS: handles OOM exit code 137"
    PASS=$((PASS + 1))
else
    echo "  FAIL: does not handle OOM exit code 137"
    FAIL=$((FAIL + 1))
fi

# --- Test: output dir is benchmarks/memory_constrained ---
if grep -qF "benchmarks/memory_constrained" "${SCRIPT_UNDER_TEST}"; then
    echo "  PASS: output dir is benchmarks/memory_constrained"
    PASS=$((PASS + 1))
else
    echo "  FAIL: output dir is not benchmarks/memory_constrained"
    FAIL=$((FAIL + 1))
fi

# --- Summary ---
echo ""
echo "=== Results: ${PASS} passed, ${FAIL} failed ==="

if [[ "${FAIL}" -gt 0 ]]; then
    exit 1
fi
