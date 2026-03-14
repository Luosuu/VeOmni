#!/usr/bin/env bash
# Tests for benchmark_memory_constrained.sh
#
# These tests verify the CLI behavior of the wrapper script:
# argument parsing, help output, error messages, and exit codes.
#
# Usage:
#   bash tests/test_benchmark_memory_constrained.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_UNDER_TEST="${SCRIPT_DIR}/../scripts/benchmark_memory_constrained.sh"

PASS=0
FAIL=0

# Helper: run a test case
run_test() {
    local name="$1"
    local expected_exit="$2"
    shift 2
    local cmd=("$@")

    # Capture output and exit code
    local output
    local actual_exit
    output=$("${cmd[@]}" 2>&1) && actual_exit=0 || actual_exit=$?

    if [[ "${actual_exit}" -eq "${expected_exit}" ]]; then
        echo "  PASS: ${name}"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: ${name} (expected exit=${expected_exit}, got exit=${actual_exit})"
        echo "        output: ${output}"
        FAIL=$((FAIL + 1))
    fi
}

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

echo "=== Tests for benchmark_memory_constrained.sh ==="
echo ""

# --- Test: --help shows usage and exits 0 ---
run_test_output_contains \
    "--help shows usage" \
    0 \
    "Usage:" \
    bash "${SCRIPT_UNDER_TEST}" --help

# --- Test: --help shows --memory-limit in usage ---
run_test_output_contains \
    "--help mentions --memory-limit" \
    0 \
    "--memory-limit" \
    bash "${SCRIPT_UNDER_TEST}" --help

# --- Test: no args exits with error ---
run_test_output_contains \
    "no args exits with error" \
    1 \
    "--memory-limit is required" \
    bash "${SCRIPT_UNDER_TEST}"

# --- Test: --memory-limit without value exits with error ---
run_test_output_contains \
    "--memory-limit without value exits with error" \
    1 \
    "requires a value" \
    bash "${SCRIPT_UNDER_TEST}" --memory-limit

# --- Summary ---
echo ""
echo "=== Results: ${PASS} passed, ${FAIL} failed ==="

if [[ "${FAIL}" -gt 0 ]]; then
    exit 1
fi
