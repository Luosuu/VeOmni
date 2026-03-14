"""Tests for read_multi_column_row_range_py with Libero dataset column types.

Validates that the Rust multi-column row-range reader works correctly with:
- LIST<FLOAT32> columns (observation.state, action)
- BYTE_ARRAY columns (observation.images.image bytes)
- Mixed column type reads in a single call
- Cross-row-group boundary reads (using synthetic multi-RG files)

US-001: Investigate Rust API readiness for multi-column row-range reads.
"""

import io
import os
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from youmu import (
    ParquetReaderCachePy,
    read_list_row_range_py,
    read_multi_column_row_range_py,
    read_row_range_py,
)
from youmu.libero_dataset import _find_physical_column_index

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATASET_DIR = "/home/ubuntu/dataset/hf_vla_64k"
DATA_FILE = os.path.join(DATASET_DIR, "data/chunk-000/file-000.parquet")
DATASET_EXISTS = os.path.exists(DATA_FILE)

skip_no_dataset = pytest.mark.skipif(
    not DATASET_EXISTS,
    reason=f"Libero dataset not found at {DATASET_DIR}",
)


def _get_column_indices():
    """Return (state_col_idx, action_col_idx, image_bytes_col_idx) for hf_vla_64k."""
    pf = pq.ParquetFile(DATA_FILE)
    schema = pf.schema_arrow
    state_idx = _find_physical_column_index(schema, "observation.state")
    action_idx = _find_physical_column_index(schema, "action")
    image_idx = _find_physical_column_index(
        schema, "observation.images.image", child_name="bytes"
    )
    return state_idx, action_idx, image_idx


def _create_multi_rg_parquet(path: str, rows_per_rg: int = 10, num_rgs: int = 3):
    """Create a synthetic Parquet file with multiple row groups and Libero-like schema.

    Each row group has ``rows_per_rg`` rows. The schema matches Libero's
    column layout: struct<bytes, path> image columns, list<float> state/action,
    and scalar columns.

    Args:
        path: Output file path.
        rows_per_rg: Number of rows per row group.
        num_rgs: Number of row groups to write.

    Returns:
        Total number of rows written.
    """
    total_rows = rows_per_rg * num_rgs

    # Generate deterministic data
    rng = np.random.RandomState(42)
    state_dim = 8
    action_dim = 7

    # Build arrays for the full table
    all_states = [rng.randn(state_dim).astype(np.float32).tolist() for _ in range(total_rows)]
    all_actions = [rng.randn(action_dim).astype(np.float32).tolist() for _ in range(total_rows)]

    # Create small PNG images as bytes
    image_bytes_list = []
    for i in range(total_rows):
        img = Image.fromarray(
            np.full((4, 4, 3), fill_value=i % 256, dtype=np.uint8)
        )
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes_list.append(buf.getvalue())

    # Build PyArrow table matching Libero schema
    image_struct = pa.StructArray.from_arrays(
        [pa.array(image_bytes_list, type=pa.binary()),
         pa.array([f"img_{i}.png" for i in range(total_rows)], type=pa.string())],
        names=["bytes", "path"],
    )
    image2_struct = pa.StructArray.from_arrays(
        [pa.array(image_bytes_list, type=pa.binary()),
         pa.array([f"img2_{i}.png" for i in range(total_rows)], type=pa.string())],
        names=["bytes", "path"],
    )
    state_array = pa.array(all_states, type=pa.list_(pa.float32()))
    action_array = pa.array(all_actions, type=pa.list_(pa.float32()))
    timestamp_array = pa.array(np.arange(total_rows, dtype=np.float32))
    frame_index_array = pa.array(np.arange(total_rows, dtype=np.int64))
    episode_index_array = pa.array(np.zeros(total_rows, dtype=np.int64))
    index_array = pa.array(np.arange(total_rows, dtype=np.int64))
    task_index_array = pa.array(np.zeros(total_rows, dtype=np.int64))

    table = pa.table({
        "observation.images.image": image_struct,
        "observation.images.image2": image2_struct,
        "observation.state": state_array,
        "action": action_array,
        "timestamp": timestamp_array,
        "frame_index": frame_index_array,
        "episode_index": episode_index_array,
        "index": index_array,
        "task_index": task_index_array,
    })

    # Write with multiple row groups and page index enabled
    writer = pq.ParquetWriter(
        path,
        table.schema,
        use_dictionary=False,
        write_page_index=True,
    )
    for rg_start in range(0, total_rows, rows_per_rg):
        rg_end = min(rg_start + rows_per_rg, total_rows)
        writer.write_table(table.slice(rg_start, rg_end - rg_start))
    writer.close()

    return total_rows


# ---------------------------------------------------------------------------
# Tests: BYTE_ARRAY columns (image bytes) — should work with multi-column API
# ---------------------------------------------------------------------------


@skip_no_dataset
class TestMultiColumnByteArray:
    """Test read_multi_column_row_range_py with BYTE_ARRAY (image) columns."""

    def test_read_single_image_column(self):
        """Reading a single BYTE_ARRAY column returns correct types and row count."""
        _, _, image_idx = _get_column_indices()
        cache = ParquetReaderCachePy()
        cache.preload([DATA_FILE])

        results = read_multi_column_row_range_py(
            DATA_FILE, 0, [image_idx], 0, 5, cache=cache
        )

        assert len(results) == 1
        arr, row_count = results[0]
        assert row_count == 5
        assert len(arr) == 5
        # BYTE_ARRAY returns BinaryArray
        assert isinstance(arr, pa.BinaryArray)

    def test_image_bytes_are_valid_png(self):
        """Each binary value from the image column is a valid PNG image."""
        _, _, image_idx = _get_column_indices()
        cache = ParquetReaderCachePy()
        cache.preload([DATA_FILE])

        results = read_multi_column_row_range_py(
            DATA_FILE, 0, [image_idx], 0, 3, cache=cache
        )
        arr, _ = results[0]

        for i in range(len(arr)):
            png_bytes = arr[i].as_py()
            assert isinstance(png_bytes, bytes)
            assert len(png_bytes) > 0
            # Verify it's a valid image
            img = Image.open(io.BytesIO(png_bytes))
            img_np = np.array(img)
            assert img_np.ndim == 3  # (H, W, C)
            assert img_np.shape[2] == 3  # RGB

    def test_image_matches_single_column_read(self):
        """Multi-column read of image bytes matches single-column read_row_range_py."""
        _, _, image_idx = _get_column_indices()
        cache = ParquetReaderCachePy()
        cache.preload([DATA_FILE])

        # Multi-column read
        multi_results = read_multi_column_row_range_py(
            DATA_FILE, 0, [image_idx], 0, 10, cache=cache
        )
        multi_arr, multi_rows = multi_results[0]

        # Single-column read
        single_arr, single_rows = read_row_range_py(
            DATA_FILE, 0, image_idx, 0, 10, cache=cache
        )

        assert multi_rows == single_rows == 10
        assert len(multi_arr) == len(single_arr) == 10
        for i in range(10):
            assert multi_arr[i].as_py() == single_arr[i].as_py()


# ---------------------------------------------------------------------------
# Tests: LIST<FLOAT32> columns — document behavior with multi-column API
# ---------------------------------------------------------------------------


@skip_no_dataset
class TestMultiColumnListFloat:
    """Test read_multi_column_row_range_py with LIST<FLOAT32> (state/action) columns.

    IMPORTANT: read_multi_column_row_range_py reads physical leaf columns.
    For list<float> columns, this returns the flat float values, NOT ListArrays.
    This means it does NOT return one list per row — it returns individual float
    values from the flattened column. For proper list-aware reads, use
    read_list_row_range_py instead.
    """

    def test_list_column_returns_flat_values(self):
        """Multi-column read of a list<float> column returns flat floats, not lists.

        This documents a known API behavior: the multi-column reader operates on
        physical leaf columns and does not reconstruct list structure.
        """
        state_idx, _, _ = _get_column_indices()
        cache = ParquetReaderCachePy()
        cache.preload([DATA_FILE])

        num_rows = 3
        results = read_multi_column_row_range_py(
            DATA_FILE, 0, [state_idx], 0, num_rows, cache=cache
        )

        arr, row_count = results[0]
        assert row_count == num_rows
        # The reader returns flat float values, not list-wrapped values.
        # For a list<float> column with 8 elements per list and 3 rows,
        # this returns only 3 float values (the first 3 leaf floats), not 24.
        assert isinstance(arr, pa.FloatArray)
        assert len(arr) == num_rows  # NOT num_rows * list_dim

    def test_flat_values_match_first_elements_of_lists(self):
        """The flat floats from multi-column read match the start of the flattened list data."""
        state_idx, _, _ = _get_column_indices()
        cache = ParquetReaderCachePy()
        cache.preload([DATA_FILE])

        num_rows = 5

        # Multi-column returns flat floats
        multi_results = read_multi_column_row_range_py(
            DATA_FILE, 0, [state_idx], 0, num_rows, cache=cache
        )
        multi_arr, _ = multi_results[0]
        multi_values = multi_arr.to_pylist()

        # List reader returns proper ListArray
        list_arr, _ = read_list_row_range_py(
            DATA_FILE, 0, state_idx, 0, num_rows, cache=cache
        )

        # Flatten all list elements in order
        all_list_floats = []
        for i in range(len(list_arr)):
            all_list_floats.extend(list_arr[i].as_py())

        # The multi-column flat values should match the first N elements
        # of the flattened list data (they're just reading the leaf float column)
        for i in range(num_rows):
            assert abs(multi_values[i] - all_list_floats[i]) < 1e-6, (
                f"Mismatch at index {i}: multi={multi_values[i]}, "
                f"list_flat={all_list_floats[i]}"
            )

    def test_list_column_read_via_proper_api(self):
        """read_list_row_range_py correctly reads list<float> columns with proper structure."""
        state_idx, action_idx, _ = _get_column_indices()
        cache = ParquetReaderCachePy()
        cache.preload([DATA_FILE])

        num_rows = 10

        # State
        state_arr, state_rows = read_list_row_range_py(
            DATA_FILE, 0, state_idx, 0, num_rows, cache=cache
        )
        assert state_rows == num_rows
        assert isinstance(state_arr, pa.ListArray)
        assert len(state_arr) == num_rows
        # Each list should have the same dimension
        state_dim = len(state_arr[0].as_py())
        for i in range(num_rows):
            assert len(state_arr[i].as_py()) == state_dim

        # Action
        action_arr, action_rows = read_list_row_range_py(
            DATA_FILE, 0, action_idx, 0, num_rows, cache=cache
        )
        assert action_rows == num_rows
        assert isinstance(action_arr, pa.ListArray)
        assert len(action_arr) == num_rows


# ---------------------------------------------------------------------------
# Tests: Mixed column types in a single call
# ---------------------------------------------------------------------------


@skip_no_dataset
class TestMultiColumnMixed:
    """Test reading mixed column types in a single multi-column call."""

    def test_read_two_list_and_one_binary(self):
        """Reading [state, action, image_bytes] in one call returns 3 results.

        NOTE: state and action return flat float values (not lists) because
        multi-column read operates on physical leaf columns.
        """
        state_idx, action_idx, image_idx = _get_column_indices()
        cache = ParquetReaderCachePy()
        cache.preload([DATA_FILE])

        num_rows = 5
        results = read_multi_column_row_range_py(
            DATA_FILE, 0, [state_idx, action_idx, image_idx],
            0, num_rows, cache=cache,
        )

        assert len(results) == 3

        # State (flat floats from list<float> leaf)
        state_arr, state_rows = results[0]
        assert state_rows == num_rows
        assert isinstance(state_arr, pa.FloatArray)

        # Action (flat floats from list<float> leaf)
        action_arr, action_rows = results[1]
        assert action_rows == num_rows
        assert isinstance(action_arr, pa.FloatArray)

        # Image bytes (binary — works correctly)
        image_arr, image_rows = results[2]
        assert image_rows == num_rows
        assert isinstance(image_arr, pa.BinaryArray)
        assert len(image_arr) == num_rows

    def test_column_order_matches_input_order(self):
        """Results are returned in the same order as the input column_indices."""
        state_idx, _, image_idx = _get_column_indices()
        cache = ParquetReaderCachePy()
        cache.preload([DATA_FILE])

        # Read image first, then state
        results = read_multi_column_row_range_py(
            DATA_FILE, 0, [image_idx, state_idx], 0, 3, cache=cache
        )

        assert len(results) == 2
        # First result should be image (binary)
        assert isinstance(results[0][0], pa.BinaryArray)
        # Second result should be state (float)
        assert isinstance(results[1][0], pa.FloatArray)


# ---------------------------------------------------------------------------
# Tests: Cross-row-group boundary reads (synthetic multi-RG file)
# ---------------------------------------------------------------------------


class TestCrossRowGroupBoundary:
    """Test behavior when reading across row group boundaries.

    Uses a synthetic Parquet file with multiple small row groups to verify
    that read_multi_column_row_range_py correctly rejects out-of-bounds reads
    (since it operates within a single row group).
    """

    @pytest.fixture(autouse=True)
    def setup_multi_rg_file(self, tmp_path):
        """Create a synthetic multi-row-group parquet file."""
        self.rg_size = 10
        self.num_rgs = 3
        self.file_path = str(tmp_path / "multi_rg.parquet")
        self.total_rows = _create_multi_rg_parquet(
            self.file_path, self.rg_size, self.num_rgs
        )
        self.cache = ParquetReaderCachePy()
        self.cache.preload([self.file_path])

        # Discover column indices from the synthetic file
        pf = pq.ParquetFile(self.file_path)
        schema = pf.schema_arrow
        self.state_idx = _find_physical_column_index(schema, "observation.state")
        self.image_idx = _find_physical_column_index(
            schema, "observation.images.image", child_name="bytes"
        )

    def test_file_has_multiple_row_groups(self):
        """Verify the fixture file was created with the expected row group count."""
        pf = pq.ParquetFile(self.file_path)
        assert pf.metadata.num_row_groups == self.num_rgs
        for i in range(self.num_rgs):
            assert pf.metadata.row_group(i).num_rows == self.rg_size

    def test_read_within_single_rg(self):
        """Reading within a single row group works correctly for BYTE_ARRAY."""
        results = read_multi_column_row_range_py(
            self.file_path, 0, [self.image_idx], 0, 5, cache=self.cache
        )
        arr, row_count = results[0]
        assert row_count == 5
        assert len(arr) == 5
        assert isinstance(arr, pa.BinaryArray)

    def test_read_full_rg(self):
        """Reading all rows in a row group works correctly."""
        results = read_multi_column_row_range_py(
            self.file_path, 1, [self.image_idx], 0, self.rg_size, cache=self.cache
        )
        arr, row_count = results[0]
        assert row_count == self.rg_size
        assert len(arr) == self.rg_size

    def test_read_from_different_rgs_gives_different_data(self):
        """Data from different row groups is actually different."""
        # Read from RG 0
        results_rg0 = read_multi_column_row_range_py(
            self.file_path, 0, [self.image_idx], 0, 5, cache=self.cache
        )
        # Read from RG 1
        results_rg1 = read_multi_column_row_range_py(
            self.file_path, 1, [self.image_idx], 0, 5, cache=self.cache
        )

        arr0 = results_rg0[0][0]
        arr1 = results_rg1[0][0]

        # Image bytes should differ between row groups (different fill values)
        differs = False
        for i in range(5):
            if arr0[i].as_py() != arr1[i].as_py():
                differs = True
                break
        assert differs, "Data from different row groups should not be identical"

    def test_end_row_exceeds_rg_size_raises_error(self):
        """Reading past row group boundary raises an error.

        read_multi_column_row_range_py operates within a single row group.
        It does NOT support spans across row groups — callers must handle
        cross-RG reads by splitting into per-RG calls.
        """
        with pytest.raises(Exception):
            read_multi_column_row_range_py(
                self.file_path, 0, [self.image_idx],
                0, self.rg_size + 5,  # exceeds RG 0 size
                cache=self.cache,
            )

    def test_multi_column_mixed_types_across_rgs(self):
        """Reading multiple column types from different row groups works for flat types."""
        for rg_idx in range(self.num_rgs):
            results = read_multi_column_row_range_py(
                self.file_path, rg_idx, [self.image_idx], 0, 3, cache=self.cache
            )
            arr, row_count = results[0]
            assert row_count == 3
            assert isinstance(arr, pa.BinaryArray)
            assert len(arr) == 3

    def test_cached_vs_uncached_reads_match(self):
        """Cached and uncached multi-column reads return identical data."""
        col_indices = [self.image_idx]

        # With cache
        cached_results = read_multi_column_row_range_py(
            self.file_path, 0, col_indices, 0, 5, cache=self.cache
        )

        # Without cache
        uncached_results = read_multi_column_row_range_py(
            self.file_path, 0, col_indices, 0, 5, cache=None
        )

        assert len(cached_results) == len(uncached_results)
        for (c_arr, c_rows), (u_arr, u_rows) in zip(cached_results, uncached_results):
            assert c_rows == u_rows
            assert len(c_arr) == len(u_arr)
            for i in range(len(c_arr)):
                assert c_arr[i].as_py() == u_arr[i].as_py()


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


@skip_no_dataset
class TestEdgeCases:
    """Edge case tests for read_multi_column_row_range_py."""

    def test_empty_column_indices(self):
        """Empty column_indices returns empty result list."""
        cache = ParquetReaderCachePy()
        cache.preload([DATA_FILE])

        results = read_multi_column_row_range_py(
            DATA_FILE, 0, [], 0, 5, cache=cache
        )
        assert results == []

    def test_single_row_read(self):
        """Reading exactly one row works correctly."""
        _, _, image_idx = _get_column_indices()
        cache = ParquetReaderCachePy()
        cache.preload([DATA_FILE])

        results = read_multi_column_row_range_py(
            DATA_FILE, 0, [image_idx], 0, 1, cache=cache
        )
        arr, row_count = results[0]
        assert row_count == 1
        assert len(arr) == 1

    def test_invalid_row_range_raises_error(self):
        """start_row >= end_row raises an error."""
        _, _, image_idx = _get_column_indices()
        cache = ParquetReaderCachePy()
        cache.preload([DATA_FILE])

        with pytest.raises(Exception):
            read_multi_column_row_range_py(
                DATA_FILE, 0, [image_idx], 5, 5, cache=cache
            )

        with pytest.raises(Exception):
            read_multi_column_row_range_py(
                DATA_FILE, 0, [image_idx], 10, 5, cache=cache
            )

    def test_duplicate_column_indices(self):
        """Reading the same column twice returns two identical results."""
        _, _, image_idx = _get_column_indices()
        cache = ParquetReaderCachePy()
        cache.preload([DATA_FILE])

        results = read_multi_column_row_range_py(
            DATA_FILE, 0, [image_idx, image_idx], 0, 3, cache=cache
        )
        assert len(results) == 2
        arr0, rows0 = results[0]
        arr1, rows1 = results[1]
        assert rows0 == rows1
        for i in range(3):
            assert arr0[i].as_py() == arr1[i].as_py()
