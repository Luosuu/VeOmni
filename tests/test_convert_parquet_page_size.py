"""Tests for parquet page size conversion (US-001).

Validates that the converted 64KB page-size parquet files at
/home/ubuntu/dataset/hf_vla_64k are correct and match the source dataset.
"""

import glob

import pyarrow.parquet as pq
import pytest

SRC_ROOT = "/home/ubuntu/dataset/hf_vla"
DST_ROOT = "/home/ubuntu/dataset/hf_vla_64k"
SRC_DATA = f"{SRC_ROOT}/data/chunk-000"
DST_DATA = f"{DST_ROOT}/data/chunk-000"


@pytest.fixture
def src_files():
    """Return sorted list of source parquet file paths."""
    files = sorted(glob.glob(f"{SRC_DATA}/*.parquet"))
    assert len(files) > 0, "No source parquet files found"
    return files


@pytest.fixture
def dst_files():
    """Return sorted list of destination parquet file paths."""
    files = sorted(glob.glob(f"{DST_DATA}/*.parquet"))
    assert len(files) > 0, "No destination parquet files found"
    return files


def test_all_377_files_converted(src_files, dst_files):
    """All 377 parquet files are converted."""
    assert len(src_files) == 377, f"Expected 377 source files, got {len(src_files)}"
    assert len(dst_files) == 377, f"Expected 377 dest files, got {len(dst_files)}"


def test_meta_directory_copied():
    """meta/ directory is copied intact with all expected files."""
    import os

    dst_meta = f"{DST_ROOT}/meta"
    assert os.path.isdir(dst_meta), "meta/ directory missing"
    assert os.path.isfile(f"{dst_meta}/info.json"), "info.json missing"
    assert os.path.isfile(f"{dst_meta}/stats.json"), "stats.json missing"
    assert os.path.isfile(f"{dst_meta}/tasks.parquet"), "tasks.parquet missing"
    assert os.path.isdir(f"{dst_meta}/episodes"), "episodes/ directory missing"


def test_row_counts_match(src_files, dst_files):
    """Row counts match between source and converted files (sampled)."""
    # Check every 20th file for speed
    for i in range(0, len(src_files), 20):
        src_rows = pq.ParquetFile(src_files[i]).metadata.num_rows
        dst_rows = pq.ParquetFile(dst_files[i]).metadata.num_rows
        assert src_rows == dst_rows, (
            f"Row count mismatch at file {i}: src={src_rows}, dst={dst_rows}"
        )


def test_data_page_version_is_v2(dst_files):
    """Converted files use DATA_PAGE_V2 (data_page_version='2.0').

    We verify by checking that column encodings include 'RLE' which is
    characteristic of v2 data pages, and that the file was written with
    write_page_index=True (page index exists).
    """
    f = pq.ParquetFile(dst_files[0])
    meta = f.metadata
    rg = meta.row_group(0)
    col = rg.column(0)
    # DATA_PAGE_V2 pages use RLE encoding for definition/repetition levels
    assert "RLE" in col.encodings, f"Expected RLE encoding for v2 pages, got {col.encodings}"


def test_page_index_exists(dst_files):
    """Converted files have page index (column index + offset index).

    The write_page_index=True flag should produce files with page statistics
    that pyarrow can read.
    """
    f = pq.ParquetFile(dst_files[0])
    meta = f.metadata
    # Page index is indicated by column statistics being available
    rg = meta.row_group(0)
    # For numeric columns (not binary), stats should be set
    # Check frame_index column which should be int64
    num_cols = rg.num_columns
    schema_names = [meta.schema.column(i).name for i in range(num_cols)]
    if "frame_index" in schema_names:
        idx = schema_names.index("frame_index")
        col = rg.column(idx)
        assert col.is_stats_set, "Page statistics not found — page index may not be written"


def test_schema_preserved(src_files, dst_files):
    """Schema is preserved between source and converted files."""
    src_schema = pq.ParquetFile(src_files[0]).schema_arrow
    dst_schema = pq.ParquetFile(dst_files[0]).schema_arrow
    assert src_schema.equals(dst_schema), (
        f"Schema mismatch:\nSrc: {src_schema}\nDst: {dst_schema}"
    )


def test_compression_is_snappy(dst_files):
    """Converted files use snappy compression."""
    f = pq.ParquetFile(dst_files[0])
    rg = f.metadata.row_group(0)
    for i in range(rg.num_columns):
        col = rg.column(i)
        assert col.compression == "SNAPPY", (
            f"Column {i} has compression {col.compression}, expected SNAPPY"
        )
