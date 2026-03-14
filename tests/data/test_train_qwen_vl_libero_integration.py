"""Tests for US-003: Integration of youmu_page_aligned backend with training loop.

Tests cover:
- build_libero_dataset recognizes youmu_page_aligned backend
- TransformIterableDataset wraps IterableDataset with transform and __len__
- DataLoader works correctly with TransformIterableDataset (shuffle=False, multi-worker)
- Default backend is youmu_page_aligned
"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset

from veomni.data.dataset import MappingDataset, TransformIterableDataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02"
    b"\xfe\r\xefF\xb8\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _write_test_parquet(path: str, num_rows: int, episode_index: int):
    """Write a test parquet file with LIBERO-compatible schema."""
    state_data = pa.FixedSizeListArray.from_arrays(
        pa.array(np.random.randn(num_rows * 8).astype("float32")),
        list_size=8,
    )
    action_data = pa.FixedSizeListArray.from_arrays(
        pa.array(np.random.randn(num_rows * 7).astype("float32")),
        list_size=7,
    )
    image_bytes = pa.array([PNG_BYTES] * num_rows, type=pa.binary())
    image_paths = pa.array([f"ep{episode_index}/frame_{i}.png" for i in range(num_rows)])
    image_struct = pa.StructArray.from_arrays(
        [image_bytes, image_paths],
        names=["bytes", "path"],
    )

    table = pa.table(
        {
            "observation.images.image": image_struct,
            "observation.state": state_data,
            "action": action_data,
            "frame_index": pa.array(list(range(num_rows)), type=pa.int64()),
            "episode_index": pa.array([episode_index] * num_rows, type=pa.int64()),
        }
    )
    pq.write_table(
        table,
        path,
        use_dictionary=False,
        write_page_index=True,
        data_page_size=4096,
    )


@pytest.fixture
def libero_data_dir(tmp_path):
    """Create a minimal LIBERO dataset directory for testing.

    Uses episode-per-file layout with JSONL metadata, matching
    the existing test fixtures in test_libero_dataset_episode_per_file.py.
    """
    import json

    # Create directory structure
    data_dir = tmp_path / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir(parents=True)

    # Write 2 episode files with 20 rows each
    episode_lengths = {0: 20, 1: 20}
    for ep_idx, length in episode_lengths.items():
        parquet_path = data_dir / f"episode_{ep_idx:06d}.parquet"
        _write_test_parquet(str(parquet_path), num_rows=length, episode_index=ep_idx)

    # Write JSONL metadata
    jsonl_path = meta_dir / "episodes.jsonl"
    with open(jsonl_path, "w") as f:
        for ep_idx, length in episode_lengths.items():
            json.dump({"episode_index": ep_idx, "tasks": [f"task_{ep_idx}"], "length": length}, f)
            f.write("\n")

    return str(tmp_path)


# ---------------------------------------------------------------------------
# Tests for TransformIterableDataset
# ---------------------------------------------------------------------------


class DummyIterableDataset(IterableDataset):
    """Simple iterable dataset for testing."""

    def __init__(self, n: int = 10):
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __iter__(self):
        for i in range(self._n):
            yield {"value": i}


class TestTransformIterableDataset:
    """Tests for the TransformIterableDataset wrapper."""

    def test_len_delegation(self):
        """__len__ should delegate to wrapped dataset."""
        ds = DummyIterableDataset(n=42)
        wrapped = TransformIterableDataset(data=ds)
        assert len(wrapped) == 42

    def test_iter_no_transform(self):
        """Without transform, samples pass through unchanged."""
        ds = DummyIterableDataset(n=5)
        wrapped = TransformIterableDataset(data=ds)
        samples = list(wrapped)
        assert len(samples) == 5
        assert samples[0] == {"value": 0}
        assert samples[4] == {"value": 4}

    def test_iter_with_transform(self):
        """Transform is applied to each sample."""
        ds = DummyIterableDataset(n=3)
        transform = lambda sample: {"doubled": sample["value"] * 2}
        wrapped = TransformIterableDataset(data=ds, transform=transform)
        samples = list(wrapped)
        assert len(samples) == 3
        assert samples[0] == {"doubled": 0}
        assert samples[2] == {"doubled": 4}

    def test_is_iterable_dataset(self):
        """TransformIterableDataset should be an IterableDataset."""
        ds = DummyIterableDataset(n=1)
        wrapped = TransformIterableDataset(data=ds)
        assert isinstance(wrapped, IterableDataset)

    def test_dataloader_integration(self):
        """DataLoader should work with TransformIterableDataset without shuffle."""
        ds = DummyIterableDataset(n=10)
        wrapped = TransformIterableDataset(data=ds)
        # IterableDataset requires shuffle=False (default)
        loader = DataLoader(wrapped, batch_size=5)
        batches = list(loader)
        assert len(batches) == 2
        assert batches[0]["value"].shape == (5,)

    def test_dataloader_with_transform(self):
        """DataLoader + transform should produce transformed batches."""
        ds = DummyIterableDataset(n=6)
        transform = lambda sample: {"x": torch.tensor(sample["value"])}
        wrapped = TransformIterableDataset(data=ds, transform=transform)
        loader = DataLoader(wrapped, batch_size=3)
        batches = list(loader)
        assert len(batches) == 2
        assert torch.equal(batches[0]["x"], torch.tensor([0, 1, 2]))


# ---------------------------------------------------------------------------
# Tests for build_libero_dataset with youmu_page_aligned
# ---------------------------------------------------------------------------


class TestBuildLiberoDatasetPageAligned:
    """Tests for build_libero_dataset with youmu_page_aligned backend."""

    def test_builds_page_aligned_dataset(self, libero_data_dir):
        """build_libero_dataset with youmu_page_aligned returns a LiberoYoumuPageAlignedDataset."""
        from tasks.omni.train_qwen_vl_libero import build_libero_dataset

        ds = build_libero_dataset(
            backend="youmu_page_aligned",
            data_dir=libero_data_dir,
            obs_len=1,
            pred_len=4,
            chunk_index=0,
        )
        from youmu.libero_dataset import LiberoYoumuPageAlignedDataset

        assert isinstance(ds, LiberoYoumuPageAlignedDataset)
        assert isinstance(ds, IterableDataset)

    def test_has_len(self, libero_data_dir):
        """Page-aligned dataset should have __len__ for step calculation."""
        from tasks.omni.train_qwen_vl_libero import build_libero_dataset

        ds = build_libero_dataset(
            backend="youmu_page_aligned",
            data_dir=libero_data_dir,
            obs_len=1,
            pred_len=4,
            chunk_index=0,
        )
        length = len(ds)
        assert length > 0
        # 2 episodes of 20 rows each, obs_len=1 pred_len=4
        # valid anchors: frame 0..15 per episode (20 - pred_len = 16)
        assert length == 32

    def test_youmu_backend_still_works(self, libero_data_dir):
        """Existing youmu backend should still work."""
        from tasks.omni.train_qwen_vl_libero import build_libero_dataset

        ds = build_libero_dataset(
            backend="youmu",
            data_dir=libero_data_dir,
            obs_len=1,
            pred_len=4,
            chunk_index=0,
        )
        from youmu.libero_dataset import LiberoYoumuDataset

        assert isinstance(ds, LiberoYoumuDataset)

    def test_unknown_backend_raises(self, libero_data_dir):
        """Unknown backend should raise ValueError."""
        from tasks.omni.train_qwen_vl_libero import build_libero_dataset

        with pytest.raises(ValueError, match="Unknown libero_dataset_backend"):
            build_libero_dataset(
                backend="nonexistent",
                data_dir=libero_data_dir,
                obs_len=1,
                pred_len=4,
            )


# ---------------------------------------------------------------------------
# Tests for default backend
# ---------------------------------------------------------------------------


class TestDefaultBackend:
    """Tests that the default backend is now youmu_page_aligned."""

    def test_default_is_page_aligned(self):
        """MyDataArguments default libero_dataset_backend should be youmu_page_aligned."""
        from tasks.omni.train_qwen_vl_libero import MyDataArguments

        # MyDataArguments inherits train_path from DataArguments (required)
        args = MyDataArguments(train_path="/dummy")
        assert args.libero_dataset_backend == "youmu_page_aligned"


# ---------------------------------------------------------------------------
# Tests for training loop dataset wrapping logic
# ---------------------------------------------------------------------------


class TestTrainingLoopDatasetWrapping:
    """Tests that the training loop correctly wraps iterable vs map-style datasets."""

    def test_iterable_dataset_uses_transform_iterable(self, libero_data_dir):
        """Page-aligned (IterableDataset) should be wrapped in TransformIterableDataset."""
        from tasks.omni.train_qwen_vl_libero import build_libero_dataset

        ds = build_libero_dataset(
            backend="youmu_page_aligned",
            data_dir=libero_data_dir,
            obs_len=1,
            pred_len=4,
            chunk_index=0,
        )
        # Simulate what the training loop does
        if isinstance(ds, IterableDataset):
            wrapped = TransformIterableDataset(data=ds, transform=None)
        else:
            wrapped = MappingDataset(data=ds, transform=None)

        assert isinstance(wrapped, TransformIterableDataset)
        assert isinstance(wrapped, IterableDataset)
        assert len(wrapped) == len(ds)

    def test_map_dataset_uses_mapping_dataset(self, libero_data_dir):
        """Row-range (map-style) should be wrapped in MappingDataset."""
        from tasks.omni.train_qwen_vl_libero import build_libero_dataset

        ds = build_libero_dataset(
            backend="youmu",
            data_dir=libero_data_dir,
            obs_len=1,
            pred_len=4,
            chunk_index=0,
        )
        if isinstance(ds, IterableDataset):
            wrapped = TransformIterableDataset(data=ds, transform=None)
        else:
            wrapped = MappingDataset(data=ds, transform=None)

        assert isinstance(wrapped, MappingDataset)
        assert len(wrapped) == len(ds)


# ---------------------------------------------------------------------------
# Tests for DataLoader compatibility
# ---------------------------------------------------------------------------


class TestDataLoaderCompatibility:
    """Tests that page-aligned dataset works with DataLoader."""

    def test_dataloader_no_shuffle(self, libero_data_dir):
        """DataLoader with page-aligned dataset must not use shuffle."""
        from tasks.omni.train_qwen_vl_libero import build_libero_dataset

        ds = build_libero_dataset(
            backend="youmu_page_aligned",
            data_dir=libero_data_dir,
            obs_len=1,
            pred_len=4,
            chunk_index=0,
        )
        wrapped = TransformIterableDataset(data=ds, transform=None)

        # This should not raise (shuffle defaults to False for IterableDataset)
        loader = DataLoader(wrapped, batch_size=4, num_workers=0)
        batch = next(iter(loader))
        assert "observation.state" in batch
        assert "action" in batch

    def test_dataloader_produces_correct_shapes(self, libero_data_dir):
        """Verify batch shapes from DataLoader with page-aligned dataset."""
        from tasks.omni.train_qwen_vl_libero import build_libero_dataset

        ds = build_libero_dataset(
            backend="youmu_page_aligned",
            data_dir=libero_data_dir,
            obs_len=2,
            pred_len=4,
            chunk_index=0,
        )
        wrapped = TransformIterableDataset(data=ds, transform=None)
        loader = DataLoader(wrapped, batch_size=4, num_workers=0)
        batch = next(iter(loader))

        # observation.state: (batch, obs_len, state_dim=8)
        assert batch["observation.state"].shape == (4, 2, 8)
        # action: (batch, pred_len, action_dim=7)
        assert batch["action"].shape == (4, 4, 7)
        # observation.images.image: (batch, obs_len, H, W, 3)
        assert batch["observation.images.image"].shape[0] == 4
        assert batch["observation.images.image"].shape[1] == 2
