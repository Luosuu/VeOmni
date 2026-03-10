"""Tests for LiberoYoumuDataset validation with the converted 64KB dataset.

Tests verify dataset instantiation, sample structure, shapes, dtypes,
and windowed access for multiple obs_len values.
"""

import random

import pytest
import torch

from youmu.libero_dataset import LiberoYoumuDataset

DATA_DIR = "/home/ubuntu/dataset/hf_vla_64k"
PRED_LEN = 4
EXPECTED_KEYS = {
    "observation.state",
    "action",
    "observation.images.image",
    "frame_index",
    "episode_index",
}


@pytest.fixture(params=[1, 2, 4], ids=["obs1", "obs2", "obs4"])
def dataset(request):
    """Create LiberoYoumuDataset with parametrized obs_len."""
    return LiberoYoumuDataset(
        data_dir=DATA_DIR,
        obs_len=request.param,
        pred_len=PRED_LEN,
        state_column="observation.state",
        action_column="action",
        image_column="observation.images.image",
    )


class TestDatasetLength:
    """Tests for dataset length."""

    def test_length_positive(self, dataset):
        """Dataset length must be a positive integer."""
        assert len(dataset) > 0

    def test_length_decreases_with_obs_len(self):
        """Longer obs windows should yield fewer valid anchors."""
        lengths = {}
        for obs_len in [1, 2, 4]:
            ds = LiberoYoumuDataset(
                data_dir=DATA_DIR,
                obs_len=obs_len,
                pred_len=PRED_LEN,
                state_column="observation.state",
                action_column="action",
                image_column="observation.images.image",
            )
            lengths[obs_len] = len(ds)
        assert lengths[1] > lengths[2] > lengths[4]


class TestSampleKeys:
    """Tests for sample dict keys."""

    def test_first_sample_keys(self, dataset):
        """First sample must have all expected keys."""
        sample = dataset[0]
        assert set(sample.keys()) == EXPECTED_KEYS

    def test_last_sample_keys(self, dataset):
        """Last sample must have all expected keys."""
        sample = dataset[len(dataset) - 1]
        assert set(sample.keys()) == EXPECTED_KEYS


class TestObservationState:
    """Tests for observation.state tensor."""

    def test_shape(self, dataset):
        """Shape must be (obs_len, state_dim)."""
        sample = dataset[0]
        obs = sample["observation.state"]
        assert obs.ndim == 2
        assert obs.shape[0] == dataset.obs_len
        assert obs.shape[1] > 0

    def test_dtype(self, dataset):
        """Dtype must be float32."""
        assert dataset[0]["observation.state"].dtype == torch.float32

    def test_finite_values(self, dataset):
        """Values must be finite (no NaN/Inf)."""
        assert torch.isfinite(dataset[0]["observation.state"]).all()


class TestAction:
    """Tests for action tensor."""

    def test_shape(self, dataset):
        """Shape must be (pred_len, action_dim)."""
        sample = dataset[0]
        action = sample["action"]
        assert action.ndim == 2
        assert action.shape[0] == PRED_LEN
        assert action.shape[1] > 0

    def test_dtype(self, dataset):
        """Dtype must be float32."""
        assert dataset[0]["action"].dtype == torch.float32

    def test_finite_values(self, dataset):
        """Values must be finite (no NaN/Inf)."""
        assert torch.isfinite(dataset[0]["action"]).all()


class TestImage:
    """Tests for observation.images.image tensor."""

    def test_shape(self, dataset):
        """Shape must be (obs_len, H, W, 3)."""
        img = dataset[0]["observation.images.image"]
        assert img.ndim == 4
        assert img.shape[0] == dataset.obs_len
        assert img.shape[3] == 3

    def test_dtype(self, dataset):
        """Dtype must be uint8."""
        assert dataset[0]["observation.images.image"].dtype == torch.uint8

    def test_image_dimensions(self, dataset):
        """Image H and W should be 256x256 for this dataset."""
        img = dataset[0]["observation.images.image"]
        assert img.shape[1] == 256
        assert img.shape[2] == 256


class TestMetadataFields:
    """Tests for frame_index and episode_index."""

    def test_frame_index_type(self, dataset):
        """frame_index must be an int."""
        assert isinstance(dataset[0]["frame_index"], int)

    def test_episode_index_type(self, dataset):
        """episode_index must be an int."""
        assert isinstance(dataset[0]["episode_index"], int)

    def test_frame_index_non_negative(self, dataset):
        """frame_index must be >= 0."""
        assert dataset[0]["frame_index"] >= 0

    def test_episode_index_non_negative(self, dataset):
        """episode_index must be >= 0."""
        assert dataset[0]["episode_index"] >= 0


class TestRandomAccess:
    """Tests for random index access without errors."""

    def test_100_random_indices(self, dataset):
        """Accessing 100 random indices should not raise."""
        random.seed(42)
        indices = random.sample(range(len(dataset)), min(100, len(dataset)))
        for idx in indices:
            sample = dataset[idx]
            assert set(sample.keys()) == EXPECTED_KEYS
            assert sample["observation.state"].shape[0] == dataset.obs_len
            assert sample["action"].shape[0] == PRED_LEN
