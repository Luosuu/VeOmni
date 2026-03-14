"""US-004: Correctness validation for LiberoYoumuPageAlignedDataset.

Integration tests verifying that the page-aligned dataset produces the exact
same set of samples as the row-range dataset (LiberoYoumuDataset), ensuring
training semantics are preserved.

Requires the real dataset at /home/ubuntu/dataset/hf_vla_64k.

Tests:
1. Full-epoch anchor set comparison (frame_index, episode_index) matches.
2. Tensor comparison for random anchor frames (state, action, image).
3. Shuffle buffer randomness verification.
4. Edge anchor (page boundary) correctness.
"""

import os
import random

import pytest
import torch

from youmu.libero_dataset import (
    LiberoYoumuDataset,
    LiberoYoumuPageAlignedDataset,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_DIR = "/home/ubuntu/dataset/hf_vla_64k"
OBS_LEN = 2
PRED_LEN = 4

# Column names matching the hf_vla_64k schema
STATE_COL = "observation.state"
ACTION_COL = "action"
IMAGE_COL = "observation.images.image"


def _dataset_available() -> bool:
    """Check if the real dataset is available."""
    return os.path.isdir(DATASET_DIR) and os.path.isdir(
        os.path.join(DATASET_DIR, "data", "chunk-000")
    )


skip_no_dataset = pytest.mark.skipif(
    not _dataset_available(),
    reason=f"Real dataset not found at {DATASET_DIR}",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def row_range_dataset():
    """Create the row-range (map-style) dataset once per module."""
    return LiberoYoumuDataset(
        data_dir=DATASET_DIR,
        obs_len=OBS_LEN,
        pred_len=PRED_LEN,
        state_column=STATE_COL,
        action_column=ACTION_COL,
        image_column=IMAGE_COL,
    )


@pytest.fixture(scope="module")
def row_range_no_image():
    """Row-range dataset without images for anchor set comparison."""
    return LiberoYoumuDataset(
        data_dir=DATASET_DIR,
        obs_len=OBS_LEN,
        pred_len=PRED_LEN,
        state_column=STATE_COL,
        action_column=ACTION_COL,
        image_column=None,
    )


@pytest.fixture(scope="module")
def page_aligned_dataset():
    """Create the page-aligned (iterable) dataset once per module."""
    return LiberoYoumuPageAlignedDataset(
        data_dir=DATASET_DIR,
        obs_len=OBS_LEN,
        pred_len=PRED_LEN,
        state_column=STATE_COL,
        action_column=ACTION_COL,
        image_column=IMAGE_COL,
        shuffle_buffer_size=300000,  # large enough to hold all anchors
        page_cache_size=5,
    )


@pytest.fixture(scope="module")
def page_aligned_no_image():
    """Page-aligned dataset without images for faster full-epoch iteration."""
    return LiberoYoumuPageAlignedDataset(
        data_dir=DATASET_DIR,
        obs_len=OBS_LEN,
        pred_len=PRED_LEN,
        state_column=STATE_COL,
        action_column=ACTION_COL,
        image_column=None,
        shuffle_buffer_size=300000,
        page_cache_size=5,
    )


@pytest.fixture(scope="module")
def full_epoch_no_image_samples(page_aligned_no_image):
    """Iterate full epoch of page-aligned (no images) and collect all samples.

    Returns a dict keyed by (frame_index, episode_index) mapping to sample
    dicts, plus the ordered list of frame_indices for randomness checks.
    """
    samples_by_key = {}
    frame_index_order = []
    episode_index_order = []

    for sample in page_aligned_no_image:
        key = (sample["frame_index"], sample["episode_index"])
        frame_index_order.append(sample["frame_index"])
        episode_index_order.append(sample["episode_index"])
        # Store first occurrence (all should be identical)
        if key not in samples_by_key:
            samples_by_key[key] = sample

    return {
        "by_key": samples_by_key,
        "frame_index_order": frame_index_order,
        "episode_index_order": episode_index_order,
    }


@pytest.fixture(scope="module")
def anchor_to_idx(row_range_no_image):
    """Build anchor (frame, ep) -> row-range index map."""
    mapping = {}
    for idx, (frame, ep) in enumerate(row_range_no_image.anchors):
        mapping[(frame, ep)] = idx
    return mapping


# ---------------------------------------------------------------------------
# Test 1: Full-epoch anchor set comparison
# ---------------------------------------------------------------------------


@skip_no_dataset
class TestAnchorSetComparison:
    """Verify page-aligned yields the exact same (frame_index, episode_index) set."""

    def test_anchor_sets_identical(
        self, row_range_no_image, full_epoch_no_image_samples
    ):
        """Both datasets must produce the exact same set of
        (frame_index, episode_index) pairs.
        """
        rr_anchors = set(
            (anchor[0], anchor[1]) for anchor in row_range_no_image.anchors
        )
        pa_anchors = set(full_epoch_no_image_samples["by_key"].keys())

        assert len(pa_anchors) == len(rr_anchors), (
            f"Anchor count mismatch: page-aligned={len(pa_anchors)}, "
            f"row-range={len(rr_anchors)}"
        )
        assert pa_anchors == rr_anchors, (
            f"Anchor sets differ. "
            f"In PA not RR: {len(pa_anchors - rr_anchors)}. "
            f"In RR not PA: {len(rr_anchors - pa_anchors)}."
        )

    def test_anchor_count_matches_len(
        self, page_aligned_no_image, full_epoch_no_image_samples
    ):
        """__len__ returns the same count as actual iteration."""
        expected_len = len(page_aligned_no_image)
        actual_count = len(full_epoch_no_image_samples["frame_index_order"])
        assert actual_count == expected_len, (
            f"__len__={expected_len} but iterated {actual_count} samples"
        )


# ---------------------------------------------------------------------------
# Test 2: Tensor comparison for random anchor frames
# ---------------------------------------------------------------------------


@skip_no_dataset
class TestTensorComparison:
    """Verify tensor outputs match between page-aligned and row-range datasets."""

    def test_state_and_action_match_for_random_anchors(
        self,
        row_range_no_image,
        anchor_to_idx,
        full_epoch_no_image_samples,
    ):
        """For 100 random anchor frames, verify state and action tensors match.

        Uses no-image datasets for speed, comparing float tensors with allclose.
        """
        all_keys = list(full_epoch_no_image_samples["by_key"].keys())
        compare_keys = random.sample(all_keys, min(100, len(all_keys)))

        for key in compare_keys:
            pa_sample = full_epoch_no_image_samples["by_key"][key]
            rr_sample = row_range_no_image[anchor_to_idx[key]]

            assert torch.allclose(
                pa_sample["observation.state"],
                rr_sample["observation.state"],
                atol=1e-6,
            ), f"State mismatch at anchor {key}"

            assert torch.allclose(
                pa_sample["action"],
                rr_sample["action"],
                atol=1e-6,
            ), f"Action mismatch at anchor {key}"

    def test_images_match_for_random_anchors(
        self, row_range_dataset, page_aligned_dataset
    ):
        """For 20 random anchor frames, verify image tensors match exactly.

        Uses image-enabled datasets. Collects a limited number of samples
        to keep runtime manageable.
        """
        anchor_to_idx_img = {}
        for idx, (frame, ep) in enumerate(row_range_dataset.anchors):
            anchor_to_idx_img[(frame, ep)] = idx

        # Collect first 3000 samples with images
        pa_samples = []
        for i, sample in enumerate(page_aligned_dataset):
            pa_samples.append(sample)
            if i >= 2999:
                break

        compare_samples = random.sample(pa_samples, min(20, len(pa_samples)))

        for pa_sample in compare_samples:
            frame = pa_sample["frame_index"]
            ep = pa_sample["episode_index"]
            rr_sample = row_range_dataset[anchor_to_idx_img[(frame, ep)]]

            assert torch.equal(
                pa_sample["observation.images.image"],
                rr_sample["observation.images.image"],
            ), f"Image mismatch at anchor ({frame}, {ep})"


# ---------------------------------------------------------------------------
# Test 3: Shuffle buffer randomness
# ---------------------------------------------------------------------------


@skip_no_dataset
class TestShuffleBufferRandomness:
    """Verify that the shuffle buffer produces randomized output order."""

    def test_first_1000_not_sorted(self, full_epoch_no_image_samples):
        """First 1000 yielded frame_indices should not be in sorted order."""
        frame_indices = full_epoch_no_image_samples["frame_index_order"][:1000]

        assert frame_indices != sorted(frame_indices), (
            "First 1000 frame_indices are in sorted order — "
            "shuffle buffer may not be working"
        )

        assert frame_indices != sorted(frame_indices, reverse=True), (
            "First 1000 frame_indices are in reverse-sorted order"
        )

    def test_shuffled_output_has_mixed_episodes(
        self, full_epoch_no_image_samples
    ):
        """First 1000 samples should come from many different episodes.

        Verifies page-level shuffling produces diverse episode coverage.
        """
        episode_indices = full_epoch_no_image_samples["episode_index_order"][
            :1000
        ]
        unique_episodes = set(episode_indices)
        # With 1693 episodes, first 1000 samples should span many episodes
        assert len(unique_episodes) > 10, (
            f"Only {len(unique_episodes)} unique episodes in first 1000 "
            f"samples — expected more diversity from shuffling"
        )


# ---------------------------------------------------------------------------
# Test 4: Edge anchor (page boundary) correctness
# ---------------------------------------------------------------------------


@skip_no_dataset
class TestEdgeAnchorCorrectness:
    """Verify anchors at page boundaries produce correct obs/pred windows."""

    def test_first_anchor_per_page_matches_row_range(
        self,
        row_range_no_image,
        page_aligned_no_image,
        anchor_to_idx,
        full_epoch_no_image_samples,
    ):
        """First valid anchor in each page (obs window may extend into
        previous page) must produce the same tensor as row-range.

        Tests page cache correctness for backward-looking edge anchors.
        """
        edge_anchors = []
        for page_key, anchors in page_aligned_no_image._anchors_by_page.items():
            if anchors:
                first = min(anchors, key=lambda a: a[0])
                edge_anchors.append(first)

        test_anchors = random.sample(
            edge_anchors, min(50, len(edge_anchors))
        )

        pa_by_key = full_epoch_no_image_samples["by_key"]

        for frame, ep in test_anchors:
            key = (frame, ep)
            assert key in pa_by_key, (
                f"Edge anchor {key} not found in page-aligned output"
            )
            pa_sample = pa_by_key[key]
            rr_sample = row_range_no_image[anchor_to_idx[key]]

            assert torch.allclose(
                pa_sample["observation.state"],
                rr_sample["observation.state"],
                atol=1e-6,
            ), f"State mismatch at first edge anchor {key}"

            assert torch.allclose(
                pa_sample["action"],
                rr_sample["action"],
                atol=1e-6,
            ), f"Action mismatch at first edge anchor {key}"

    def test_last_anchor_per_page_matches_row_range(
        self,
        row_range_no_image,
        page_aligned_no_image,
        anchor_to_idx,
        full_epoch_no_image_samples,
    ):
        """Last valid anchor in each page (pred window may extend into
        next page) must produce the same tensor as row-range.

        Tests page cache correctness for forward-looking edge anchors.
        """
        edge_anchors = []
        for page_key, anchors in page_aligned_no_image._anchors_by_page.items():
            if anchors:
                last = max(anchors, key=lambda a: a[0])
                edge_anchors.append(last)

        test_anchors = random.sample(
            edge_anchors, min(50, len(edge_anchors))
        )

        pa_by_key = full_epoch_no_image_samples["by_key"]

        for frame, ep in test_anchors:
            key = (frame, ep)
            assert key in pa_by_key, (
                f"Edge anchor {key} not found in page-aligned output"
            )
            pa_sample = pa_by_key[key]
            rr_sample = row_range_no_image[anchor_to_idx[key]]

            assert torch.allclose(
                pa_sample["observation.state"],
                rr_sample["observation.state"],
                atol=1e-6,
            ), f"State mismatch at last edge anchor {key}"

            assert torch.allclose(
                pa_sample["action"],
                rr_sample["action"],
                atol=1e-6,
            ), f"Action mismatch at last edge anchor {key}"
