"""Validate LiberoYoumuDataset with converted 64KB page-size parquet dataset.

Instantiates the dataset with various obs_len values and validates:
- Dataset length is positive
- Sample dict keys match expected schema
- Tensor shapes and dtypes are correct
- Random access works without errors
"""

import random
import sys

import torch

from youmu.libero_dataset import LiberoYoumuDataset

DATA_DIR = "/home/ubuntu/dataset/hf_vla_64k"
PRED_LEN = 4


def validate_sample(sample: dict, obs_len: int, pred_len: int, idx: int) -> None:
    """Validate a single sample's keys, shapes, and dtypes.

    Args:
        sample: Dict returned by dataset[idx].
        obs_len: Expected observation window length.
        pred_len: Expected prediction window length.
        idx: Index used to fetch this sample (for error messages).
    """
    # Check required keys
    expected_keys = {
        "observation.state",
        "action",
        "observation.images.image",
        "frame_index",
        "episode_index",
    }
    actual_keys = set(sample.keys())
    assert actual_keys == expected_keys, (
        f"Sample {idx}: expected keys {expected_keys}, got {actual_keys}"
    )

    # observation.state: (obs_len, state_dim), float32
    obs_state = sample["observation.state"]
    assert obs_state.dtype == torch.float32, (
        f"Sample {idx}: observation.state dtype={obs_state.dtype}, expected float32"
    )
    assert obs_state.ndim == 2, (
        f"Sample {idx}: observation.state ndim={obs_state.ndim}, expected 2"
    )
    assert obs_state.shape[0] == obs_len, (
        f"Sample {idx}: observation.state shape[0]={obs_state.shape[0]}, expected {obs_len}"
    )
    state_dim = obs_state.shape[1]
    assert state_dim > 0, f"Sample {idx}: state_dim={state_dim}, expected > 0"

    # action: (pred_len, action_dim), float32
    action = sample["action"]
    assert action.dtype == torch.float32, (
        f"Sample {idx}: action dtype={action.dtype}, expected float32"
    )
    assert action.ndim == 2, (
        f"Sample {idx}: action ndim={action.ndim}, expected 2"
    )
    assert action.shape[0] == pred_len, (
        f"Sample {idx}: action shape[0]={action.shape[0]}, expected {pred_len}"
    )
    action_dim = action.shape[1]
    assert action_dim > 0, f"Sample {idx}: action_dim={action_dim}, expected > 0"

    # observation.images.image: (obs_len, H, W, 3), uint8
    img = sample["observation.images.image"]
    assert img.dtype == torch.uint8, (
        f"Sample {idx}: image dtype={img.dtype}, expected uint8"
    )
    assert img.ndim == 4, (
        f"Sample {idx}: image ndim={img.ndim}, expected 4"
    )
    assert img.shape[0] == obs_len, (
        f"Sample {idx}: image shape[0]={img.shape[0]}, expected {obs_len}"
    )
    assert img.shape[3] == 3, (
        f"Sample {idx}: image shape[3]={img.shape[3]}, expected 3 (RGB)"
    )

    # frame_index and episode_index should be ints
    assert isinstance(sample["frame_index"], int), (
        f"Sample {idx}: frame_index type={type(sample['frame_index'])}"
    )
    assert isinstance(sample["episode_index"], int), (
        f"Sample {idx}: episode_index type={type(sample['episode_index'])}"
    )


def validate_dataset(obs_len: int) -> None:
    """Validate LiberoYoumuDataset with a given obs_len.

    Args:
        obs_len: Number of observation frames.
    """
    print(f"\n=== Validating obs_len={obs_len}, pred_len={PRED_LEN} ===")

    dataset = LiberoYoumuDataset(
        data_dir=DATA_DIR,
        obs_len=obs_len,
        pred_len=PRED_LEN,
        state_column="observation.state",
        action_column="action",
        image_column="observation.images.image",
    )

    length = len(dataset)
    assert length > 0, f"Dataset length is {length}, expected > 0"
    print(f"  len(dataset) = {length}")

    # Validate first sample
    sample = dataset[0]
    validate_sample(sample, obs_len, PRED_LEN, idx=0)
    print(f"  sample[0] OK — state: {sample['observation.state'].shape}, "
          f"action: {sample['action'].shape}, "
          f"image: {sample['observation.images.image'].shape}")

    # Iterate 100 random indices
    random.seed(42)
    indices = random.sample(range(length), min(100, length))
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        validate_sample(sample, obs_len, PRED_LEN, idx=idx)
        if (i + 1) % 25 == 0:
            print(f"  Validated {i + 1}/100 random samples...")

    print(f"  All 100 random samples validated for obs_len={obs_len}")


def main() -> None:
    """Run validation for obs_len=1, 2, and 4."""
    print("LiberoYoumuDataset Validation")
    print(f"Data dir: {DATA_DIR}")

    for obs_len in [1, 2, 4]:
        validate_dataset(obs_len)

    print("\n" + "=" * 50)
    print("PASS — All validations succeeded!")


if __name__ == "__main__":
    main()
