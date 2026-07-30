"""PyTorch datasets and grouped-neighbourhood loading."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch.utils.data import DataLoader, Dataset


class RegressionDataset(Dataset):
    """Pair local XYZ neighbourhoods with physical features and a target."""

    def __init__(self, neighbourhoods: ArrayLike, labels: ArrayLike | None = None):
        self.neighbourhoods = torch.as_tensor(neighbourhoods, dtype=torch.float32)
        self.labels = (
            None if labels is None else torch.as_tensor(labels, dtype=torch.float32)
        )
        if self.neighbourhoods.ndim != 3 or self.neighbourhoods.shape[-1] != 3:
            raise ValueError("neighbourhoods must have shape (N, K, 3)")
        if self.labels is not None and len(self.labels) != len(self.neighbourhoods):
            raise ValueError("neighbourhoods and labels must have equal length")

    def __getitem__(self, index: int):
        if self.labels is None:
            return self.neighbourhoods[index]
        return self.neighbourhoods[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.neighbourhoods)


def load_group_arrays(
    scan_ids: Sequence[int | float],
    directory: str | Path,
    filename_prefix: str = "Group",
) -> torch.Tensor:
    """Load and concatenate one grouped-neighbourhood array per scan."""
    root = Path(directory)
    arrays = []
    for scan_id in scan_ids:
        path = root / f"{filename_prefix}_{scan_id}.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        array = np.load(path, allow_pickle=False)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"{path} must have shape (N, K, 3)")
        arrays.append(torch.as_tensor(array, dtype=torch.float32))

    if not arrays:
        raise ValueError("scan_ids must contain at least one scan")
    return torch.cat(arrays, dim=0)


def make_dataloaders(
    train_neighbourhoods: ArrayLike,
    train_labels: ArrayLike,
    validation_neighbourhoods: ArrayLike,
    validation_labels: ArrayLike,
    test_neighbourhoods: ArrayLike,
    test_labels: ArrayLike,
    *,
    batch_size: int,
    test_batch_size: int = 1,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and test data loaders."""
    if batch_size < 1 or test_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    use_pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory

    train = RegressionDataset(train_neighbourhoods, train_labels)
    validation = RegressionDataset(
        validation_neighbourhoods, validation_labels
    )
    test = RegressionDataset(test_neighbourhoods, test_labels)
    return (
        DataLoader(
            train,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=use_pin_memory,
        ),
        DataLoader(
            validation,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=use_pin_memory,
        ),
        DataLoader(
            test,
            batch_size=test_batch_size,
            shuffle=False,
            pin_memory=use_pin_memory,
        ),
    )
