"""UESF EEGDataset - PyTorch Dataset wrapper for preprocessed numpy data."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    """Wraps numpy arrays as a PyTorch Dataset.

    Args:
        data: Feature array, shape (n_samples, channels, samples) or similar.
        labels: Label array, shape (n_samples,) or (n_samples, ...).
    """

    def __init__(self, data: np.ndarray, labels: np.ndarray) -> None:
        self.data = data.astype(np.float32)
        self.labels = labels
        assert len(self.data) == len(self.labels), (
            f"Data and labels length mismatch: {len(self.data)} vs {len(self.labels)}"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.data[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
