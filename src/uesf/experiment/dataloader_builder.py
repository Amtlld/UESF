"""UESF DataLoader Builder - multi-channel dictionary DataLoader.

Creates per-channel DataLoaders and combines them via a CombinedIterator
to produce batch dicts: {channel_name: (data, labels)}.
"""

from __future__ import annotations

from torch.utils.data import DataLoader, Dataset

from uesf.core.logging import get_logger

logger = get_logger("experiment.dataloader")


class CombinedIterator:
    """Iterates multiple DataLoaders in parallel, yielding batch dicts.

    Stops when the shortest DataLoader is exhausted.
    """

    def __init__(self, loaders: dict[str, DataLoader]) -> None:
        self.loaders = loaders

    def __iter__(self):
        iterators = {name: iter(loader) for name, loader in self.loaders.items()}
        while True:
            batch = {}
            try:
                for name, it in iterators.items():
                    batch[name] = next(it)
            except StopIteration:
                break
            yield batch

    def __len__(self) -> int:
        if not self.loaders:
            return 0
        return min(len(loader) for loader in self.loaders.values())


def build_dataloaders(
    datasets: dict[str, Dataset],
    batch_size: int = 32,
    shuffle_train: bool = True,
    num_workers: int = 0,
    phase: str = "train",
) -> CombinedIterator:
    """Build a CombinedIterator from a dict of channel datasets.

    Args:
        datasets: Dict mapping channel names to Dataset instances.
        batch_size: Batch size for all loaders.
        shuffle_train: Whether to shuffle (only for train phase).
        num_workers: Number of workers for DataLoader.
        phase: Current phase ('train', 'val', 'test').

    Returns:
        CombinedIterator wrapping all channel DataLoaders.
    """
    should_shuffle = shuffle_train and phase == "train"

    loaders = {}
    for name, dataset in datasets.items():
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            num_workers=num_workers,
            drop_last=False,
        )

    return CombinedIterator(loaders)
