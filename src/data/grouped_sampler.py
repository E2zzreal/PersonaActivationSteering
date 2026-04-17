"""
Personality-Grouped Sampler for contrastive learning.

Ensures each batch contains samples from the same personality group,
so that SupervisedContrastiveLoss can find positive pairs.

Strategy: each batch is composed of `group_size` samples from the same
personality, repeated until `batch_size` is filled. Across batches,
personalities are cycled in a round-robin fashion with shuffling.
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from torch.utils.data import Sampler


class PersonalityGroupedSampler(Sampler[list[int]]):
    """Batch sampler that groups samples by personality.

    Each yielded batch is a list of dataset indices that share the same
    personality string (or at least `group_size` of them do).

    Args:
        data_path: Path to the JSONL data file (same as ALOEDataset).
        batch_size: Total batch size.
        group_size: Minimum number of same-personality samples per batch.
            Defaults to batch_size (entire batch from one personality).
        shuffle: Whether to shuffle personalities and samples each epoch.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        data_path: str | Path,
        batch_size: int = 4,
        group_size: int | None = None,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.group_size = group_size or batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Build personality -> [index] mapping
        self.personality_to_indices: dict[str, list[int]] = defaultdict(list)
        with open(data_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                item = json.loads(line)
                p = item.get("personality", "")
                self.personality_to_indices[p].append(idx)

        self._total = sum(len(v) for v in self.personality_to_indices.values())

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        # Shuffle indices within each personality group
        groups: dict[str, list[int]] = {}
        for p, indices in self.personality_to_indices.items():
            idxs = indices.copy()
            if self.shuffle:
                rng.shuffle(idxs)
            groups[p] = idxs

        # Round-robin across personalities, yielding batches
        personality_keys = list(groups.keys())
        if self.shuffle:
            rng.shuffle(personality_keys)

        # Build batches: take group_size from one personality at a time
        batches: list[list[int]] = []
        pointers = {p: 0 for p in personality_keys}

        while True:
            made_progress = False
            for p in personality_keys:
                ptr = pointers[p]
                idxs = groups[p]
                if ptr >= len(idxs):
                    continue
                end = min(ptr + self.batch_size, len(idxs))
                batch = idxs[ptr:end]
                if len(batch) >= 2:  # Need at least 2 for contrastive
                    batches.append(batch)
                    made_progress = True
                pointers[p] = end
            if not made_progress:
                break

        if self.shuffle:
            rng.shuffle(batches)

        yield from batches

    def __len__(self) -> int:
        # Approximate number of batches
        count = 0
        for indices in self.personality_to_indices.values():
            count += max(1, len(indices) // self.batch_size)
        return count

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for shuffling determinism."""
        self.epoch = epoch
