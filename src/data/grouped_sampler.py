"""
Personality-Grouped Sampler for contrastive learning.

Ensures each batch contains samples from MULTIPLE personality groups,
so that SupervisedContrastiveLoss has both positive pairs (same personality)
and negative pairs (different personalities).

Strategy: each batch combines `group_size` samples from each of
`num_groups` different personalities, yielding `batch_size = group_size * num_groups`.
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from torch.utils.data import Sampler


class PersonalityGroupedSampler(Sampler[list[int]]):
    """Batch sampler that mixes multiple personality groups per batch.

    Each yielded batch contains samples from `num_groups` different
    personalities, with `group_size` samples per personality.
    This provides both positive pairs (same personality) and
    negative pairs (different personalities) for contrastive learning.

    Example with batch_size=4, group_size=2:
      batch = [personA_idx1, personA_idx2, personB_idx1, personB_idx2]
      → positive pairs: (A1,A2), (B1,B2)
      → negative pairs: (A1,B1), (A1,B2), (A2,B1), (A2,B2)

    Args:
        data_path: Path to the JSONL data file (same as ALOEDataset).
        batch_size: Total batch size. Must be divisible by group_size.
        group_size: Number of same-personality samples per group.
            Defaults to 2 (minimum for positive pairs).
        shuffle: Whether to shuffle personalities and samples each epoch.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        data_path: str | Path,
        batch_size: int = 4,
        group_size: int = 2,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.group_size = group_size
        self.num_groups = batch_size // group_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        assert batch_size % group_size == 0, \
            f"batch_size ({batch_size}) must be divisible by group_size ({group_size})"
        assert self.num_groups >= 2, \
            f"Need at least 2 personality groups per batch for contrastive learning, " \
            f"got {self.num_groups} (batch_size={batch_size}, group_size={group_size})"

        # Build personality -> [index] mapping
        self.personality_to_indices: dict[str, list[int]] = defaultdict(list)
        with open(data_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                item = json.loads(line)
                p = item.get("personality", "")
                self.personality_to_indices[p].append(idx)

        # Only keep personalities with enough samples
        self.valid_personalities = [
            p for p, idxs in self.personality_to_indices.items()
            if len(idxs) >= group_size
        ]
        self._total = sum(len(self.personality_to_indices[p])
                         for p in self.valid_personalities)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        # Shuffle indices within each personality group
        queues: dict[str, list[int]] = {}
        for p in self.valid_personalities:
            idxs = self.personality_to_indices[p].copy()
            if self.shuffle:
                rng.shuffle(idxs)
            queues[p] = idxs

        pointers = {p: 0 for p in self.valid_personalities}

        # Track which personalities still have enough samples
        active = set(self.valid_personalities)

        batches: list[list[int]] = []

        while len(active) >= self.num_groups:
            # Pick num_groups different personalities
            selected = rng.sample(sorted(active), self.num_groups)

            batch = []
            exhausted = []
            for p in selected:
                ptr = pointers[p]
                end = ptr + self.group_size
                idxs = queues[p]

                if end > len(idxs):
                    # Not enough samples left for this personality
                    exhausted.append(p)
                    continue

                batch.extend(idxs[ptr:end])
                pointers[p] = end

                # Check if personality is exhausted for future batches
                if end + self.group_size > len(idxs):
                    exhausted.append(p)

            # Only yield if we got samples from at least 2 personalities
            if len(batch) >= self.group_size * 2:
                batches.append(batch)

            for p in exhausted:
                active.discard(p)

        if self.shuffle:
            rng.shuffle(batches)

        yield from batches

    def __len__(self) -> int:
        # Approximate: total samples / batch_size
        return self._total // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for shuffling determinism."""
        self.epoch = epoch
