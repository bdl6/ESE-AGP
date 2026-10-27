"""Shared ranking metrics and train-only, non-neural baselines.

Catalog IDs are 1..N. ID 0 represents an unknown item or padding and is
never recommended. Rankings contain unique item IDs in their predicted order;
repeat recommendations of items already present in a session are allowed.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Iterable, Sequence


def _integer(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _catalog_item(value: Any, num_items: int) -> int:
    item_id = _integer(value, "item ID", minimum=0)
    if item_id > num_items:
        raise ValueError(f"item ID {item_id} is outside catalog 0..{num_items}")
    return item_id


def evaluate_rankings(
    rankings: Iterable[Sequence[int]],
    targets: Iterable[int],
    k: int = 20,
) -> dict[str, float | int]:
    """Evaluate next-item predictions with OOV targets retained as misses.

    ``target == 0`` is OOV. Overall HR/MRR use *all* examples as their
    denominator; the known-target variants exclude OOV targets. Empty
    denominators yield 0.0, with explicit counts to make that convention clear.
    Values are proportions in [0, 1], not percentages.
    """

    k = _integer(k, "k", minimum=1)
    ranking_list = list(rankings)
    target_list = list(targets)
    if len(ranking_list) != len(target_list):
        raise ValueError("rankings and targets must contain the same number of examples")

    hits = 0
    reciprocal_rank_sum = 0.0
    known_count = 0
    for ranking, target_value in zip(ranking_list, target_list):
        target = _integer(target_value, "target item ID", minimum=0)
        items = [_integer(item, "ranked item ID", minimum=1) for item in ranking]
        if len(set(items)) != len(items):
            raise ValueError("each ranking must contain unique item IDs")
        if target == 0:
            continue
        known_count += 1
        try:
            rank = items[:k].index(target) + 1
        except ValueError:
            continue
        hits += 1
        reciprocal_rank_sum += 1.0 / rank

    total = len(target_list)
    oov_count = total - known_count
    return {
        f"hr@{k}": hits / total if total else 0.0,
        f"mrr@{k}": reciprocal_rank_sum / total if total else 0.0,
        f"known_hr@{k}": hits / known_count if known_count else 0.0,
        f"known_mrr@{k}": reciprocal_rank_sum / known_count if known_count else 0.0,
        "oov_target_rate": oov_count / total if total else 0.0,
        "num_examples": total,
        "num_known_targets": known_count,
        "num_oov_targets": oov_count,
    }


@dataclass(frozen=True)
class PopularityBaseline:
    """Rank the catalog by training occurrence count, then by item ID.

    The data protocol creates one example per session. Its full training
    sequence is therefore ``input_item_ids + [target_item_id]``. Calling this
    method with prefix-augmented examples would intentionally count those
    repeated occurrences and is a different preprocessing protocol.
    """

    num_items: int
    counts: tuple[int, ...]
    _ranking: tuple[int, ...]

    @classmethod
    def fit(cls, train: Iterable[Any], num_items: int) -> PopularityBaseline:
        num_items = _integer(num_items, "num_items", minimum=1)
        counts = [0] * (num_items + 1)
        for example in train:
            sequence = [*example.input_item_ids, example.target_item_id]
            for value in sequence:
                item_id = _catalog_item(value, num_items)
                if item_id:
                    counts[item_id] += 1
        ranking = tuple(sorted(range(1, num_items + 1), key=lambda item: (-counts[item], item)))
        return cls(num_items=num_items, counts=tuple(counts), _ranking=ranking)

    def recommend(self, prefix: Sequence[int], top_k: int) -> list[int]:
        top_k = _integer(top_k, "top_k", minimum=1)
        # Prefix is irrelevant to popularity, but validate the shared input contract.
        for value in prefix:
            _catalog_item(value, self.num_items)
        return list(self._ranking[:top_k])


@dataclass(frozen=True)
class TransitionBaseline:
    """Last-item transition counts fitted exclusively on training sessions.

    Unknown positions interrupt adjacency: [A, 0, B] adds neither A->B nor
    transitions involving 0. Empty histories or histories ending in 0 use the
    training popularity ranking. For a known final item, transition counts are
    the scores for all catalog items, including zero-count items; ties are
    broken by ascending item ID. There is no seen-item filter.
    """

    num_items: int
    popularity: PopularityBaseline
    transitions: dict[int, dict[int, int]]

    @classmethod
    def fit(cls, train: Iterable[Any], num_items: int) -> TransitionBaseline:
        num_items = _integer(num_items, "num_items", minimum=1)
        examples = list(train)
        popularity = PopularityBaseline.fit(examples, num_items)
        transitions: dict[int, Counter[int]] = defaultdict(Counter)
        for example in examples:
            sequence = [
                _catalog_item(value, num_items)
                for value in [*example.input_item_ids, example.target_item_id]
            ]
            for source, target in zip(sequence, sequence[1:]):
                if source and target:
                    transitions[source][target] += 1
        return cls(
            num_items=num_items,
            popularity=popularity,
            transitions={source: dict(counts) for source, counts in transitions.items()},
        )

    def recommend(self, prefix: Sequence[int], top_k: int) -> list[int]:
        top_k = _integer(top_k, "top_k", minimum=1)
        items = [_catalog_item(value, self.num_items) for value in prefix]
        if not items or items[-1] == 0:
            return self.popularity.recommend(items, top_k)
        counts = self.transitions.get(items[-1], {})
        ranking = sorted(range(1, self.num_items + 1), key=lambda item: (-counts.get(item, 0), item))
        return ranking[:top_k]
