"""Auditable, small-data session preparation without third-party dependencies.

This module is the offline JSONL prototype, not a full Amazon ingestion job.
Timestamps are integer milliseconds. Unknown items and padding both use ID 0,
but their original positions are retained so they cannot create new edges.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence


DAY_MS = 24 * 60 * 60 * 1000


@dataclass(frozen=True)
class Interaction:
    user_id: str
    item_id: str
    timestamp: int

    def __post_init__(self) -> None:
        if not isinstance(self.user_id, str) or not self.user_id:
            raise ValueError("user_id must be a nonempty string")
        if not isinstance(self.item_id, str) or not self.item_id:
            raise ValueError("item_id must be a nonempty string")
        if isinstance(self.timestamp, bool) or not isinstance(self.timestamp, int):
            raise ValueError("timestamp must be an integer number of milliseconds")


@dataclass(frozen=True)
class SessionExample:
    input_item_ids: tuple[int, ...]
    target_item_id: int
    user_id: str
    session_start: int
    session_end: int
    session_id: str = ""


@dataclass
class PreparedData:
    train: list[SessionExample]
    valid: list[SessionExample]
    test: list[SessionExample]
    catalog: dict[str, int]
    popularity: list[int]
    manifest: dict


@dataclass(frozen=True)
class LocalGraph:
    node_item_ids: list[int]
    edge_index: tuple[list[int], list[int]]
    position_to_node: list[list[int]]
    position_mask: list[list[bool]]
    has_known_history: list[bool]


@dataclass(frozen=True)
class _RawSession:
    user_id: str
    window: int
    events: tuple[Interaction, ...]

    @property
    def start(self) -> int:
        return self.events[0].timestamp

    @property
    def end(self) -> int:
        return self.events[-1].timestamp


def load_interactions_jsonl(path: str | Path) -> list[Interaction]:
    """Read fixture records, preserving input order for equal timestamps."""
    result = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                result.append(Interaction(row["user_id"], row["item_id"], row["timestamp"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid interaction at line {line_number}: {exc}") from exc
    return result


def _nearest_rank(values: Sequence[int], fraction: float) -> int:
    return values[max(0, math.ceil(len(values) * fraction) - 1)]


def prepare(
    interactions: Iterable[Interaction],
    *,
    min_user_events: int = 3,
    min_item_events: int = 5,
    max_session_length: int = 20,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
) -> PreparedData:
    """Prepare one last-item example per fixed 24-hour user window.

    Split boundaries come from raw length-at-least-two session end times.
    Crossing sessions are dropped *before* truncation or frequency filtering.
    User filtering then item filtering is a single train-only pass. A removed
    training target discards its example; earlier events never replace a label.
    Evaluation retains target ID 0, which the evaluator must count as a miss.
    """
    if min_user_events < 1 or min_item_events < 1:
        raise ValueError("frequency thresholds must be at least 1")
    if max_session_length < 2:
        raise ValueError("max_session_length must include an input and a target")
    if not (0 < train_ratio < 1 and 0 < valid_ratio < 1 - train_ratio):
        raise ValueError("ratios must leave nonempty train, validation and test intervals")

    records = list(interactions)
    if any(not isinstance(record, Interaction) for record in records):
        raise TypeError("interactions must contain Interaction objects")
    if not records:
        raise ValueError("at least one interaction is required")

    by_user: dict[str, list[tuple[int, Interaction]]] = defaultdict(list)
    digest = hashlib.sha256()
    for original_row, record in enumerate(records):
        by_user[record.user_id].append((original_row, record))
        digest.update(json.dumps(
            [record.user_id, record.item_id, record.timestamp],
            ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8") + b"\n")

    raw_sessions = []
    for user_id in sorted(by_user):
        ordered = sorted(by_user[user_id], key=lambda pair: (pair[1].timestamp, pair[0]))
        anchor = ordered[0][1].timestamp
        windows: dict[int, list[Interaction]] = defaultdict(list)
        for _, record in ordered:
            windows[(record.timestamp - anchor) // DAY_MS].append(record)
        raw_sessions.extend(
            _RawSession(user_id, window, tuple(events))
            for window, events in sorted(windows.items())
        )
    sessions = [session for session in raw_sessions if len(session.events) >= 2]
    if not sessions:
        raise ValueError("no session has both an input and a target")
    sessions.sort(key=lambda session: (session.end, session.start, session.user_id, session.window))
    end_times = sorted(session.end for session in sessions)
    train_end = _nearest_rank(end_times, train_ratio)
    valid_end = _nearest_rank(end_times, train_ratio + valid_ratio)
    raw_splits: dict[str, list[_RawSession]] = {"train": [], "valid": [], "test": []}
    crossing = 0
    for session in sessions:
        if any(session.start <= boundary < session.end for boundary in (train_end, valid_end)):
            crossing += 1
        elif session.end <= train_end:
            raw_splits["train"].append(session)
        elif session.end <= valid_end:
            raw_splits["valid"].append(session)
        else:
            raw_splits["test"].append(session)

    user_counts = Counter(
        session.user_id
        for session in raw_splits["train"]
        for _ in session.events
    )
    eligible_users = {user for user, count in user_counts.items() if count >= min_user_events}
    item_counts = Counter(
        event.item_id
        for session in raw_splits["train"] if session.user_id in eligible_users
        for event in session.events
    )
    catalog = {
        item_id: index
        for index, item_id in enumerate(
            sorted(item for item, count in item_counts.items() if count >= min_item_events),
            start=1,
        )
    }

    prepared: dict[str, list[SessionExample]] = {"train": [], "valid": [], "test": []}
    removed_user_sessions = removed_target = removed_history = 0
    for split, split_sessions in raw_splits.items():
        for session in split_sessions:
            if split == "train" and session.user_id not in eligible_users:
                removed_user_sessions += 1
                continue
            recent = session.events[-max_session_length:]
            prefix = tuple(catalog.get(event.item_id, 0) for event in recent[:-1])
            target = catalog.get(recent[-1].item_id, 0)
            if split == "train" and target == 0:
                removed_target += 1
                continue
            if split == "train" and not any(prefix):
                removed_history += 1
                continue
            prepared[split].append(SessionExample(
                input_item_ids=prefix,
                target_item_id=target,
                user_id=session.user_id,
                session_start=session.start,
                session_end=session.end,
                session_id=json.dumps([session.user_id, session.window], separators=(",", ":")),
            ))

    train_counts = Counter(
        item
        for example in prepared["train"]
        for item in (*example.input_item_ids, example.target_item_id)
        if item != 0
    )
    popularity = sorted(catalog.values(), key=lambda item: (-train_counts[item], item))
    manifest = {
        "protocol": "tiny-jsonl-fixed24h-v1",
        "timestamp_unit": "milliseconds",
        "interactions_sha256": digest.hexdigest(),
        "window_ms": DAY_MS,
        "anchor": "first_event_per_user",
        "duplicates": "preserved; equal timestamps retain input row order",
        "boundary_method": "nearest_rank_raw_session_end; upper_boundary_inclusive",
        "train_end": train_end,
        "valid_end": valid_end,
        "train_ratio": train_ratio,
        "valid_ratio": valid_ratio,
        "max_session_length_including_target": max_session_length,
        "min_user_events": min_user_events,
        "min_item_events": min_item_events,
        "frequency_filter": "single_pass_user_then_item_train_only_before_truncation",
        "raw_interactions": len(records),
        "raw_sessions": len(raw_sessions),
        "short_sessions_removed": len(raw_sessions) - len(sessions),
        "crossing_sessions_removed": crossing,
        "raw_split_sessions": {split: len(value) for split, value in raw_splits.items()},
        "training_user_sessions_removed": removed_user_sessions,
        "training_unknown_target_removed": removed_target,
        "training_no_known_history_removed": removed_history,
        "split_examples": {split: len(value) for split, value in prepared.items()},
        "oov_targets": {
            split: sum(example.target_item_id == 0 for example in value)
            for split, value in prepared.items()
        },
        "catalog_size": len(catalog),
        "train_item_counts": {str(item): count for item, count in sorted(train_counts.items())},
    }
    return PreparedData(**prepared, catalog=catalog, popularity=popularity, manifest=manifest)


def prefix_to_local_graph(prefixes: Iterable[Sequence[int]]) -> LocalGraph:
    """Create disjoint session graphs; original-adjacency edges only.

    A known repeated click shares its session's node. Unknown and padded
    positions map to dummy node 0. There are no artificial edges across an
    unknown position. True repeated-click self transitions are retained.
    """
    rows = [list(prefix) for prefix in prefixes]
    for row in rows:
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in row):
            raise ValueError("prefix item IDs must be nonnegative integers")
    width = max(1, max((len(row) for row in rows), default=0))
    node_item_ids = [0]
    edges: set[tuple[int, int]] = set()
    position_to_node = []
    position_mask = []
    has_known_history = []
    for row in rows:
        session_nodes: dict[int, int] = {}
        for item in row:
            if item and item not in session_nodes:
                session_nodes[item] = len(node_item_ids)
                node_item_ids.append(item)
        positions = [session_nodes.get(item, 0) for item in row]
        for source, destination in zip(positions, positions[1:]):
            if source and destination:
                edges.add((source, destination))
        mask = [item != 0 for item in row]
        padding = width - len(row)
        position_to_node.append(positions + [0] * padding)
        position_mask.append(mask + [False] * padding)
        has_known_history.append(any(mask))
    ordered_edges = sorted(edges)
    return LocalGraph(
        node_item_ids=node_item_ids,
        edge_index=([edge[0] for edge in ordered_edges], [edge[1] for edge in ordered_edges]),
        position_to_node=position_to_node,
        position_mask=position_mask,
        has_known_history=has_known_history,
    )
