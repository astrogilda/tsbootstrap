"""Partition the mutation ratchet across CI jobs and merge the pieces back, refusing gaps.

The full ratchet (2,781 mutants at four workers) took four to five and a half hours on one
GitHub-hosted runner. That sat within half an hour of the hosted runners' six-hour job
ceiling, which ``timeout-minutes`` cannot raise, and on 2026-09-25 the runner was lost at
5h38m and the night produced no verdict at all. Splitting the run into shards keeps every
job short. The price is a new way to pass wrongly: a shard that never ran, or a merge that
reads fewer files than it should, must never look like a shard whose mutants all died. So
every shard file records which shard it is, how many shards there are and how many mutants
the full enumeration held, and :func:`merge_shards` refuses anything short of the exact
enumeration, each mutant exactly once.

Stdlib only, so the unit tests can import it without the private Layer-2 package.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

SHARD_FORMAT = 1
STATUSES = frozenset({"killed", "survived", "timeout"})


class ShardMergeError(Exception):
    """The shard files do not add up to one complete run of the enumerated mutants."""


def parse_shard(spec: str) -> tuple[int, int]:
    """Parse ``"K/N"`` into ``(K, N)`` with ``0 <= K < N``."""
    index_text, sep, count_text = spec.partition("/")
    if not sep or not index_text.isdigit() or not count_text.isdigit():
        raise ValueError(f"shard must look like K/N, got {spec!r}")
    index, count = int(index_text), int(count_text)
    if count < 1 or index >= count:
        raise ValueError(f"shard index must satisfy 0 <= K < N, got {spec!r}")
    return index, count


def select_shard(names: Iterable[str], index: int, count: int) -> list[str]:
    """Deterministic round-robin slice of the sorted mutant names.

    Round-robin over the sorted list spreads every module across all shards, so no shard
    inherits one slow module's whole tail.
    """
    return [name for i, name in enumerate(sorted(names)) if i % count == index]


def shard_record(
    index: int, count: int, enumerated: int, outcomes: Mapping[str, str]
) -> dict[str, object]:
    """The JSON document one shard job writes."""
    return {
        "format": SHARD_FORMAT,
        "shard": index,
        "shards": count,
        "enumerated": enumerated,
        "outcomes": dict(sorted(outcomes.items())),
    }


def merge_shards(
    paths: Iterable[Path], expected_shards: int, enumerated: Iterable[str]
) -> dict[str, str]:
    """Merge shard files into one ``{mutant: status}`` map, or raise :class:`ShardMergeError`.

    The merge passes only when there is exactly one file for each shard index in
    ``range(expected_shards)``, every file agrees on the shard count and on the size of the
    enumeration, no mutant appears twice, and the merged names equal ``enumerated`` exactly.
    """
    expected = set(enumerated)
    merged: dict[str, str] = {}
    seen: dict[int, Path] = {}
    for path in sorted(paths):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("format") != SHARD_FORMAT:
            raise ShardMergeError(f"{path}: unknown shard format {record.get('format')!r}")
        index, count = record["shard"], record["shards"]
        if count != expected_shards:
            raise ShardMergeError(f"{path}: written for {count} shards, expected {expected_shards}")
        if record["enumerated"] != len(expected):
            raise ShardMergeError(
                f"{path}: its run enumerated {record['enumerated']} mutants, this one {len(expected)}"
            )
        if index in seen:
            raise ShardMergeError(f"shard {index} appears twice: {seen[index]} and {path}")
        seen[index] = path
        for name, status in record["outcomes"].items():
            if status not in STATUSES:
                raise ShardMergeError(f"{path}: {name} has unknown status {status!r}")
            if name in merged:
                raise ShardMergeError(f"{name} was run by more than one shard")
            merged[name] = status
    missing_shards = sorted(set(range(expected_shards)) - set(seen))
    if missing_shards:
        raise ShardMergeError(f"no outcomes for shard(s) {missing_shards}")
    unrun = sorted(expected - set(merged))
    unknown = sorted(set(merged) - expected)
    if unrun or unknown:
        raise ShardMergeError(
            f"merged outcomes do not match the enumeration: {len(unrun)} never ran "
            f"(first: {unrun[:3]}), {len(unknown)} not enumerated here (first: {unknown[:3]})"
        )
    return merged
