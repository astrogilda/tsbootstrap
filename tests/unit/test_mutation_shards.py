"""Pins for the sharded mutation ratchet: a shard that never ran must fail the merge.

The ratchet runs as eight CI jobs plus a gate job. These tests hold the two properties the
split depends on: the shards partition the enumeration exactly, and the merge refuses every
way a partial run could otherwise read as a complete one (a missing shard, a duplicated
shard, a mutant run twice or never, a record from a different enumeration). The module
under test is stdlib-only, so nothing here needs the private Layer-2 package.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tools.mutation_shards import (
    ShardMergeError,
    merge_shards,
    parse_shard,
    select_shard,
    shard_record,
)

ROOT = Path(__file__).resolve().parents[2]
NAMES = [f"tsbootstrap.model.fit.x_f__mutmut_{i}" for i in range(23)]


def _write_shards(tmp_path: Path, count: int, names: list[str] = NAMES) -> list[Path]:
    paths = []
    for k in range(count):
        mine = select_shard(names, k, count)
        path = tmp_path / f"{k}.json"
        path.write_text(
            json.dumps(shard_record(k, count, len(names), dict.fromkeys(mine, "killed")))
        )
        paths.append(path)
    return paths


@pytest.mark.parametrize("count", [1, 2, 3, 8, 23, 30])
def test_shards_partition_the_enumeration_exactly(count: int) -> None:
    slices = [select_shard(reversed(NAMES), k, count) for k in range(count)]
    flat = [n for s in slices for n in s]
    assert sorted(flat) == sorted(NAMES)
    assert len(flat) == len(set(flat))
    assert max(map(len, slices)) - min(map(len, slices)) <= 1


def test_merge_of_complete_shards_returns_every_outcome(tmp_path: Path) -> None:
    merged = merge_shards(_write_shards(tmp_path, 4), 4, NAMES)
    assert merged == dict.fromkeys(NAMES, "killed")


def test_merge_refuses_a_missing_shard(tmp_path: Path) -> None:
    paths = _write_shards(tmp_path, 4)
    with pytest.raises(ShardMergeError, match=r"no outcomes for shard\(s\) \[2\]"):
        merge_shards([p for p in paths if p.name != "2.json"], 4, NAMES)


def test_merge_refuses_a_duplicated_shard(tmp_path: Path) -> None:
    paths = _write_shards(tmp_path, 4)
    copy = tmp_path / "1-again.json"
    copy.write_text(paths[1].read_text())
    with pytest.raises(ShardMergeError, match="shard 1 appears twice"):
        merge_shards([*paths, copy], 4, NAMES)


def test_merge_refuses_a_mutant_run_by_two_shards(tmp_path: Path) -> None:
    paths = _write_shards(tmp_path, 2)
    record = json.loads(paths[1].read_text())
    record["outcomes"][NAMES[0]] = "killed"  # NAMES[0] belongs to shard 0
    paths[1].write_text(json.dumps(record))
    with pytest.raises(ShardMergeError, match="more than one shard"):
        merge_shards(paths, 2, NAMES)


def test_merge_refuses_a_mutant_that_never_ran(tmp_path: Path) -> None:
    paths = _write_shards(tmp_path, 2)
    record = json.loads(paths[0].read_text())
    del record["outcomes"][NAMES[0]]
    paths[0].write_text(json.dumps(record))
    with pytest.raises(ShardMergeError, match="1 never ran"):
        merge_shards(paths, 2, NAMES)


def test_merge_refuses_records_from_a_different_enumeration(tmp_path: Path) -> None:
    paths = _write_shards(tmp_path, 2, NAMES[:-1])
    with pytest.raises(ShardMergeError, match="enumerated 22 mutants, this one 23"):
        merge_shards(paths, 2, NAMES)


def test_merge_refuses_a_different_shard_count(tmp_path: Path) -> None:
    with pytest.raises(ShardMergeError, match="written for 3 shards, expected 4"):
        merge_shards(_write_shards(tmp_path, 3), 4, NAMES)


def test_merge_refuses_an_unknown_status(tmp_path: Path) -> None:
    paths = _write_shards(tmp_path, 1)
    record = json.loads(paths[0].read_text())
    record["outcomes"][NAMES[0]] = "skipped"
    paths[0].write_text(json.dumps(record))
    with pytest.raises(ShardMergeError, match="unknown status 'skipped'"):
        merge_shards(paths, 1, NAMES)


def test_merge_of_no_records_fails(tmp_path: Path) -> None:
    with pytest.raises(ShardMergeError, match="no outcomes for shard"):
        merge_shards([], 8, NAMES)


@pytest.mark.parametrize("spec", ["3", "3/", "/8", "8/8", "-1/8", "0/0", "a/b"])
def test_parse_shard_refuses_malformed_specs(spec: str) -> None:
    with pytest.raises(ValueError, match="shard"):
        parse_shard(spec)


def test_parse_shard_accepts_k_of_n() -> None:
    assert parse_shard("7/8") == (7, 8)


def test_workflow_matrix_matches_its_shard_count() -> None:
    doc = yaml.safe_load((ROOT / ".github/workflows/mutation.yml").read_text(encoding="utf-8"))
    shards = int(doc["env"]["SHARDS"])
    assert doc["jobs"]["shard"]["strategy"]["matrix"]["shard"] == list(range(shards))
    assert doc["jobs"]["gate"]["needs"] == "shard"
