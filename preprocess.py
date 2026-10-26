from __future__ import annotations

import argparse
import ast
import gzip
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Interaction:
    user: str
    item: str
    timestamp: int
    source_line_number: int


def parse_line(line: str, release: int) -> dict[str, Any]:
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        if release != 2014:
            raise
        record = ast.literal_eval(line)
    if not isinstance(record, dict):
        raise ValueError("Amazon JSONL record must be an object")
    return record


def iter_gzip_records(path: Path, release: int) -> Iterator[tuple[int, dict[str, Any]]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield line_number, parse_line(line, release)


def read_interactions(path: Path, release: int) -> list[Interaction]:
    interactions = []
    for line_number, record in iter_gzip_records(path, release):
        try:
            interaction = Interaction(
                user=str(record["reviewerID"]),
                item=str(record["asin"]),
                timestamp=int(record["unixReviewTime"]),
                source_line_number=line_number,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid interaction at {path}:{line_number}") from exc
        interactions.append(interaction)
    return interactions


def read_metadata(
    path: Path, release: int, strict_jsonl_path: Path
) -> dict[str, dict[str, Any]]:
    retained_fields = (
        "asin", "title", "description", "price", "brand", "categories", "category"
    )
    metadata: dict[str, dict[str, Any]] = {}
    with strict_jsonl_path.open("w", encoding="utf-8") as output:
        for _, record in iter_gzip_records(path, release):
            output.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            asin = record.get("asin")
            if asin is None:
                continue
            metadata.setdefault(
                str(asin),
                {field: record[field] for field in retained_fields if field in record},
            )
    return metadata


def filter_by_metadata(
    interactions: list[Interaction], metadata_items: set[str]
) -> list[Interaction]:
    return [row for row in interactions if row.item in metadata_items]


def deduplicate_user_items(interactions: list[Interaction]) -> list[Interaction]:
    earliest: dict[tuple[str, str], Interaction] = {}
    for row in interactions:
        key = (row.user, row.item)
        current = earliest.get(key)
        if current is None or (row.timestamp, row.source_line_number) < (
            current.timestamp,
            current.source_line_number,
        ):
            earliest[key] = row
    return sorted(earliest.values(), key=lambda row: row.source_line_number)


def iterative_k_core(interactions: list[Interaction], k: int = 5) -> list[Interaction]:
    rows = interactions
    while True:
        before = len(rows)
        item_counts = Counter(row.item for row in rows)
        removed_items = {item for item, count in item_counts.items() if count < k}
        rows = [row for row in rows if row.item not in removed_items]
        user_counts = Counter(row.user for row in rows)
        removed_users = {user for user, count in user_counts.items() if count < k}
        rows = [row for row in rows if row.user not in removed_users]
        if len(rows) == before:
            return rows


def first_seen_mapping(values: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping) + 1
    return mapping


def build_sequences(
    interactions: list[Interaction],
    user_mapping: dict[str, int],
    item_mapping: dict[str, int],
) -> dict[int, list[int]]:
    grouped: dict[str, list[Interaction]] = defaultdict(list)
    for row in interactions:
        grouped[row.user].append(row)
    sequences = {}
    for raw_user, rows in grouped.items():
        rows.sort(key=lambda row: (row.timestamp, row.source_line_number))
        sequences[user_mapping[raw_user]] = [item_mapping[row.item] for row in rows]
    return sequences


def split_sequences(
    sequences: dict[int, list[int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train, valid, test = [], [], []
    for user in sorted(sequences):
        sequence = sequences[user]
        if len(sequence) < 3:
            raise ValueError(f"User {user} has fewer than three interactions")
        train.append({"user": user, "history": sequence[:-3], "target": sequence[-3]})
        valid.append({"user": user, "history": sequence[:-2], "target": sequence[-2]})
        test.append({"user": user, "history": sequence[:-1], "target": sequence[-1]})
    return train, valid, test


def _write_mapping(path: Path, mapping: dict[str, int]) -> None:
    import numpy as np

    with path.open("wb") as handle:
        np.save(handle, mapping)


def _write_split(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            pa.field("user", pa.int64(), nullable=False),
            pa.field("history", pa.list_(pa.int64()), nullable=False),
            pa.field("target", pa.int64(), nullable=False),
        ]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def preprocess_dataset(
    review_path: Path,
    metadata_path: Path,
    release: int,
    output_dir: Path,
    overwrite: bool = False,
) -> Path:
    if release not in (2014, 2018):
        raise ValueError("release must be 2014 or 2018")
    for path in (review_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_dir / filename
        for filename in (
            "metadata.jsonl", "user_mapping.npy", "item_mapping.npy",
            "train.parquet", "valid.parquet", "test.parquet",
        )
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Processed outputs already exist: {existing}")

    with tempfile.TemporaryDirectory(prefix="preprocess_", dir=output_dir) as temp_name:
        temp_dir = Path(temp_name)
        metadata = read_metadata(metadata_path, release, temp_dir / "metadata.jsonl")
        interactions = read_interactions(review_path, release)
        if release == 2018:
            interactions = filter_by_metadata(interactions, set(metadata))
            interactions = deduplicate_user_items(interactions)
            interactions = iterative_k_core(interactions)

        user_counts = Counter(row.user for row in interactions)
        item_counts = Counter(row.item for row in interactions)
        if min(user_counts.values(), default=0) < 5 or min(
            item_counts.values(), default=0
        ) < 5:
            raise ValueError("Processed dataset is empty or is not 5-core")
        missing = set(item_counts) - set(metadata)
        if missing:
            raise ValueError(f"Metadata missing for {len(missing)} final items")

        user_mapping = first_seen_mapping([row.user for row in interactions])
        item_mapping = first_seen_mapping([row.item for row in interactions])
        sequences = build_sequences(interactions, user_mapping, item_mapping)
        splits = split_sequences(sequences)
        _write_mapping(temp_dir / "user_mapping.npy", user_mapping)
        _write_mapping(temp_dir / "item_mapping.npy", item_mapping)
        for filename, rows in zip(
            ("train.parquet", "valid.parquet", "test.parquet"), splits
        ):
            _write_split(temp_dir / filename, rows)
        for output in outputs:
            source = temp_dir / output.name
            if output.exists() and overwrite:
                output.unlink()
            shutil.move(str(source), output)
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--release", type=int, choices=(2014, 2018), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    preprocess_dataset(
        args.reviews, args.metadata, args.release, args.output_dir, args.overwrite
    )
