from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _field(record: dict[str, Any], name: str) -> Any:
    aliases = {"categories": ("categories", "category")}.get(name, (name,))
    for alias in aliases:
        value = record.get(alias)
        if value is not None:
            if name == "categories" and isinstance(value, list) and value:
                if isinstance(value[0], list):
                    return value[0]
            return value
    return ""


def _text_value(value: Any) -> str:
    if value is None or (isinstance(value, (list, dict)) and not value):
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def build_item_text(record: dict[str, Any], item_id: int) -> str:
    values = {
        name: _text_value(_field(record, name))
        for name in ("title", "price", "brand", "categories", "description")
    }
    if not any(value.strip() for value in values.values()):
        return f"item_{item_id}"
    return (
        "Title: {title}; Price: {price}; Brand: {brand}; "
        "Categories: {categories}; Description: {description}; "
    ).format(**values)


def build_embeddings(
    metadata_path: Path,
    mapping_path: Path,
    output_path: Path,
    batch_size: int,
    model_name: str = "sentence-transformers/sentence-t5-xxl",
    overwrite: bool = False,
) -> Path:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from sentence_transformers import SentenceTransformer

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Item embeddings already exist: {output_path}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    mapping = np.load(mapping_path, allow_pickle=True).item()
    if not mapping or set(mapping.values()) != set(range(1, len(mapping) + 1)):
        raise ValueError("Item mapping must be nonempty, contiguous, and one-based")
    needed = set(mapping)
    metadata = {}
    with metadata_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            asin = record.get("asin")
            if asin is not None and str(asin) in needed:
                metadata.setdefault(str(asin), record)
    missing = needed - set(metadata)
    if missing:
        raise ValueError(f"Metadata missing for {len(missing)} mapped items")

    ordered = sorted(mapping.items(), key=lambda pair: pair[1])
    texts = [build_item_text(metadata[asin], item_id) for asin, item_id in ordered]
    model = SentenceTransformer(model_name)
    vectors = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
    )
    if vectors.shape != (len(texts), 768):
        raise ValueError(f"Unexpected embedding shape: {vectors.shape}")
    table = pa.table(
        {
            "ItemID": pa.array(range(1, len(texts) + 1), type=pa.int64()),
            "embedding": pa.array(
                vectors.astype("float32").tolist(), type=pa.list_(pa.float32())
            ),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--model", default="sentence-transformers/sentence-t5-xxl")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_embeddings(
        args.metadata, args.mapping, args.output,
        args.batch_size, args.model, args.overwrite,
    )
