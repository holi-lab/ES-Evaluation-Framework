#!/usr/bin/env python3
"""Count dialogues per feature/label in the step13 split dataset."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple, Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count how many dialogues appear for each feature label."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory that contains train/val/test jsonl files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save the aggregated counts (default: data_analysis/feature_label_count/<input_dir>/feature_label_counts.json).",
    )
    parser.add_argument(
        "--feature-key",
        type=str,
        default="seeker_features",
        help="Top-level key that holds the feature dictionary in each JSON line.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse {path}:{line_no}") from exc


def iter_leaf_items(obj: Any, prefix: str = "") -> Iterable[Tuple[str, Any]]:
    """Yield flattened feature paths and their values."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            yield from iter_leaf_items(value, child_prefix)
    else:
        yield prefix, obj


def normalize_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        # Keep a readable float representation without excessive precision.
        return f"{value:.6g}"
    return str(value)


def default_output_path(input_dir: Path) -> Path:
    """Build default output path under data_analysis/feature_label_count mirroring input path."""
    base = Path("feature_label_counts")
    try:
        rel = input_dir.resolve().relative_to(Path.cwd())
    except ValueError:
        rel = Path(input_dir.resolve().as_posix().replace("/", "__"))
    return base / rel / "feature_label_counts.json"


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_path: Path = args.output or default_output_path(input_dir)

    split_files = {
        "train": input_dir / "train.jsonl",
        "val": input_dir / "val.jsonl",
        "test": input_dir / "test.jsonl",
    }

    counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    split_totals: Dict[str, int] = defaultdict(int)
    processed_splits = []

    for split_name, file_path in split_files.items():
        if not file_path.exists():
            continue
        processed_splits.append(split_name)
        for record in load_jsonl(file_path):
            features = record.get(args.feature_key, {})
            if not isinstance(features, dict):
                continue
            split_totals[split_name] += 1
            for feature_path, value in iter_leaf_items(features):
                label = normalize_value(value)
                counts[feature_path][label][split_name] += 1
                counts[feature_path][label]["all"] += 1

    # Convert defaultdicts to plain dicts for JSON serialization.
    serializable_counts = {
        feature: {
            label: dict(split_counts) for label, split_counts in labels.items()
        }
        for feature, labels in counts.items()
    }

    payload = {
        "meta": {
            "input_dir": str(input_dir),
            "feature_key": args.feature_key,
            "processed_splits": processed_splits,
            "split_totals": dict(split_totals),
        },
        "counts": serializable_counts,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved feature/label dialogue counts to: {output_path}")


if __name__ == "__main__":
    main()
