"""Exploratory profiling of the two raw CORA.RDR JSON files.

Runs on the real (large) files with the standard library only -- no Spark, no
extra dependencies -- so it can be used to design the parsing logic *before*
the pipeline exists. Nothing here is imported by the pipeline itself.

Usage:
    python scripts/explore_raw.py                      # default data/raw paths
    python scripts/explore_raw.py --raw-dir tests/fixtures
    python scripts/explore_raw.py --json-out report.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# The published labels file has been distributed under both names.
LABEL_FILE_CANDIDATES = ("labelled_alternatives.json", "labelled_dataset.json")
FEATURE_FILE_CANDIDATES = ("frozen_dataset.json",)

DIGITS = re.compile(r"\d+")


def resolve(raw_dir: Path, candidates: tuple[str, ...]) -> Path | None:
    for name in candidates:
        path = raw_dir / name
        if path.exists():
            return path
    return None


def id_pattern(scenario_id: str) -> str:
    """Abstract an id into a shape, e.g. 'control_health_1068' -> 'control_health_<n>'."""
    return DIGITS.sub("<n>", scenario_id)


def id_prefix(scenario_id: str) -> str:
    """Leading non-numeric token of an id, or '<numeric>' when it has none."""
    head = scenario_id.split("_")[0]
    return "<numeric>" if DIGITS.fullmatch(head) else head


def describe_lengths(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
        return {"count": 0}
    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": round(statistics.fmean(lengths), 3),
        "histogram": dict(sorted(Counter(lengths).items())),
    }


def profile_file(path: Path, id_key: str, array_key: str) -> dict[str, Any]:
    """Load one file and summarise ids, array lengths and per-scenario product ids."""
    print(f"  loading {path.name} ({path.stat().st_size / 1e6:.1f} MB) ...", flush=True)
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)

    if not isinstance(records, list):
        raise SystemExit(
            f"{path.name}: expected a top-level JSON array, got {type(records).__name__}"
        )

    ids: list[str] = []
    lengths: list[int] = []
    prod_ids: dict[str, list[str]] = {}
    field_presence: Counter[str] = Counter()
    nested_keys: Counter[str] = Counter()
    scalar_ranges: dict[str, list[float]] = {}
    aux_lengths: dict[str, list[int]] = {}
    missing_array = 0

    for record in records:
        scenario_id = str(record.get(id_key))
        ids.append(scenario_id)
        field_presence.update(record.keys())

        for key in ("stakeholder_preference", "situations"):
            if isinstance(record.get(key), list):
                aux_lengths.setdefault(key, []).append(len(record[key]))

        items = record.get(array_key)
        if not isinstance(items, list):
            missing_array += 1
            continue
        lengths.append(len(items))
        prod_ids[scenario_id] = [str(item.get("id_prod")) for item in items]

        for item in items:
            nested_keys.update(item.keys())
            for key, value in item.items():
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    scalar_ranges.setdefault(key, []).append(float(value))
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)) and not isinstance(sub_value, bool):
                            scalar_ranges.setdefault(f"{key}.{sub_key}", []).append(float(sub_value))

    del records  # the real files are large; release before the next one loads

    return {
        "path": str(path),
        "size_mb": round(path.stat().st_size / 1e6, 1),
        "scenario_count": len(ids),
        "unique_scenario_ids": len(set(ids)),
        "duplicate_scenario_ids": [i for i, n in Counter(ids).items() if n > 1][:20],
        "field_presence": dict(field_presence),
        "item_field_presence": dict(nested_keys),
        "missing_array": missing_array,
        "array_lengths": describe_lengths(lengths),
        "aux_lengths": {k: describe_lengths(v) for k, v in aux_lengths.items()},
        "scalar_ranges": {
            key: {
                "min": min(values),
                "max": max(values),
                "mean": round(statistics.fmean(values), 4),
            }
            for key, values in sorted(scalar_ranges.items())
        },
        "_ids": ids,
        "_prod_ids": prod_ids,
    }


def id_taxonomy(ids: list[str]) -> dict[str, Any]:
    patterns = Counter(id_pattern(i) for i in ids)
    prefixes = Counter(id_prefix(i) for i in ids)
    examples: dict[str, str] = {}
    for scenario_id in ids:
        examples.setdefault(id_pattern(scenario_id), scenario_id)
    return {
        "prefixes": dict(prefixes.most_common()),
        "patterns": [
            {"pattern": pattern, "count": count, "example": examples[pattern]}
            for pattern, count in patterns.most_common()
        ],
    }


def compare(labels: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    label_ids = set(labels["_ids"])
    feature_ids = set(features["_ids"])
    only_labels = sorted(label_ids - feature_ids)
    only_features = sorted(feature_ids - label_ids)

    prod_mismatches = []
    for scenario_id in sorted(label_ids & feature_ids):
        left = set(labels["_prod_ids"].get(scenario_id, []))
        right = set(features["_prod_ids"].get(scenario_id, []))
        if left != right:
            prod_mismatches.append(
                {
                    "scenario_id": scenario_id,
                    "only_in_labels": sorted(left - right),
                    "only_in_features": sorted(right - left),
                }
            )

    return {
        "scenarios_in_both": len(label_ids & feature_ids),
        "only_in_labels": {"count": len(only_labels), "examples": only_labels[:20]},
        "only_in_features": {"count": len(only_features), "examples": only_features[:20]},
        "id_prod_mismatches": {"count": len(prod_mismatches), "examples": prod_mismatches[:20]},
    }


def print_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw", type=Path)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--top-patterns", type=int, default=40)
    args = parser.parse_args(argv)

    label_path = resolve(args.raw_dir, LABEL_FILE_CANDIDATES)
    feature_path = resolve(args.raw_dir, FEATURE_FILE_CANDIDATES)

    if label_path is None or feature_path is None:
        print(f"No raw files found under {args.raw_dir.resolve()}.")
        print(f"  expected one of {LABEL_FILE_CANDIDATES} and {FEATURE_FILE_CANDIDATES}")
        print("These files are not committed (CC BY-NC 4.0); download them from")
        print("https://doi.org/10.34810/DATA3164 and place them in data/raw/.")
        return 1

    print(f"Profiling raw files in {args.raw_dir.resolve()}")
    labels = profile_file(label_path, "id", "labelled_alternatives")
    features = profile_file(feature_path, "id", "alternatives")

    print_section("SCENARIO COUNTS")
    for name, prof in (("labels", labels), ("features", features)):
        print(
            f"{name:>9}: {prof['scenario_count']:>7} scenarios "
            f"({prof['unique_scenario_ids']} unique, {prof['size_mb']} MB)"
        )
        if prof["duplicate_scenario_ids"]:
            print(f"           duplicate ids: {prof['duplicate_scenario_ids']}")

    print_section("ARRAY LENGTHS")
    print("labelled_alternatives:", labels["array_lengths"])
    print("alternatives         :", features["array_lengths"])
    for key, stats in features["aux_lengths"].items():
        print(f"{key:<21}:", stats)

    print_section("ID TAXONOMY (labels file)")
    taxonomy = id_taxonomy(labels["_ids"])
    print("prefixes:")
    for prefix, count in taxonomy["prefixes"].items():
        print(f"  {prefix:<28} {count:>7}")
    print(f"\npatterns (top {args.top_patterns}):")
    for row in taxonomy["patterns"][: args.top_patterns]:
        print(f"  {row['pattern']:<40} {row['count']:>7}  e.g. {row['example']}")
    if len(taxonomy["patterns"]) > args.top_patterns:
        print(f"  ... {len(taxonomy['patterns']) - args.top_patterns} more patterns")

    features_taxonomy = id_taxonomy(features["_ids"])
    if features_taxonomy["prefixes"] != taxonomy["prefixes"]:
        print("\nNOTE: features file prefixes differ from labels file:")
        print(" ", features_taxonomy["prefixes"])

    print_section("CROSS-FILE ID CONSISTENCY")
    comparison = compare(labels, features)
    print(f"scenarios in both files : {comparison['scenarios_in_both']}")
    print(
        f"only in labels          : {comparison['only_in_labels']['count']} "
        f"{comparison['only_in_labels']['examples']}"
    )
    print(
        f"only in features        : {comparison['only_in_features']['count']} "
        f"{comparison['only_in_features']['examples']}"
    )
    print(f"id_prod set mismatches  : {comparison['id_prod_mismatches']['count']}")
    for row in comparison["id_prod_mismatches"]["examples"][:10]:
        print(f"  {row}")

    print_section("FIELD PRESENCE")
    print("labels record keys   :", labels["field_presence"])
    print("label item keys      :", labels["item_field_presence"])
    print("features record keys :", features["field_presence"])
    print("feature item keys    :", features["item_field_presence"])

    print_section("NUMERIC RANGES")
    for source, prof in (("labels", labels), ("features", features)):
        for key, stats in prof["scalar_ranges"].items():
            print(
                f"  {source:<9} {key:<26} min={stats['min']:<12} "
                f"max={stats['max']:<12} mean={stats['mean']}"
            )

    if args.json_out:
        report = {
            "labels": {k: v for k, v in labels.items() if not k.startswith("_")},
            "features": {k: v for k, v in features.items() if not k.startswith("_")},
            "labels_id_taxonomy": taxonomy,
            "features_id_taxonomy": features_taxonomy,
            "cross_file": comparison,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report to {args.json_out}")

    return 0


if __name__ == "__main__":
    # Exit non-zero only on failure. `sys.exit(0)` would raise SystemExit, and
    # Databricks runs a job's python_file through exec() -- there any SystemExit
    # escaping the script is reported as a task failure, even for status 0.
    _status = main()
    if _status:
        sys.exit(_status)
