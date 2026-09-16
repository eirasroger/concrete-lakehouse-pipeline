"""Sanity-check gold_scenarios: does `pref` move the way it should?

Not a model, and not a claim about concrete. It correlates `pref` against gwp,
health and circ_orig, broken down by source_type, and checks only the *sign* of
each relationship. A wrong sign means the pipeline joined labels to the wrong
features -- exactly the failure a row count cannot catch.

    python scripts/validate_gold.py
    python scripts/validate_gold.py --fixtures

Exits non-zero if any correlation points the wrong way.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401  (adds src/ to sys.path)

from pyspark.sql import functions as F

from concrete_pipeline import config as cfg
from concrete_pipeline.bronze import read_delta
from concrete_pipeline.session import get_spark
from concrete_pipeline.validation import (
    EXPECTED_DIRECTIONS,
    MIN_ROWS_FOR_DIRECTION,
    correlations,
    format_summary,
    safe_corr,
    wrong_sign_checks,
)


def print_control_axis_detail(gold) -> None:
    """Control scenarios vary one axis by construction -- show that it shows up.

    Each `control_<axis>` family holds every other variable roughly fixed, so
    its own axis should dominate while the others sit near zero. This is the
    sharpest available evidence that the join is sound.
    """
    controls = gold.filter(F.col("source_type") == "control_synthetic")
    if controls.limit(1).count() == 0:
        return

    axis = F.regexp_extract(F.col("scenario_id"), r"^control_([a-z][a-z_]*[a-z])_\d+$", 1)
    by_axis = (
        controls.withColumn("control_axis", axis)
        .groupBy("control_axis")
        .agg(
            F.count(F.lit(1)).alias("rows"),
            *[safe_corr("pref", variable).alias(variable) for variable in EXPECTED_DIRECTIONS],
        )
        .orderBy("control_axis")
    )

    header = (
        f"\n{'control_axis':<20} {'rows':>9} "
        + " ".join(f"{variable:>12}" for variable in EXPECTED_DIRECTIONS)
    )
    print("\nControl families, correlation of pref with each variable")
    print(header)
    print("-" * len(header.strip()))
    for row in by_axis.collect():
        values = " ".join(
            f"{'n/a':>12}" if row[v] is None else f"{row[v]:>+12.4f}" for v in EXPECTED_DIRECTIONS
        )
        print(f"{row['control_axis']:<20} {row['rows']:>9,} {values}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lakehouse-dir", type=Path, default=None)
    parser.add_argument(
        "--namespace",
        default=None,
        help="Unity Catalog catalog.schema to read from instead of a local folder",
    )
    parser.add_argument("--fixtures", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also fail when a correlation is flat rather than merely mis-signed",
    )
    args = parser.parse_args(argv)

    lakehouse = args.lakehouse_dir or cfg.REPO_ROOT / "data" / "lakehouse"
    config = (
        cfg.PipelineConfig.for_fixtures(lakehouse)
        if args.fixtures
        else cfg.PipelineConfig(lakehouse_dir=lakehouse)
    )
    if args.namespace:
        config = replace(config, namespace=args.namespace)

    spark = get_spark()
    try:
        gold = read_delta(spark, config, cfg.GOLD_SCENARIOS)
        total = gold.count()
        print(f"gold_scenarios: {total:,} rows from {config.table_ref(cfg.GOLD_SCENARIOS)}")
        if total == 0:
            print("Table is empty -- run scripts/run_pipeline.py first.")
            return 1

        checks = correlations(gold)
        print("\nDirection of pref against each sustainability variable")
        print(format_summary(checks))
        print_control_axis_detail(gold)

        wrong = wrong_sign_checks(checks)
        if wrong:
            print(f"\nFAIL: {len(wrong)} correlation(s) point the wrong way:")
            for check in wrong:
                print(f"  {check.source_type}/{check.variable}: {check.correlation:+.4f}")
            return 1

        inconclusive = [
            check for check in checks if check.verdict in {"FLAT", "NO DATA", "TOO FEW"}
        ]
        print("\nPASS: every directional correlation has the expected sign.")
        if inconclusive:
            print(
                f"      ({len(inconclusive)} inconclusive: flat, undefined, or under "
                f"{MIN_ROWS_FOR_DIRECTION} rows -- expected for control families "
                "and for --fixtures)"
            )
            if args.strict:
                return 1
        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
