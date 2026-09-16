"""Delta time travel: ship a bad classifier, fix it, and diff the two versions.

Fixing a classification bug should not destroy the evidence of what the table
used to say. The demo:

  1. builds `gold_scenarios` with a deliberately defective classifier (v0),
  2. overwrites it with the corrected classifier (v1),
  3. prints the Delta history, and
  4. diffs v0 against v1 with time travel, showing which rows moved.

The defective rule has the two mistakes that are easy to make if you guess the
id taxonomy from a few samples instead of profiling the files:

  * no `expert_` rule, so all 272 expert scenarios fall into the
    `llm_generated` catch-all;
  * a closed axis list that omits `archfinish_slump`, so 3,000 control
    scenarios miss the control pattern and land in the same catch-all.

Usage:
    python scripts/delta_versioning_demo.py              # real files
    python scripts/delta_versioning_demo.py --fixtures   # 5-scenario fixture
    python scripts/delta_versioning_demo.py --namespace workspace.concrete
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401  (adds src/ to sys.path)

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from concrete_pipeline import config as cfg
from concrete_pipeline.bronze import delta_table, latest_version, read_delta_version
from concrete_pipeline.pipeline import run_pipeline
from concrete_pipeline.session import get_spark
from concrete_pipeline.source_type import naive_source_type_column, source_type_column

KEYS = ["scenario_id", "id_prod"]


def banner(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def print_history(spark: SparkSession, config: cfg.PipelineConfig) -> None:
    history = (
        delta_table(spark, config, cfg.GOLD_SCENARIOS)
        .history()
        .select("version", "timestamp", "operation", "operationMetrics")
        .orderBy("version")
    )
    print(f"\n{'version':>7}  {'operation':<12} {'rows':>10}  timestamp")
    print("-" * 62)
    for row in history.collect():
        metrics = row["operationMetrics"] or {}
        print(
            f"{row['version']:>7}  {row['operation']:<12} "
            f"{str(metrics.get('numOutputRows', '?')):>10}  {row['timestamp']}"
        )


def _mix(df: DataFrame, column: str) -> dict[str, int]:
    """source_type -> row count, as a plain dict.

    Uses the DataFrame API rather than `.rdd`: Databricks serverless compute runs
    on Spark Connect, where the RDD API is not available.
    """
    return {row[column]: row["count"] for row in df.groupBy(column).count().collect()}


def diff_versions(
    spark: SparkSession, config: cfg.PipelineConfig, old: int, new: int
) -> None:
    """Compare `source_type` between two Delta versions of gold_scenarios."""
    before = read_delta_version(spark, config, cfg.GOLD_SCENARIOS, old).select(
        *KEYS, F.col("source_type").alias("before")
    )
    after = read_delta_version(spark, config, cfg.GOLD_SCENARIOS, new).select(
        *KEYS, F.col("source_type").alias("after")
    )

    joined = before.join(after, on=KEYS, how="fullouter").cache()

    changed = joined.filter(
        F.col("before").isNotNull()
        & F.col("after").isNotNull()
        & (F.col("before") != F.col("after"))
    )
    added = joined.filter(F.col("before").isNull())
    removed = joined.filter(F.col("after").isNull())

    changed_count = changed.count()
    print(
        f"\nRows: {changed_count:,} reclassified, {added.count():,} added, "
        f"{removed.count():,} removed"
    )

    if changed_count:
        print(f"\n{'before':<20} -> {'after':<20} {'rows':>10}  {'scenarios':>10}")
        print("-" * 66)
        transitions = (
            changed.groupBy("before", "after")
            .agg(
                F.count(F.lit(1)).alias("rows"),
                F.countDistinct("scenario_id").alias("scenarios"),
            )
            .orderBy(F.col("rows").desc())
        )
        for row in transitions.collect():
            print(
                f"{row['before']:<20} -> {row['after']:<20} "
                f"{row['rows']:>10,}  {row['scenarios']:>10,}"
            )

        print("\nSample of reclassified scenarios")
        print("-" * 66)
        for row in (
            changed.select("scenario_id", "before", "after")
            .distinct()
            .orderBy("scenario_id")
            .limit(10)
            .collect()
        ):
            print(f"  {row['scenario_id']:<32} {row['before']:<20} -> {row['after']}")

    print("\nsource_type totals, side by side")
    print("-" * 66)
    before_mix = _mix(before, "before")
    after_mix = _mix(after, "after")
    print(f"{'source_type':<24} {'v' + str(old):>12} {'v' + str(new):>12} {'delta':>12}")
    for source_type in sorted(set(before_mix) | set(after_mix)):
        old_count = before_mix.get(source_type, 0)
        new_count = after_mix.get(source_type, 0)
        print(
            f"{source_type:<24} {old_count:>12,} {new_count:>12,} "
            f"{new_count - old_count:>+12,}"
        )

    joined.unpersist()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=cfg.REPO_ROOT / "data" / "raw")
    parser.add_argument("--lakehouse-dir", type=Path, default=None)
    parser.add_argument(
        "--namespace",
        default=None,
        help="Unity Catalog catalog.schema to write to instead of a local folder",
    )
    parser.add_argument("--fixtures", action="store_true")
    args = parser.parse_args(argv)

    lakehouse = args.lakehouse_dir or cfg.REPO_ROOT / "data" / "lakehouse_versioning_demo"
    if args.fixtures:
        config = cfg.PipelineConfig.for_fixtures(lakehouse)
    else:
        config = cfg.PipelineConfig(raw_dir=args.raw_dir, lakehouse_dir=lakehouse)
    if args.namespace:
        config = replace(config, namespace=args.namespace)

    spark = get_spark(app_name="concrete-delta-versioning-demo")
    try:
        banner("STEP 1  Build gold_scenarios with the DEFECTIVE classifier")
        print("  no expert_ rule; control axis list omits archfinish_slump")
        first = run_pipeline(spark, config, classifier=naive_source_type_column)
        print(f"  wrote {first.gold_scenarios.count():,} rows")
        version_before = latest_version(spark, config, cfg.GOLD_SCENARIOS)

        banner("STEP 2  Overwrite with the CORRECTED classifier")
        print("  expert_<n> recognised; any control_<axis>_<n> shape accepted")
        second = run_pipeline(spark, config, classifier=source_type_column)
        print(f"  wrote {second.gold_scenarios.count():,} rows")
        version_after = latest_version(spark, config, cfg.GOLD_SCENARIOS)

        banner("STEP 3  Delta history for gold_scenarios")
        print_history(spark, config)

        banner(f"STEP 4  Time-travel diff: v{version_before} -> v{version_after}")
        diff_versions(spark, config, version_before, version_after)

        ref = config.table_ref(cfg.GOLD_SCENARIOS)
        clause = "table" if config.uses_catalog else "load"
        print(
            f"\nThe old classification is still readable:\n"
            f'  spark.read.format("delta").option("versionAsOf", {version_before})'
            f'.{clause}("{ref}")\n'
        )
        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
