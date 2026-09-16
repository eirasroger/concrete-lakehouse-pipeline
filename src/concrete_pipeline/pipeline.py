"""End-to-end orchestration: raw JSON -> bronze -> silver -> gold."""

from __future__ import annotations

from dataclasses import dataclass

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from . import config as cfg
from .bronze import build_bronze
from .gold import build_gold
from .quality import gate_summary
from .silver import build_silver
from .source_type import Classifier, source_type_column


@dataclass
class PipelineResult:
    """Every table a run produced, keyed by table name."""

    bronze: dict[str, DataFrame]
    silver: dict[str, DataFrame]
    gold: dict[str, DataFrame]

    @property
    def gold_scenarios(self) -> DataFrame:
        return self.gold[cfg.GOLD_SCENARIOS]

    @property
    def gold_rejected(self) -> DataFrame:
        return self.gold[cfg.GOLD_SCENARIOS_REJECTED]

    def counts(self) -> dict[str, int]:
        """Row count per table. Triggers a job per table -- call once, print once."""
        return {
            name: df.count()
            for layer in (self.bronze, self.silver, self.gold)
            for name, df in layer.items()
        }


def run_pipeline(
    spark: SparkSession,
    config: cfg.PipelineConfig | None = None,
    classifier: Classifier = source_type_column,
    infer_schema: bool = False,
) -> PipelineResult:
    """Build every layer in order and return the resulting tables.

    `classifier` is injectable so the Delta versioning demo can rebuild gold with
    a deliberately defective rule without touching the pipeline code.
    """
    config = config or cfg.PipelineConfig()
    if not config.uses_catalog:
        # Only meaningful for the local target. On Databricks the tables go to
        # Unity Catalog, and creating this would write into the Repo checkout.
        config.lakehouse_dir.mkdir(parents=True, exist_ok=True)

    bronze = build_bronze(spark, config, infer_schema=infer_schema)
    silver = build_silver(spark, config, bronze, classifier=classifier)
    gold = build_gold(spark, config, silver)

    return PipelineResult(bronze=bronze, silver=silver, gold=gold)


def print_run_report(result: PipelineResult) -> None:
    """Human-readable summary of a run: row counts, gate outcome, provenance mix."""
    print("\n" + "=" * 68)
    print("PIPELINE RUN REPORT")
    print("=" * 68)

    print("\nRow counts")
    for table, count in result.counts().items():
        print(f"  {table:<30} {count:>10,}")

    rejected = result.gold_rejected
    rejected_count = rejected.count()
    print(f"\nQuality gate: {rejected_count:,} row(s) rejected")
    if rejected_count:
        for row in gate_summary(rejected).collect():
            print(f"  {row['rejection_reason']:<36} {row['count']:>8,}")
        print("\n  Rejected rows:")
        for row in (
            rejected.select("scenario_id", "id_prod", "pref", "conf", "rejection_reasons")
            .orderBy("scenario_id", "id_prod")
            .limit(20)
            .collect()
        ):
            print(
                f"    {row['scenario_id']:<22} {str(row['id_prod']):<8} "
                f"pref={str(row['pref']):<6} conf={str(row['conf']):<6} "
                f"{row['rejection_reasons']}"
            )

    print("\nsource_type mix in gold_scenarios")
    mix = (
        result.gold_scenarios.groupBy("source_type")
        .agg(
            F.count(F.lit(1)).alias("rows"),
            F.countDistinct("scenario_id").alias("scenarios"),
        )
        .orderBy(F.col("rows").desc())
    )
    for row in mix.collect():
        print(f"  {row['source_type']:<22} {row['rows']:>10,} rows  {row['scenarios']:>8,} scenarios")
    print()
