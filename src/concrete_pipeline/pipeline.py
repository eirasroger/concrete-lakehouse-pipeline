"""End-to-end orchestration: raw JSON -> bronze -> silver -> gold, incrementally.

One run does this:

1. **Hash** every raw file and compare against `bronze_ingest_log`. Nothing new
   means nothing to do, and the run exits having written nothing.
2. **Append** the unseen files to bronze, tagged with their content hash.
3. **Upsert** silver for the scenarios those files contain -- inserting,
   updating, and deleting rows a scenario has lost.
4. **Read silver's Change Data Feed** to find which scenarios actually moved,
   and recompute gold for exactly those.
5. **Enforce** the gate's limits, failing the run if rejections spike.

Step 4 reads from the table rather than trusting step 3's own account of itself,
so a manual correction to silver, or a previous run that died between silver and
gold, still produces the right gold rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from . import changes
from . import config as cfg
from . import ingest, quality
from .bronze import ingest_bronze, read_delta, table_exists
from .gold import build_gold
from .silver import build_silver
from .source_type import Classifier, source_type_column

#: Silver tables whose change feed drives gold.
SILVER_TABLES = [
    cfg.SILVER_LABELS,
    cfg.SILVER_FEATURES,
    cfg.SILVER_SCENARIO_STAKEHOLDER,
    cfg.SILVER_SCENARIO_SITUATION,
]


@dataclass
class PipelineResult:
    """Every table a run produced, plus what the run actually did."""

    bronze: dict[str, DataFrame]
    silver: dict[str, DataFrame]
    gold: dict[str, DataFrame]
    ingested_files: list[ingest.SourceFile] = field(default_factory=list)
    changed_scenarios: list[str] | None = None
    batch_id: str = ""
    skipped: bool = False

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


def _existing_tables(
    spark: SparkSession, config: cfg.PipelineConfig, names: list[str]
) -> dict[str, DataFrame]:
    return {
        name: read_delta(spark, config, name)
        for name in names
        if table_exists(spark, config, name)
    }


def run_pipeline(
    spark: SparkSession,
    config: cfg.PipelineConfig | None = None,
    classifier: Classifier = source_type_column,
    full_refresh: bool = False,
) -> PipelineResult:
    """Build every layer, touching only what changed.

    `classifier` is injectable so the Delta versioning demo can rebuild gold with
    a deliberately defective rule without touching the pipeline code.
    `full_refresh` reprocesses every scenario already in bronze, which is what a
    classifier change or a backfill needs.
    """
    config = config or cfg.PipelineConfig()
    if not config.uses_catalog:
        config.lakehouse_dir.mkdir(parents=True, exist_ok=True)

    bronze, ingested, batch_id = ingest_bronze(spark, config)

    if not ingested and not full_refresh and table_exists(spark, config, cfg.GOLD_SCENARIOS):
        # Nothing new arrived. Re-deriving identical tables would only churn
        # Delta versions and invalidate downstream caches for no gain.
        return PipelineResult(
            bronze=bronze,
            silver=_existing_tables(spark, config, SILVER_TABLES),
            gold=_existing_tables(
                spark, config, [cfg.GOLD_SCENARIOS, cfg.GOLD_SCENARIOS_REJECTED]
            ),
            batch_id=batch_id,
            changed_scenarios=[],
            skipped=True,
        )

    # CDF has to be on before the commits whose changes we intend to read.
    changes.enable_change_feed_on(spark, config, SILVER_TABLES)
    before = changes.snapshot_versions(spark, config, SILVER_TABLES)

    incoming = bronze
    if not full_refresh and ingested:
        names = [s.name for s in ingested]
        incoming = {
            table: df.filter(F.col("_source_file").isin(names))
            for table, df in bronze.items()
        }

    silver = build_silver(
        spark, config, incoming, classifier=classifier, incremental=not full_refresh
    )

    changed = (
        None
        if full_refresh
        else changes.changed_scenarios(spark, config, SILVER_TABLES, before)
    )

    gold = build_gold(spark, config, silver, scenarios=changed)

    quality.enforce_gate_limits(
        gold[cfg.GOLD_SCENARIOS].count(),
        gold[cfg.GOLD_SCENARIOS_REJECTED],
        config.gate_limits,
    )

    return PipelineResult(
        bronze=bronze,
        silver=silver,
        gold=gold,
        ingested_files=ingested,
        changed_scenarios=changed,
        batch_id=batch_id,
    )


def print_run_report(result: PipelineResult) -> None:
    """Human-readable summary: what was ingested, what moved, what was rejected."""
    print("\n" + "=" * 68)
    print("PIPELINE RUN REPORT")
    print("=" * 68)

    print(f"\nBatch {result.batch_id}")
    if result.skipped:
        print("  no new or changed source files -- nothing to do")
    elif result.ingested_files:
        for source in result.ingested_files:
            print(
                f"  ingested {source.name}  ({source.byte_size / 1e6:.1f} MB, "
                f"sha256 {source.file_hash[:12]}...)"
            )
    else:
        print("  full refresh from existing bronze")

    if result.changed_scenarios is None:
        print("  scenarios recomputed: all")
    else:
        print(f"  scenarios recomputed: {len(result.changed_scenarios):,}")

    print("\nRow counts")
    for table, count in result.counts().items():
        print(f"  {table:<30} {count:>10,}")

    rejected = result.gold_rejected
    rejected_count = rejected.count()
    print(f"\nQuality gate: {rejected_count:,} row(s) rejected")
    if rejected_count:
        for row in quality.gate_summary(rejected).collect():
            print(f"  {row['rejection_reason']:<36} {row['count']:>8,}")
        print("\n  Rejected rows:")
        for row in (
            rejected.select("scenario_id", "id_prod", "pref", "conf", "rejection_reasons")
            .orderBy("scenario_id", "id_prod", "pref")
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
        print(
            f"  {row['source_type']:<22} {row['rows']:>10,} rows  "
            f"{row['scenarios']:>8,} scenarios"
        )
    print()
