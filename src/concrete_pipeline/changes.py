"""Finding what changed in silver, using Delta Change Data Feed.

Gold could be told directly which scenarios this batch ingested -- bronze knows.
Reading it from silver's Change Data Feed instead is deliberate: it makes gold
depend on *what actually changed in the table it reads from*, not on what some
earlier stage believes it did.

That matters in the cases that bite. A backfill that corrects one scenario, a
manual `UPDATE` to patch a bad label, a partially failed run that landed silver
but not gold -- in every one of those, the ingestion batch is a lie and the
change feed is the truth. It also means gold stays correct if silver is ever
written by something other than this pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from . import config as cfg
from .bronze import _sql_ref, latest_version, table_exists

#: CDF rows describing data that exists after the change. `update_preimage` is
#: excluded -- it describes the old value of an updated row, and including it
#: would be harmless but redundant, since the postimage has the same key.
CHANGE_TYPES = ("insert", "update_postimage", "delete")


@dataclass(frozen=True)
class TableVersions:
    """The version of each table before a write, for reading changes after it."""

    versions: dict[str, int]

    def before(self, table: str) -> int | None:
        return self.versions.get(table)


def snapshot_versions(
    spark: SparkSession, config: cfg.PipelineConfig, tables: list[str]
) -> TableVersions:
    """Record each table's current version. Absent tables map to nothing."""
    versions: dict[str, int] = {}
    for table in tables:
        if table_exists(spark, config, table):
            versions[table] = latest_version(spark, config, table)
    return TableVersions(versions=versions)


def changed_scenarios(
    spark: SparkSession,
    config: cfg.PipelineConfig,
    tables: list[str],
    since: TableVersions,
) -> list[str] | None:
    """Scenario ids whose rows changed in any of `tables` since `since`.

    Returns None when the change set cannot be determined -- a table that did not
    exist before, or one without Change Data Feed enabled. None means "rebuild
    everything", which is the safe answer: better a slow correct run than a fast
    one that silently skips scenarios.
    """
    found: set[str] = set()

    for table in tables:
        start = since.before(table)
        if start is None:
            return None

        end = latest_version(spark, config, table)
        if end == start:
            continue  # nothing was written to this table

        try:
            changes = (
                spark.read.format("delta")
                .option("readChangeFeed", "true")
                .option("startingVersion", start + 1)
                .option("endingVersion", end)
                .table(config.table_ref(table))
                if config.uses_catalog
                else spark.read.format("delta")
                .option("readChangeFeed", "true")
                .option("startingVersion", start + 1)
                .option("endingVersion", end)
                .load(config.table_ref(table))
            )
        except Exception:
            # CDF was not enabled when those commits were made.
            return None

        rows = (
            changes.filter(F.col("_change_type").isin(list(CHANGE_TYPES)))
            .select("scenario_id")
            .distinct()
            .collect()
        )
        found.update(row["scenario_id"] for row in rows)

    return sorted(found)


def enable_change_feed_on(
    spark: SparkSession, config: cfg.PipelineConfig, tables: list[str]
) -> None:
    """Enable CDF on tables that already exist, ignoring those that do not.

    Idempotent: setting the property twice is a no-op, and it has to be set
    *before* the commits whose changes you intend to read.
    """
    for table in tables:
        if table_exists(spark, config, table):
            spark.sql(
                f"ALTER TABLE {_sql_ref(config, table)} "
                "SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
            )
