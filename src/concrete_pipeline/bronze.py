"""Bronze layer: land raw files in Delta, unchanged, append-only.

No reshaping. The nested arrays stay nested; the only added columns are
provenance (`_source_file`, `_file_hash`, `_batch_id`, `_ingested_at`) and the
`_rescued_data` column that catches anything the schema did not expect.

Bronze is **append-only**, so it accumulates every version of every file ever
ingested. That is the point: when a re-release replaces `frozen_dataset.json`,
the previous contents are still there to diff against. `current_bronze()`
resolves the newest version per file name for downstream layers.
"""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from . import config as cfg
from . import ingest
from .schemas import FEATURES_SCHEMA, LABELS_SCHEMA


def table_exists(spark: SparkSession, config: cfg.PipelineConfig, table: str) -> bool:
    """Whether a table has been created yet -- true after the first run."""
    ref = config.table_ref(table)
    if config.uses_catalog:
        return spark.catalog.tableExists(ref)
    return (Path(ref) / "_delta_log").exists()


def write_delta(df: DataFrame, config: cfg.PipelineConfig, table: str) -> str:
    """Overwrite a Delta table. Used for derived tables rebuilt in full."""
    return _write(df, config, table, mode="overwrite")


def append_delta(df: DataFrame, config: cfg.PipelineConfig, table: str) -> str:
    """Append to a Delta table, creating it on first use."""
    return _write(df, config, table, mode="append")


def _write(df: DataFrame, config: cfg.PipelineConfig, table: str, mode: str) -> str:
    ref = config.table_ref(table)
    writer = df.write.format("delta").mode(mode)
    if mode == "overwrite":
        writer = writer.option("overwriteSchema", "true")
    else:
        # A later release may add a field; let it land rather than fail the run.
        writer = writer.option("mergeSchema", "true")
    if config.uses_catalog:
        writer.saveAsTable(ref)
    else:
        writer.save(ref)
    return ref


def read_delta(spark: SparkSession, config: cfg.PipelineConfig, table: str) -> DataFrame:
    """Read a Delta table written by this pipeline."""
    if config.uses_catalog:
        return spark.table(config.table_ref(table))
    return spark.read.format("delta").load(config.table_ref(table))


def delta_table(spark: SparkSession, config: cfg.PipelineConfig, table: str):
    """The `DeltaTable` handle, for history, time travel and merges."""
    from delta.tables import DeltaTable

    ref = config.table_ref(table)
    if config.uses_catalog:
        return DeltaTable.forName(spark, ref)
    return DeltaTable.forPath(spark, ref)


def latest_version(spark: SparkSession, config: cfg.PipelineConfig, table: str) -> int:
    """The version number of the most recent commit to a table."""
    return delta_table(spark, config, table).history(1).collect()[0]["version"]


def read_delta_version(
    spark: SparkSession, config: cfg.PipelineConfig, table: str, version: int
) -> DataFrame:
    """Read a table as it was at a given Delta version (time travel)."""
    reader = spark.read.format("delta").option("versionAsOf", version)
    if config.uses_catalog:
        return reader.table(config.table_ref(table))
    return reader.load(config.table_ref(table))


def enable_change_feed(spark: SparkSession, config: cfg.PipelineConfig, table: str) -> None:
    """Turn on Change Data Feed so downstream layers can see what moved.

    Set after creation rather than at write time: the property has to exist on
    the table before the commit whose changes you want to read.
    """
    spark.sql(
        f"ALTER TABLE {_sql_ref(config, table)} "
        "SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
    )


def _sql_ref(config: cfg.PipelineConfig, table: str) -> str:
    """How to name a table inside a SQL statement."""
    if config.uses_catalog:
        return config.table_ref(table)
    return f"delta.`{config.table_ref(table)}`"


def current_bronze(df: DataFrame) -> DataFrame:
    """Keep only the newest ingested version of each source file.

    Bronze holds every version ever landed. Downstream wants one: the latest
    batch per file name. Ranking by `_batch_id` (a UTC timestamp string) rather
    than `_ingested_at` keeps this deterministic when several files land inside
    the same second.
    """
    newest = Window.partitionBy("_source_file").orderBy(F.col("_batch_id").desc())
    return (
        df.withColumn("_rank", F.dense_rank().over(newest))
        .filter(F.col("_rank") == 1)
        .drop("_rank")
    )


def ingest_bronze(
    spark: SparkSession, config: cfg.PipelineConfig
) -> tuple[dict[str, DataFrame], list[ingest.SourceFile], str]:
    """Land any raw file whose content has not been seen before.

    Returns the current bronze tables, the files consumed by this batch (empty
    when there was nothing new), and the batch id.
    """
    batch_id = ingest.new_batch_id()
    sources = ingest.describe_sources(config)
    pending = ingest.pending_sources(spark, config, sources)

    by_kind = {
        "labels": ([s for s in pending if s.kind == "labels"], LABELS_SCHEMA, cfg.BRONZE_LABELS),
        "features": (
            [s for s in pending if s.kind == "features"],
            FEATURES_SCHEMA,
            cfg.BRONZE_FEATURES,
        ),
    }

    row_counts: dict[str, int] = {}
    for _, (files, schema, table) in by_kind.items():
        frame = ingest.read_sources(spark, config, files, schema, batch_id)
        if frame is None:
            continue

        append_delta(frame, config, table)

        # Count from the table just written, not from the source frame.
        # Counting first would need the frame cached to avoid re-parsing the
        # JSON, and `cache()` is rejected outright on Databricks serverless
        # ("PERSIST TABLE is not supported"). Reading back this batch's rows is
        # a cheap columnar scan and works on every runtime.
        written = read_delta(spark, config, table).filter(F.col("_batch_id") == batch_id)
        for row in written.groupBy("_source_file").count().collect():
            row_counts[row["_source_file"]] = row["count"]

    ingest.record_ingestion(spark, config, pending, row_counts, batch_id)

    return (
        {
            cfg.BRONZE_LABELS: current_bronze(read_delta(spark, config, cfg.BRONZE_LABELS)),
            cfg.BRONZE_FEATURES: current_bronze(read_delta(spark, config, cfg.BRONZE_FEATURES)),
        },
        pending,
        batch_id,
    )
