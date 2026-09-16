"""Bronze layer: land both raw JSON files in Delta, unchanged.

No reshaping happens here. The nested `labelled_alternatives` / `alternatives`
arrays stay nested; the only added columns are ingestion metadata, so a bronze
table can always be traced back to the file and run that produced it.
"""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from . import config as cfg
from .schemas import FEATURES_SCHEMA, LABELS_SCHEMA


def _read_json(
    spark: SparkSession,
    path: Path,
    schema,
    infer_schema: bool = False,
) -> DataFrame:
    """Read a pretty-printed JSON array into a DataFrame.

    Both files are a single multi-line JSON array, so `multiLine` is required --
    without it Spark expects one object per line and yields all-null rows.
    """
    reader = spark.read.option("multiLine", "true")
    if infer_schema:
        return reader.json(str(path))
    return reader.schema(schema).json(str(path))


def _with_ingest_metadata(df: DataFrame, source_file: Path) -> DataFrame:
    return df.withColumn("_source_file", F.lit(source_file.name)).withColumn(
        "_ingested_at", F.current_timestamp()
    )


def load_labels(
    spark: SparkSession, config: cfg.PipelineConfig, infer_schema: bool = False
) -> DataFrame:
    """Raw `labelled_alternatives.json` as-is, plus ingestion metadata."""
    path = config.resolve_label_file()
    return _with_ingest_metadata(_read_json(spark, path, LABELS_SCHEMA, infer_schema), path)


def load_features(
    spark: SparkSession, config: cfg.PipelineConfig, infer_schema: bool = False
) -> DataFrame:
    """Raw `frozen_dataset.json` as-is, plus ingestion metadata."""
    path = config.resolve_feature_file()
    return _with_ingest_metadata(_read_json(spark, path, FEATURES_SCHEMA, infer_schema), path)


def write_delta(df: DataFrame, config: cfg.PipelineConfig, table: str) -> str:
    """Overwrite a Delta table and return how to address it.

    `overwriteSchema` is set because re-running the pipeline after a schema
    change should replace the table rather than fail on a mismatch -- these are
    derived tables, rebuilt from the raw files on every run.

    On Databricks this becomes a Unity Catalog managed table; locally it is a
    folder of parquet plus a `_delta_log`. Same DataFrame either way.
    """
    ref = config.table_ref(table)
    writer = df.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
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
    """The `DeltaTable` handle, for history and time travel."""
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


def build_bronze(
    spark: SparkSession, config: cfg.PipelineConfig, infer_schema: bool = False
) -> dict[str, DataFrame]:
    """Write `bronze_labels` and `bronze_features`, returning both DataFrames."""
    labels = load_labels(spark, config, infer_schema)
    features = load_features(spark, config, infer_schema)

    write_delta(labels, config, cfg.BRONZE_LABELS)
    write_delta(features, config, cfg.BRONZE_FEATURES)

    return {
        cfg.BRONZE_LABELS: read_delta(spark, config, cfg.BRONZE_LABELS),
        cfg.BRONZE_FEATURES: read_delta(spark, config, cfg.BRONZE_FEATURES),
    }
