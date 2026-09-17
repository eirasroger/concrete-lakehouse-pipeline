"""Incremental ingestion: decide what actually needs reading, then read it.

Two arrival patterns have to work, and they need different things:

**A new file lands next to the old one** (`labelled_alternatives_batch2.json`).
Auto Loader's home turf -- it remembers which paths it has consumed.

**The existing file is replaced by a superset** (a dataset re-release at the same
DOI). Auto Loader *cannot* see this. It identifies work by file path and
deliberately ignores modifications, so a rewritten `frozen_dataset.json` is
silently skipped. This is not a configuration mistake; it is what the mechanism
is for.

So the load-bearing mechanism here is a content-hash ingestion log
(`bronze_ingest_log`): one row per (file, content hash) ever consumed. A file is
pending when its *hash* is unseen, which covers both patterns with one rule and
makes re-running a no-op rather than a double load. It also works in
open-source Spark, so the incremental path is testable locally rather than only
observable in production.

Auto Loader remains available via `PipelineConfig.use_autoloader` for the
append-only case on Databricks, where its file-notification discovery genuinely
scales past what listing a directory can do. The hash log still guards
replacement underneath it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from . import config as cfg

#: Read in 8 MB blocks -- the features file is ~103 MB and hashing it is I/O
#: bound, not CPU bound.
_HASH_BLOCK = 8 * 1024 * 1024

INGEST_LOG_SCHEMA = StructType(
    [
        StructField("source_file", StringType(), False),
        StructField("file_hash", StringType(), False),
        StructField("source_kind", StringType(), False),
        StructField("byte_size", LongType(), True),
        StructField("row_count", LongType(), True),
        StructField("batch_id", StringType(), False),
        StructField("ingested_at", TimestampType(), False),
    ]
)


#: Mirrors the column Auto Loader populates, so downstream code does not care
#: which reader produced the frame.
RESCUED_COLUMN = "_rescued_data"


def with_rescued_column(schema: StructType) -> StructType:
    """A copy of `schema` with the rescued-data column appended.

    Builds a new StructType rather than calling `schema.add(...)`, which mutates
    the receiver and returns it. The schemas here are module-level constants, so
    mutating one would corrupt it for every later read -- the second ingestion in
    a process would fail with COLUMN_ALREADY_EXISTS.
    """
    if any(f.name == RESCUED_COLUMN for f in schema.fields):
        return schema
    return StructType([*schema.fields, StructField(RESCUED_COLUMN, StringType(), True)])


@dataclass(frozen=True)
class SourceFile:
    """One raw file, identified by content rather than by name."""

    path: Path
    kind: str  # "labels" | "features"
    file_hash: str
    byte_size: int

    @property
    def name(self) -> str:
        return self.path.name


def file_hash(path: Path) -> str:
    """SHA-256 of a file's bytes.

    Content, not mtime or size: a re-release that happens to preserve the byte
    count would defeat a cheaper check, and mtime changes when a file is merely
    copied. Hashing 140 MB costs about a second, against minutes of reprocessing.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_BLOCK):
            digest.update(chunk)
    return digest.hexdigest()


def describe_sources(config: cfg.PipelineConfig) -> list[SourceFile]:
    """Hash every raw file present, labels and features alike."""
    described: list[SourceFile] = []
    for kind, paths in (("labels", config.label_files()), ("features", config.feature_files())):
        for path in paths:
            described.append(
                SourceFile(
                    path=path,
                    kind=kind,
                    file_hash=file_hash(path),
                    byte_size=path.stat().st_size,
                )
            )
    return described


def ingest_log(spark: SparkSession, config: cfg.PipelineConfig) -> DataFrame:
    """The ingestion log, or an empty frame shaped like it on a first run."""
    from .bronze import read_delta, table_exists

    if not table_exists(spark, config, cfg.BRONZE_INGEST_LOG):
        return spark.createDataFrame([], INGEST_LOG_SCHEMA)
    return read_delta(spark, config, cfg.BRONZE_INGEST_LOG)


def pending_sources(
    spark: SparkSession, config: cfg.PipelineConfig, sources: list[SourceFile] | None = None
) -> list[SourceFile]:
    """The files whose content has not been ingested before.

    Re-running with nothing new returns an empty list, which is what makes the
    scheduled job idempotent: a retry after a partial failure re-reads only what
    never landed.
    """
    sources = sources if sources is not None else describe_sources(config)
    seen = {
        (row["source_file"], row["file_hash"])
        for row in ingest_log(spark, config).select("source_file", "file_hash").collect()
    }
    return [s for s in sources if (s.name, s.file_hash) not in seen]


def record_ingestion(
    spark: SparkSession,
    config: cfg.PipelineConfig,
    sources: list[SourceFile],
    row_counts: dict[str, int],
    batch_id: str,
) -> None:
    """Append one log row per file consumed in this batch."""
    from .bronze import append_delta

    if not sources:
        return
    now = datetime.now(timezone.utc)
    rows = [
        (
            s.name,
            s.file_hash,
            s.kind,
            int(s.byte_size),
            int(row_counts.get(s.name, 0)),
            batch_id,
            now,
        )
        for s in sources
    ]
    append_delta(spark.createDataFrame(rows, INGEST_LOG_SCHEMA), config, cfg.BRONZE_INGEST_LOG)


def new_batch_id() -> str:
    """A sortable identifier for one ingestion batch."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")


def read_sources(
    spark: SparkSession,
    config: cfg.PipelineConfig,
    sources: list[SourceFile],
    schema,
    batch_id: str,
) -> DataFrame | None:
    """Read the given files into one frame, tagged with their provenance.

    Reads each file separately rather than passing a glob, because each row has
    to carry the hash of the file it came from -- that is what lets bronze hold
    several versions of the same file name and still know which is current.

    `PERMISSIVE` mode plus `_corrupt_record` means a malformed file lands as
    inspectable rows instead of aborting the run. `_rescued_data` mirrors the
    column Auto Loader populates, so downstream code is identical either way.
    """
    if not sources:
        return None

    read_schema = with_rescued_column(schema)
    frames = []
    for source in sources:
        frame = (
            spark.read.option("multiLine", "true")
            .option("mode", "PERMISSIVE")
            .option("columnNameOfCorruptRecord", RESCUED_COLUMN)
            .schema(read_schema)
            .json(str(source.path))
            .withColumn("_source_file", F.lit(source.name))
            .withColumn("_file_hash", F.lit(source.file_hash))
            .withColumn("_batch_id", F.lit(batch_id))
            .withColumn("_ingested_at", F.current_timestamp())
        )
        frames.append(frame)

    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.unionByName(frame)
    return combined


def read_sources_autoloader(
    spark: SparkSession,
    config: cfg.PipelineConfig,
    schema,
    globs: tuple[str, ...],
    batch_id: str,
) -> DataFrame:
    """Auto Loader equivalent of `read_sources`, for new files on Databricks.

    `availableNow` makes this a batch that drains whatever has arrived and stops,
    which is what a scheduled job wants -- not an always-on stream. Returns a
    streaming DataFrame; the caller writes it with a checkpoint.

    Only sees files it has not consumed before, so it must be paired with the
    hash log to catch replaced files.
    """
    if config.checkpoint_dir is None:
        raise ValueError("use_autoloader requires PipelineConfig.checkpoint_dir")

    read_schema = with_rescued_column(schema)
    schema_location = str(config.checkpoint_dir / "schema")
    reader = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.schemaLocation", schema_location)
        .option("cloudFiles.schemaEvolutionMode", "rescue")
        .option("multiLine", "true")
        .option("rescuedDataColumn", RESCUED_COLUMN)
        .schema(read_schema)
    )
    pattern = globs[0]
    return (
        reader.load(f"{config.raw_dir}/{pattern}")
        .withColumn("_source_file", F.col("_metadata.file_name"))
        .withColumn("_file_hash", F.lit(None).cast("string"))
        .withColumn("_batch_id", F.lit(batch_id))
        .withColumn("_ingested_at", F.current_timestamp())
    )
