"""Table maintenance: compaction and file expiry.

Incremental writes are what make this necessary. Every `MERGE` rewrites only the
files it touches, so a table that receives many small batches accumulates many
small files, and read performance degrades even though the data is correct.
`OPTIMIZE` compacts them; `VACUUM` removes the versions those rewrites orphaned.

This is deliberately a *separate* task from the pipeline, and deliberately not
run on every ingest. Compaction is expensive and pointless after a single small
batch, and `VACUUM` destroys time travel older than its retention window --
which is the one thing the versioning demo depends on.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyspark.sql import SparkSession

from . import config as cfg
from .bronze import _sql_ref, table_exists

#: Tables worth compacting, in dependency order.
MAINTAINED_TABLES: tuple[str, ...] = (
    cfg.BRONZE_LABELS,
    cfg.BRONZE_FEATURES,
    cfg.SILVER_LABELS,
    cfg.SILVER_FEATURES,
    cfg.SILVER_SCENARIO_STAKEHOLDER,
    cfg.SILVER_SCENARIO_SITUATION,
    cfg.GOLD_SCENARIOS,
    cfg.GOLD_SCENARIOS_REJECTED,
)

#: Keep a week of history. Long enough for time travel to be useful and for a
#: bad batch to be diffed against its predecessor; short enough that orphaned
#: files from daily merges do not accumulate indefinitely.
DEFAULT_RETAIN_HOURS = 168


@dataclass
class MaintenanceResult:
    """What maintenance did to one table."""

    table: str
    optimized: bool
    vacuumed: bool
    note: str = ""


def optimize_table(spark: SparkSession, config: cfg.PipelineConfig, table: str) -> None:
    """Compact small files. Z-ordering by scenario_id helps the scoped merges."""
    ref = _sql_ref(config, table)
    try:
        spark.sql(f"OPTIMIZE {ref} ZORDER BY (scenario_id)")
    except Exception:
        # Bronze has no scenario_id (it is still `id`), and Z-ordering is not
        # supported on every Delta build. Plain compaction is the useful part.
        spark.sql(f"OPTIMIZE {ref}")


def vacuum_table(
    spark: SparkSession,
    config: cfg.PipelineConfig,
    table: str,
    retain_hours: int = DEFAULT_RETAIN_HOURS,
) -> None:
    """Delete files no longer referenced and older than the retention window."""
    spark.sql(f"VACUUM {_sql_ref(config, table)} RETAIN {retain_hours} HOURS")


def run_maintenance(
    spark: SparkSession,
    config: cfg.PipelineConfig,
    tables: tuple[str, ...] = MAINTAINED_TABLES,
    retain_hours: int = DEFAULT_RETAIN_HOURS,
    vacuum: bool = True,
) -> list[MaintenanceResult]:
    """Compact and optionally vacuum every table that exists."""
    results: list[MaintenanceResult] = []
    for table in tables:
        if not table_exists(spark, config, table):
            results.append(MaintenanceResult(table, False, False, "absent"))
            continue

        optimize_table(spark, config, table)
        did_vacuum = False
        note = ""
        if vacuum:
            try:
                vacuum_table(spark, config, table, retain_hours)
                did_vacuum = True
            except Exception as exc:  # pragma: no cover - environment dependent
                note = f"vacuum skipped: {type(exc).__name__}"
        results.append(MaintenanceResult(table, True, did_vacuum, note))
    return results


def print_maintenance_report(results: list[MaintenanceResult]) -> None:
    print(f"\n{'table':<32} {'optimized':>10} {'vacuumed':>9}  note")
    print("-" * 70)
    for r in results:
        print(f"{r.table:<32} {str(r.optimized):>10} {str(r.vacuumed):>9}  {r.note}")
