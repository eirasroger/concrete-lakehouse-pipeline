"""Scenario-scoped upserts into Delta tables.

Every derived table here is keyed by scenario, and a scenario is the atomic unit
of change: when one arrives or is re-released, *all* of its rows are rebuilt. So
an upsert has to do three things, not two:

    matched         -> update      a row that still exists, with new values
    not matched     -> insert      a row the scenario gained
    not matched by
      source, in
      scope         -> delete      a row the scenario *lost*

That third clause is the one people forget. Without it, a re-release where a
scenario drops from 6 labels to 3 leaves 3 stale rows behind forever, and no row
count anywhere reveals it -- the table simply keeps answering with data the
source no longer contains.

The delete has to be scoped. `WHEN NOT MATCHED BY SOURCE THEN DELETE` with no
condition would delete every row of every scenario not in this batch, which on
an incremental run is the entire rest of the table.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession

from . import config as cfg
from .bronze import table_exists, write_delta


def _sql_string_list(values: list[str]) -> str:
    """Render values as a SQL `IN` list, escaping embedded quotes."""
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)


def upsert_scoped(
    spark: SparkSession,
    config: cfg.PipelineConfig,
    table: str,
    source: DataFrame,
    keys: list[str],
    scope_column: str = "scenario_id",
    scope_values: list[str] | None = None,
) -> str:
    """Merge `source` into `table`, rebuilding only the scoped scenarios.

    Falls back to a full overwrite when the table does not exist yet, or when the
    change set is large enough that a targeted merge stops being the cheaper
    option (see `IngestLimits.max_scenarios_for_merge`). Both are normal: the
    first is a first load, the second is a full re-release.
    """
    if not table_exists(spark, config, table):
        return write_delta(source, config, table)

    if scope_values is None:
        return write_delta(source, config, table)

    if len(scope_values) > config.ingest_limits.max_scenarios_for_merge:
        # A merge would have to name every scenario in its delete predicate.
        # Past this size, rewriting the table is both faster and simpler.
        return write_delta(source, config, table)

    if not scope_values:
        return config.table_ref(table)

    from .bronze import delta_table

    target = delta_table(spark, config, table)
    condition = " AND ".join(f"t.{k} <=> s.{k}" for k in keys)
    in_scope = f"t.{scope_column} IN ({_sql_string_list(scope_values)})"

    builder = (
        target.alias("t")
        .merge(source.alias("s"), condition)
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
    )
    # Delta 3.0+. Without it a shrinking scenario would silently keep stale rows,
    # so refuse to run rather than write a table we cannot trust.
    if not hasattr(builder, "whenNotMatchedBySourceDelete"):
        raise RuntimeError(
            "This Delta version has no WHEN NOT MATCHED BY SOURCE support, which "
            "incremental upserts need to remove rows a scenario has lost. "
            "Upgrade to delta-spark>=3.0, or run with a full rebuild."
        )
    builder.whenNotMatchedBySourceDelete(condition=in_scope).execute()

    return config.table_ref(table)
