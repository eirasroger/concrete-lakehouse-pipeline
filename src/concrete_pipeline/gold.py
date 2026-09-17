"""Gold layer: the modelling table, and the rows that did not earn a place in it.

`gold_scenarios` is one row per label carrying its product's features, the
scenario's provenance, and the scenario-level stakeholder and situation rolled
back up as arrays. `gold_scenarios_rejected` has the same shape plus
`rejection_reasons`.

Only the scenarios that changed in silver are recomputed. The rest of the table
is left untouched, which is the whole point of the incremental path: a run that
ingests one new batch does not re-derive 149,668 rows to arrive at the same
answer.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from . import config as cfg
from . import quality
from .bronze import _sql_ref, read_delta, table_exists
from .schemas import COST_COLUMNS, FEATURE_COLUMNS
from .source_type import SOURCE_TYPES, UNKNOWN
from .upsert import upsert_scoped

#: Column order of the gold tables. Keys first, lineage next, then features.
GOLD_COLUMN_ORDER: tuple[str, ...] = (
    "scenario_id",
    "id_prod",
    "label_ordinal",
    "alternative_ordinal",
    "source_type",
    "pref",
    "conf",
    "reason",
    "stakeholder_preferences",
    "situations",
    *FEATURE_COLUMNS,
    *COST_COLUMNS,
    "_source_file",
    "_batch_id",
)

#: Null-safe composite key. Rejected rows may be missing either side of the
#: join, so both ordinals participate and comparison uses `<=>`.
GOLD_KEYS = ["scenario_id", "id_prod", "label_ordinal", "alternative_ordinal"]

#: Table-level CHECK constraints on gold_scenarios. The gate already guarantees
#: these; the constraints make the guarantee the table's own, so a future writer
#: that bypasses the pipeline cannot quietly violate it.
GOLD_CONSTRAINTS: dict[str, str] = {
    "pref_in_unit_range": "pref IS NOT NULL AND pref BETWEEN 0 AND 1",
    "conf_in_unit_range": "conf IS NOT NULL AND conf BETWEEN 0 AND 1",
    "source_type_known": "source_type IN (" + ", ".join(f"'{s}'" for s in SOURCE_TYPES) + ")",
    "has_both_sides": "label_ordinal IS NOT NULL AND alternative_ordinal IS NOT NULL",
}


def join_labels_and_features(
    silver_labels: DataFrame, silver_features: DataFrame
) -> DataFrame:
    """Full-outer join on (scenario_id, id_prod), keeping presence markers.

    A full outer join rather than an inner one: an inner join would make a label
    with no matching feature row (or the reverse) disappear, which is exactly the
    defect the gate is meant to surface. `_has_label` / `_has_feature` survive
    the join so the gate can tell which side was missing.
    """
    labels = silver_labels.withColumn("_has_label", F.lit(True))
    features = silver_features.drop("_source_file", "_batch_id").withColumn(
        "_has_feature", F.lit(True)
    )

    joined = labels.join(features, on=["scenario_id", "id_prod"], how="fullouter")
    joined = joined.fillna({"_has_label": False, "_has_feature": False})

    # A feature row with no label brings no source_type across the join. Such a
    # row is rejected anyway, but the column should never be null in either gold
    # table -- `unknown` is the classifier's own word for "not recognised".
    return joined.withColumn(
        "source_type", F.coalesce(F.col("source_type"), F.lit(UNKNOWN))
    )


def attach_scenario_attributes(
    df: DataFrame,
    silver_stakeholder: DataFrame,
    silver_situation: DataFrame,
) -> DataFrame:
    """Roll the scenario-level tables back up as arrays on each product row.

    The normalised silver tables stay the source of truth; these arrays are a
    convenience so `gold_scenarios` is self-contained for modelling.
    """
    stakeholders = silver_stakeholder.groupBy("scenario_id").agg(
        F.array_sort(F.collect_set("stakeholder_preference")).alias("stakeholder_preferences")
    )
    situations = silver_situation.groupBy("scenario_id").agg(
        F.array_sort(F.collect_set("situation")).alias("situations")
    )
    return df.join(stakeholders, on="scenario_id", how="left").join(
        situations, on="scenario_id", how="left"
    )


def _scoped(df: DataFrame, scenarios: list[str] | None) -> DataFrame:
    """Narrow a frame to the given scenarios, or leave it whole when None."""
    if scenarios is None:
        return df
    return df.filter(F.col("scenario_id").isin(scenarios))


def apply_constraints(spark: SparkSession, config: cfg.PipelineConfig) -> list[str]:
    """Add the CHECK constraints to gold_scenarios, skipping any already set.

    Delta stores these as `delta.constraints.<name>` table properties, so adding
    one twice is an error rather than a no-op -- hence the existence check.
    """
    table = cfg.GOLD_SCENARIOS
    if not table_exists(spark, config, table):
        return []

    existing = {
        row["key"].removeprefix("delta.constraints.")
        for row in spark.sql(f"SHOW TBLPROPERTIES {_sql_ref(config, table)}").collect()
        if row["key"].startswith("delta.constraints.")
    }

    added = []
    for name, expression in GOLD_CONSTRAINTS.items():
        if name in existing:
            continue
        spark.sql(
            f"ALTER TABLE {_sql_ref(config, table)} "
            f"ADD CONSTRAINT {name} CHECK ({expression})"
        )
        added.append(name)
    return added


def build_gold(
    spark: SparkSession,
    config: cfg.PipelineConfig,
    silver: dict[str, DataFrame],
    scenarios: list[str] | None = None,
) -> dict[str, DataFrame]:
    """Run the gate and upsert the two gold tables.

    `scenarios` limits the work to the scenarios that changed in silver. None
    means recompute everything.
    """
    silver_labels = _scoped(silver[cfg.SILVER_LABELS], scenarios)
    silver_features = _scoped(silver[cfg.SILVER_FEATURES], scenarios)

    joined = join_labels_and_features(silver_labels, silver_features)
    joined = attach_scenario_attributes(
        joined,
        _scoped(silver[cfg.SILVER_SCENARIO_STAKEHOLDER], scenarios),
        _scoped(silver[cfg.SILVER_SCENARIO_SITUATION], scenarios),
    )

    product_sets = quality.scenario_product_sets(silver_labels, silver_features)
    gated = quality.with_rejection_reasons(joined, config.thresholds, product_sets)
    passed, rejected = quality.split_on_gate(gated)

    passed = passed.drop("_has_label", "_has_feature").select(*GOLD_COLUMN_ORDER)
    rejected = rejected.drop("_has_label", "_has_feature").select(
        *GOLD_COLUMN_ORDER, "rejection_reasons"
    )

    for table, frame in (
        (cfg.GOLD_SCENARIOS, passed),
        (cfg.GOLD_SCENARIOS_REJECTED, rejected),
    ):
        upsert_scoped(
            spark, config, table, frame, keys=GOLD_KEYS, scope_values=scenarios
        )

    apply_constraints(spark, config)

    return {
        cfg.GOLD_SCENARIOS: read_delta(spark, config, cfg.GOLD_SCENARIOS),
        cfg.GOLD_SCENARIOS_REJECTED: read_delta(spark, config, cfg.GOLD_SCENARIOS_REJECTED),
    }
