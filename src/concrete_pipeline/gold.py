"""Gold layer: the modelling table, and the rows that did not earn a place in it.

`gold_scenarios` is one row per (scenario_id, id_prod) carrying the label, every
product feature, the scenario's provenance, and the scenario-level stakeholder
and situation rolled back up as arrays. `gold_scenarios_rejected` has the same
shape plus `rejection_reasons`.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from . import config as cfg
from . import quality
from .bronze import read_delta, write_delta
from .schemas import COST_COLUMNS, FEATURE_COLUMNS
from .source_type import UNKNOWN

#: Column order of `gold_scenarios`, keys first, label next, features last.
GOLD_COLUMN_ORDER: tuple[str, ...] = (
    "scenario_id",
    "id_prod",
    "source_type",
    "pref",
    "conf",
    "reason",
    "stakeholder_preferences",
    "situations",
    *FEATURE_COLUMNS,
    *COST_COLUMNS,
)


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
    features = silver_features.withColumn("_has_feature", F.lit(True))

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


def build_gold(
    spark: SparkSession,
    config: cfg.PipelineConfig,
    silver: dict[str, DataFrame],
) -> dict[str, DataFrame]:
    """Run the gate and write `gold_scenarios` plus `gold_scenarios_rejected`."""
    silver_labels = silver[cfg.SILVER_LABELS]
    silver_features = silver[cfg.SILVER_FEATURES]

    joined = join_labels_and_features(silver_labels, silver_features)
    joined = attach_scenario_attributes(
        joined,
        silver[cfg.SILVER_SCENARIO_STAKEHOLDER],
        silver[cfg.SILVER_SCENARIO_SITUATION],
    )

    product_sets = quality.scenario_product_sets(silver_labels, silver_features)
    gated = quality.with_rejection_reasons(joined, config.thresholds, product_sets)
    passed, rejected = quality.split_on_gate(gated)

    passed = passed.drop("_has_label", "_has_feature").select(*GOLD_COLUMN_ORDER)
    rejected = rejected.drop("_has_label", "_has_feature").select(
        *GOLD_COLUMN_ORDER, "rejection_reasons"
    )

    write_delta(passed, config, cfg.GOLD_SCENARIOS)
    write_delta(rejected, config, cfg.GOLD_SCENARIOS_REJECTED)

    return {
        cfg.GOLD_SCENARIOS: read_delta(spark, config, cfg.GOLD_SCENARIOS),
        cfg.GOLD_SCENARIOS_REJECTED: read_delta(spark, config, cfg.GOLD_SCENARIOS_REJECTED),
    }
