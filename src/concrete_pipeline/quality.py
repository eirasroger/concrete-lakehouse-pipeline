"""The silver -> gold quality gate.

Every check appends a machine-readable reason to a `rejection_reasons` array.
A row with an empty array is promoted to `gold_scenarios`; a row with any reason
is written to `gold_scenarios_rejected` *with its reasons attached*. Nothing is
dropped silently, and a rejected row can always be explained by reading one
column.

The checks, and what they caught on the published data (42,874 scenarios,
149,672 label rows):

    pref_out_of_bounds              pref outside [0, 1]              0 rows
    conf_out_of_bounds              conf outside [0, 1]              0 rows
    duplicate_scenario_product      repeated (scenario_id, id_prod)  4 rows
    scenario_product_set_mismatch   id_prod sets differ per scenario 0 rows
    missing_feature_row             label with no matching feature   0 rows
    missing_label_row               feature with no matching label   0 rows
    <column>_out_of_range           implausible physical value       0 rows

The four duplicate rows are scenarios `2647` and `13122`, which each carry two
contradictory labels for `prod_1` (pref 0.62 at conf 0.75 against pref 0.05 at
conf 0.95, and similar). Both copies are rejected: the conflict is real and
picking a winner by confidence would invent a resolution the source does not
support. The remaining alternatives in those scenarios still promote to gold.
"""

from __future__ import annotations

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from .config import QualityThresholds

REASON_PREF_BOUNDS = "pref_out_of_bounds"
REASON_CONF_BOUNDS = "conf_out_of_bounds"
REASON_DUPLICATE = "duplicate_scenario_product"
REASON_PRODUCT_SET_MISMATCH = "scenario_product_set_mismatch"
REASON_MISSING_FEATURE = "missing_feature_row"
REASON_MISSING_LABEL = "missing_label_row"

#: Marks rows that passed every check.
NO_REASONS: tuple[str, ...] = ()


def scenario_product_sets(silver_labels: DataFrame, silver_features: DataFrame) -> DataFrame:
    """Per-scenario comparison of the id_prod sets held by each source file.

    Returns one row per scenario with `label_products`, `feature_products` and a
    boolean `products_match`. A scenario present in only one file has a null set
    on the other side and does not match.
    """
    label_sets = silver_labels.groupBy("scenario_id").agg(
        F.array_sort(F.collect_set("id_prod")).alias("label_products")
    )
    feature_sets = silver_features.groupBy("scenario_id").agg(
        F.array_sort(F.collect_set("id_prod")).alias("feature_products")
    )
    return label_sets.join(feature_sets, on="scenario_id", how="fullouter").withColumn(
        "products_match",
        F.col("label_products").isNotNull()
        & F.col("feature_products").isNotNull()
        & (F.col("label_products") == F.col("feature_products")),
    )


def _bounds_reason(column: str, low: float, high: float, reason: str) -> Column:
    """Flag `reason` when a non-null value falls outside [low, high]."""
    value = F.col(column)
    violated = value.isNotNull() & (~value.between(low, high))
    return F.when(violated, F.lit(reason))


def _range_reasons(thresholds: QualityThresholds) -> list[Column]:
    return [
        _bounds_reason(rule.column, rule.minimum, rule.maximum, rule.rejection_reason)
        for rule in thresholds.ranges
    ]


def with_rejection_reasons(
    joined: DataFrame,
    thresholds: QualityThresholds,
    product_sets: DataFrame,
) -> DataFrame:
    """Attach a `rejection_reasons` array to every joined row.

    `joined` is the full-outer join of silver_labels and silver_features on
    (scenario_id, id_prod), carrying `_has_label` / `_has_feature` markers.
    """
    duplicate_window = Window.partitionBy("scenario_id", "id_prod")

    mismatched = product_sets.filter(~F.col("products_match")).select(
        F.col("scenario_id"), F.lit(True).alias("_product_set_mismatch")
    )

    annotated = joined.join(mismatched, on="scenario_id", how="left").withColumn(
        "_row_count", F.count(F.lit(1)).over(duplicate_window)
    )

    candidate_reasons = [
        _bounds_reason("pref", thresholds.pref_min, thresholds.pref_max, REASON_PREF_BOUNDS),
        _bounds_reason("conf", thresholds.conf_min, thresholds.conf_max, REASON_CONF_BOUNDS),
        F.when(F.col("_row_count") > 1, F.lit(REASON_DUPLICATE)),
        F.when(
            F.col("_product_set_mismatch").isNotNull(), F.lit(REASON_PRODUCT_SET_MISMATCH)
        ),
        F.when(~F.col("_has_feature"), F.lit(REASON_MISSING_FEATURE)),
        F.when(~F.col("_has_label"), F.lit(REASON_MISSING_LABEL)),
        *_range_reasons(thresholds),
    ]

    reasons = F.array_sort(
        F.array_compact(F.array(*candidate_reasons))
    )

    return annotated.withColumn("rejection_reasons", reasons).drop(
        "_row_count", "_product_set_mismatch"
    )


def split_on_gate(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Split a reason-annotated frame into (passed, rejected)."""
    passed = df.filter(F.size("rejection_reasons") == 0).drop("rejection_reasons")
    rejected = df.filter(F.size("rejection_reasons") > 0)
    return passed, rejected


def gate_summary(rejected: DataFrame) -> DataFrame:
    """Count rejected rows per individual reason (a row may carry several)."""
    return (
        rejected.select(F.explode("rejection_reasons").alias("rejection_reason"))
        .groupBy("rejection_reason")
        .count()
        .orderBy(F.col("count").desc(), "rejection_reason")
    )
