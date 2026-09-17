"""Silver layer: explode the nested bronze structures into flat, typed tables.

Four tables come out of two bronze tables:

    silver_labels                 one row per (scenario_id, label_ordinal)
    silver_features               one row per (scenario_id, alternative_ordinal)
    silver_scenario_stakeholder   one row per (scenario_id, stakeholder_ordinal)
    silver_scenario_situation     one row per (scenario_id, situation_ordinal)

Note the ordinals. `posexplode` rather than `explode`, because
`(scenario_id, id_prod)` is **not unique**: scenarios `2647` and `13122` each
label `prod_1` twice. A merge keyed on a non-unique column fails outright with
"Cannot perform Merge as multiple source rows matched", so incremental
processing forces every row to have a stable identity. That is an improvement
regardless -- the duplicate now has a position, so the gate can say which copy
came first instead of just that a conflict exists.

The two scenario-level attributes get their own tables because they are genuine
multi-value fields: `stakeholder_preference` reaches length 3 and `situations`
length 2 in the published data, so keeping them on the label rows would multiply
the fact grain.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from . import config as cfg
from .bronze import read_delta
from .schemas import COST_COLUMNS, FEATURE_COLUMNS
from .source_type import Classifier, source_type_column
from .upsert import upsert_scoped

#: Merge keys per silver table. Each is unique by construction.
SILVER_KEYS: dict[str, list[str]] = {
    cfg.SILVER_LABELS: ["scenario_id", "label_ordinal"],
    cfg.SILVER_FEATURES: ["scenario_id", "alternative_ordinal"],
    cfg.SILVER_SCENARIO_STAKEHOLDER: ["scenario_id", "stakeholder_ordinal"],
    cfg.SILVER_SCENARIO_SITUATION: ["scenario_id", "situation_ordinal"],
}

#: Provenance carried from bronze onto every silver row.
_LINEAGE = ("_source_file", "_batch_id")


def build_silver_labels(
    bronze_labels: DataFrame, classifier: Classifier = source_type_column
) -> DataFrame:
    """Explode `labelled_alternatives`, keeping each label's position.

    `source_type` is derived here, at the first point where the scenario id has a
    stable name, so every downstream table inherits one definition.
    """
    exploded = bronze_labels.select(
        F.col("id").alias("scenario_id"),
        *[F.col(c) for c in _LINEAGE],
        F.posexplode("labelled_alternatives").alias("label_ordinal", "label"),
    )
    return exploded.select(
        "scenario_id",
        "label_ordinal",
        F.col("label.id_prod").alias("id_prod"),
        F.col("label.pref").alias("pref"),
        F.col("label.conf").alias("conf"),
        F.col("label.reason").alias("reason"),
        *_LINEAGE,
    ).withColumn("source_type", classifier("scenario_id"))


def build_silver_features(bronze_features: DataFrame) -> DataFrame:
    """Explode `alternatives`, flattening the nested `c` struct into c_p/c_w/c_m."""
    exploded = bronze_features.select(
        F.col("id").alias("scenario_id"),
        *[F.col(c) for c in _LINEAGE],
        F.posexplode("alternatives").alias("alternative_ordinal", "alt"),
    )
    return exploded.select(
        F.col("scenario_id"),
        F.col("alternative_ordinal"),
        F.col("alt.id_prod").alias("id_prod"),
        *[F.col(f"alt.{name}").alias(name) for name in FEATURE_COLUMNS],
        *[F.col(f"alt.c.{name}").alias(name) for name in COST_COLUMNS],
        *[F.col(c) for c in _LINEAGE],
    )


def build_silver_scenario_stakeholder(bronze_features: DataFrame) -> DataFrame:
    """One row per (scenario_id, stakeholder_ordinal)."""
    return bronze_features.select(
        F.col("id").alias("scenario_id"),
        *[F.col(c) for c in _LINEAGE],
        F.posexplode("stakeholder_preference").alias(
            "stakeholder_ordinal", "stakeholder_preference"
        ),
    )


def build_silver_scenario_situation(bronze_features: DataFrame) -> DataFrame:
    """One row per (scenario_id, situation_ordinal)."""
    return bronze_features.select(
        F.col("id").alias("scenario_id"),
        *[F.col(c) for c in _LINEAGE],
        F.posexplode("situations").alias("situation_ordinal", "situation"),
    )


def scenarios_in(frames: list[DataFrame]) -> list[str]:
    """Distinct scenario ids present across the given frames."""
    if not frames:
        return []
    combined = frames[0].select("scenario_id")
    for frame in frames[1:]:
        combined = combined.unionByName(frame.select("scenario_id"))
    return sorted({row["scenario_id"] for row in combined.distinct().collect()})


def build_silver(
    spark: SparkSession,
    config: cfg.PipelineConfig,
    bronze: dict[str, DataFrame],
    classifier: Classifier = source_type_column,
    incremental: bool = True,
) -> dict[str, DataFrame]:
    """Upsert all four silver tables and return them.

    With `incremental=True` only the scenarios present in `bronze` are touched;
    the caller narrows bronze to the current batch. With `incremental=False`
    every table is rebuilt from scratch, which is what a backfill wants.
    """
    bronze_labels = bronze[cfg.BRONZE_LABELS]
    bronze_features = bronze[cfg.BRONZE_FEATURES]

    tables = {
        cfg.SILVER_LABELS: build_silver_labels(bronze_labels, classifier),
        cfg.SILVER_FEATURES: build_silver_features(bronze_features),
        cfg.SILVER_SCENARIO_STAKEHOLDER: build_silver_scenario_stakeholder(bronze_features),
        cfg.SILVER_SCENARIO_SITUATION: build_silver_scenario_situation(bronze_features),
    }

    for name, frame in tables.items():
        # Scope per table, from that table's *own* source -- never a scope shared
        # across all four. A batch that replaces only the labels file leaves the
        # features frame empty, and a shared scope would then delete every
        # feature row for the scenarios the labels mention. Empty scope means
        # "this table has nothing to do", which is exactly right.
        scope = scenarios_in([frame]) if incremental else None
        upsert_scoped(
            spark,
            config,
            name,
            frame,
            keys=SILVER_KEYS[name],
            scope_values=scope,
        )

    return {name: read_delta(spark, config, name) for name in tables}
