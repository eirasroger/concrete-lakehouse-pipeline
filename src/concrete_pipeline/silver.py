"""Silver layer: explode the nested bronze structures into flat, typed tables.

Four tables come out of two bronze tables:

    silver_labels                 one row per (scenario_id, id_prod) label
    silver_features               one row per (scenario_id, id_prod) alternative
    silver_scenario_stakeholder   one row per (scenario_id, stakeholder_preference)
    silver_scenario_situation     one row per (scenario_id, situation)

The two scenario-level attributes get their own tables because they are genuine
multi-value fields: `stakeholder_preference` reaches length 3 and `situations`
length 2 in the published data, so keeping them on the label rows would
multiply the fact grain.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from . import config as cfg
from .bronze import read_delta, write_delta
from .schemas import COST_COLUMNS, FEATURE_COLUMNS
from .source_type import Classifier, source_type_column


def build_silver_labels(
    bronze_labels: DataFrame, classifier: Classifier = source_type_column
) -> DataFrame:
    """Explode `labelled_alternatives` to one row per (scenario_id, id_prod).

    `source_type` is derived here, at the first point where the scenario id has
    a stable name, so every downstream table inherits one definition.
    """
    exploded = bronze_labels.select(
        F.col("id").alias("scenario_id"),
        F.explode("labelled_alternatives").alias("label"),
    )
    return exploded.select(
        "scenario_id",
        F.col("label.id_prod").alias("id_prod"),
        F.col("label.pref").alias("pref"),
        F.col("label.conf").alias("conf"),
        F.col("label.reason").alias("reason"),
    ).withColumn("source_type", classifier("scenario_id"))


def build_silver_features(bronze_features: DataFrame) -> DataFrame:
    """Explode `alternatives`, flattening the nested `c` struct into c_p/c_w/c_m."""
    exploded = bronze_features.select(
        F.col("id").alias("scenario_id"),
        F.explode("alternatives").alias("alt"),
    )
    columns = [
        F.col("scenario_id"),
        F.col("alt.id_prod").alias("id_prod"),
        *[F.col(f"alt.{name}").alias(name) for name in FEATURE_COLUMNS],
        *[F.col(f"alt.c.{name}").alias(name) for name in COST_COLUMNS],
    ]
    return exploded.select(*columns)


def build_silver_scenario_stakeholder(bronze_features: DataFrame) -> DataFrame:
    """One row per (scenario_id, stakeholder_preference)."""
    return bronze_features.select(
        F.col("id").alias("scenario_id"),
        F.explode("stakeholder_preference").alias("stakeholder_preference"),
    )


def build_silver_scenario_situation(bronze_features: DataFrame) -> DataFrame:
    """One row per (scenario_id, situation)."""
    return bronze_features.select(
        F.col("id").alias("scenario_id"),
        F.explode("situations").alias("situation"),
    )


def build_silver(
    spark: SparkSession,
    config: cfg.PipelineConfig,
    bronze: dict[str, DataFrame],
    classifier: Classifier = source_type_column,
) -> dict[str, DataFrame]:
    """Write all four silver tables and return them."""
    bronze_labels = bronze[cfg.BRONZE_LABELS]
    bronze_features = bronze[cfg.BRONZE_FEATURES]

    tables = {
        cfg.SILVER_LABELS: build_silver_labels(bronze_labels, classifier),
        cfg.SILVER_FEATURES: build_silver_features(bronze_features),
        cfg.SILVER_SCENARIO_STAKEHOLDER: build_silver_scenario_stakeholder(bronze_features),
        cfg.SILVER_SCENARIO_SITUATION: build_silver_scenario_situation(bronze_features),
    }

    for name, df in tables.items():
        write_delta(df, config, name)

    return {name: read_delta(spark, config, name) for name in tables}
