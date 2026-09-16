"""The provenance classifier, in plain Python and in Spark.

These are the cheapest tests in the suite and the ones that encode what the
exploration of the real files actually found, so they run without Spark where
possible.
"""

from __future__ import annotations

import pytest
from pyspark.sql import functions as F

from concrete_pipeline.source_type import (
    CONTROL_AXES,
    CONTROL_SYNTHETIC,
    EXPERT_ANNOTATED,
    LLM_GENERATED,
    UNKNOWN,
    classify_source_type,
    control_axis,
    naive_source_type_column,
    source_type_column,
)

# Real ids sampled from the published files, one per discovered pattern.
REAL_IDS = [
    ("control_health_1068", CONTROL_SYNTHETIC),
    ("control_gwp_491", CONTROL_SYNTHETIC),
    ("control_archfinish_705", CONTROL_SYNTHETIC),
    ("control_archfinish_slump_299", CONTROL_SYNTHETIC),
    ("control_fwu_384", CONTROL_SYNTHETIC),
    ("control_wdp_2666", CONTROL_SYNTHETIC),
    ("control_density_361", CONTROL_SYNTHETIC),
    ("control_cost_2985", CONTROL_SYNTHETIC),
    ("expert_118", EXPERT_ANNOTATED),
    ("expert_1", EXPERT_ANNOTATED),
    ("3782", LLM_GENERATED),
    ("1", LLM_GENERATED),
    ("18602", LLM_GENERATED),
]


@pytest.mark.parametrize("scenario_id,expected", REAL_IDS)
def test_classifies_every_real_pattern(scenario_id, expected):
    assert classify_source_type(scenario_id) == expected


@pytest.mark.parametrize("scenario_id", ["", "control_", "control_health_", "prod_1", "expert_"])
def test_unrecognised_ids_are_unknown_not_guessed(scenario_id):
    assert classify_source_type(scenario_id) == UNKNOWN


def test_none_is_unknown():
    assert classify_source_type(None) == UNKNOWN


def test_expert_and_bare_numeric_are_kept_apart():
    """`expert_118` and `118` are different scenarios in the real data.

    They collide only because both families number from 1. Conflating them
    would merge 272 expert scenarios into the LLM population.
    """
    assert classify_source_type("expert_118") != classify_source_type("118")


def test_archfinish_slump_is_its_own_axis():
    """`control_archfinish` and `control_archfinish_slump` are separate probes."""
    assert control_axis("control_archfinish_slump_299") == "archfinish_slump"
    assert control_axis("control_archfinish_705") == "archfinish"


def test_control_axis_is_none_for_non_control_ids():
    assert control_axis("expert_5") is None
    assert control_axis("3782") is None


def test_every_documented_axis_classifies_as_control():
    for axis in CONTROL_AXES:
        assert classify_source_type(f"control_{axis}_1") == CONTROL_SYNTHETIC


def _classify_in_spark(spark, ids, classifier):
    df = spark.createDataFrame([(i,) for i in ids], "scenario_id string")
    rows = df.withColumn("source_type", classifier("scenario_id")).collect()
    return {row["scenario_id"]: row["source_type"] for row in rows}


def test_spark_classifier_matches_python_reference(spark):
    ids = [scenario_id for scenario_id, _ in REAL_IDS]
    in_spark = _classify_in_spark(spark, ids, source_type_column)
    assert in_spark == {scenario_id: classify_source_type(scenario_id) for scenario_id in ids}


def test_spark_classifier_trims_whitespace(spark):
    assert _classify_in_spark(spark, ["  expert_9  "], source_type_column) == {
        "  expert_9  ": EXPERT_ANNOTATED
    }


def test_naive_classifier_carries_both_documented_bugs(spark):
    """The defective rule used as v1 of the Delta versioning demo.

    It must be wrong in exactly the two ways the demo advertises, otherwise the
    time-travel diff no longer illustrates what the README claims.
    """
    naive = _classify_in_spark(
        spark,
        ["expert_118", "control_archfinish_slump_299", "control_health_1068", "3782"],
        naive_source_type_column,
    )
    assert naive["expert_118"] == LLM_GENERATED  # bug 1: no expert_ rule
    assert naive["control_archfinish_slump_299"] == LLM_GENERATED  # bug 2: axis allow-list
    # ... and right about everything else, so the diff stays interpretable.
    assert naive["control_health_1068"] == CONTROL_SYNTHETIC
    assert naive["3782"] == LLM_GENERATED


def test_silver_labels_carry_source_type(result):
    from concrete_pipeline import config as cfg

    silver_labels = result.silver[cfg.SILVER_LABELS]
    assert "source_type" in silver_labels.columns
    observed = {row["source_type"] for row in silver_labels.select("source_type").collect()}
    assert observed == {CONTROL_SYNTHETIC, LLM_GENERATED, EXPERT_ANNOTATED}


def test_no_unknown_source_types_in_gold(gold):
    unknown = gold.filter(F.col("source_type") == UNKNOWN).count()
    assert unknown == 0
