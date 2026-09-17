"""The silver -> gold quality gate.

Two kinds of test: end-to-end assertions against the fixture's planted defects,
and unit tests that build tiny frames for the conditions the fixture does not
contain (a scenario missing from one file, a product set mismatch).
"""

from __future__ import annotations

import pytest
from pyspark.sql import functions as F

from concrete_pipeline import config as cfg
from concrete_pipeline import quality
from concrete_pipeline.config import QualityThresholds
from concrete_pipeline.gold import join_labels_and_features


def reasons_for(df, scenario_id, id_prod):
    """All rejection_reasons arrays recorded for one (scenario, product)."""
    rows = df.filter(
        (F.col("scenario_id") == scenario_id) & (F.col("id_prod") == id_prod)
    ).collect()
    return [sorted(row["rejection_reasons"]) for row in rows]


# --------------------------------------------------------------------------
# End-to-end, against the fixture
# --------------------------------------------------------------------------


def test_gate_rejects_exactly_the_planted_defects(rejected):
    assert rejected.count() == 4


def test_clean_rows_are_promoted(gold):
    assert gold.count() == 13


def test_gate_loses_nothing(result, gold, rejected):
    """Promoted + rejected must account for every joined row."""
    joined = join_labels_and_features(
        result.silver[cfg.SILVER_LABELS], result.silver[cfg.SILVER_FEATURES]
    )
    assert gold.count() + rejected.count() == joined.count()


def test_both_conflicting_duplicates_are_rejected(rejected):
    """Scenario 2647 labels prod_1 twice with contradictory prefs.

    Both copies go to the rejected table: the source contains a genuine
    contradiction and the gate must not invent a winner.
    """
    found = reasons_for(rejected, "2647", "prod_1")
    assert len(found) == 2
    assert all(quality.REASON_DUPLICATE in row for row in found)

    prefs = sorted(
        row["pref"]
        for row in rejected.filter(
            (F.col("scenario_id") == "2647") & (F.col("id_prod") == "prod_1")
        ).collect()
    )
    assert prefs == [0.05, 0.62]


def test_duplicates_do_not_contaminate_the_rest_of_their_scenario(gold):
    """2647 has 5 products; prod_1 and prod_5 are rejected, 3 still promote."""
    promoted = {row["id_prod"] for row in gold.filter(F.col("scenario_id") == "2647").collect()}
    assert promoted == {"prod_2", "prod_3", "prod_4"}


def test_out_of_bounds_pref_is_rejected(rejected):
    found = reasons_for(rejected, "2647", "prod_5")
    assert found == [[quality.REASON_PREF_BOUNDS]]


def test_a_row_can_carry_several_reasons(rejected):
    """3782/prod_4 breaks both the density and the w/c range."""
    found = reasons_for(rejected, "3782", "prod_4")
    assert found == [["density_out_of_range", "water_to_cement_ratio_out_of_range"]]


def test_gate_summary_counts_each_reason(rejected):
    summary = {row["rejection_reason"]: row["count"] for row in quality.gate_summary(rejected).collect()}
    assert summary == {
        quality.REASON_DUPLICATE: 2,
        quality.REASON_PREF_BOUNDS: 1,
        "density_out_of_range": 1,
        "water_to_cement_ratio_out_of_range": 1,
    }


def test_promoted_rows_carry_no_reasons_column(gold):
    assert "rejection_reasons" not in gold.columns


def test_rejected_rows_keep_their_data(rejected):
    """A rejected row is still a full row -- it can be inspected and fixed."""
    row = rejected.filter(
        (F.col("scenario_id") == "3782") & (F.col("id_prod") == "prod_4")
    ).collect()[0]
    assert row["density"] == 950.0
    assert row["source_type"] == "llm_generated"
    assert row["reason"].startswith("Acceptable strength")


# --------------------------------------------------------------------------
# Unit tests for conditions the fixture does not contain
# --------------------------------------------------------------------------

LABEL_SCHEMA = "scenario_id string, id_prod string, pref double, conf double, source_type string"
FEATURE_SCHEMA = (
    "scenario_id string, id_prod string, compressive_strength double, "
    "water_to_cement_ratio double, density double"
)


def _gate(spark, labels_rows, feature_rows):
    labels = spark.createDataFrame(labels_rows, LABEL_SCHEMA)
    features = spark.createDataFrame(feature_rows, FEATURE_SCHEMA)
    joined = join_labels_and_features(labels, features)
    product_sets = quality.scenario_product_sets(labels, features)
    return quality.with_rejection_reasons(joined, QualityThresholds(), product_sets)


def test_matching_product_sets_pass(spark):
    gated = _gate(
        spark,
        [("s1", "prod_1", 0.5, 0.9, "llm_generated")],
        [("s1", "prod_1", 40.0, 0.45, 2400.0)],
    )
    assert reasons_for(gated, "s1", "prod_1") == [[]]


def test_label_without_a_feature_row_is_flagged(spark):
    gated = _gate(
        spark,
        [("s1", "prod_1", 0.5, 0.9, "llm_generated"), ("s1", "prod_2", 0.4, 0.9, "llm_generated")],
        [("s1", "prod_1", 40.0, 0.45, 2400.0)],
    )
    found = reasons_for(gated, "s1", "prod_2")[0]
    assert quality.REASON_MISSING_FEATURE in found
    assert quality.REASON_PRODUCT_SET_MISMATCH in found


def test_feature_without_a_label_row_is_flagged(spark):
    gated = _gate(
        spark,
        [("s1", "prod_1", 0.5, 0.9, "llm_generated")],
        [("s1", "prod_1", 40.0, 0.45, 2400.0), ("s1", "prod_2", 41.0, 0.44, 2390.0)],
    )
    found = reasons_for(gated, "s1", "prod_2")[0]
    assert quality.REASON_MISSING_LABEL in found


def test_product_set_mismatch_flags_the_whole_scenario(spark):
    """The check is scenario-level: clean siblings are flagged too, by design.

    If the two files disagree about which products a scenario contains, no row
    in that scenario can be trusted to be joined correctly.
    """
    gated = _gate(
        spark,
        [("s1", "prod_1", 0.5, 0.9, "llm_generated"), ("s1", "prod_9", 0.4, 0.9, "llm_generated")],
        [("s1", "prod_1", 40.0, 0.45, 2400.0)],
    )
    assert quality.REASON_PRODUCT_SET_MISMATCH in reasons_for(gated, "s1", "prod_1")[0]


def test_scenario_present_in_only_one_file_is_flagged(spark):
    gated = _gate(
        spark,
        [("s1", "prod_1", 0.5, 0.9, "llm_generated")],
        [("s2", "prod_1", 40.0, 0.45, 2400.0)],
    )
    assert quality.REASON_PRODUCT_SET_MISMATCH in reasons_for(gated, "s1", "prod_1")[0]
    assert quality.REASON_PRODUCT_SET_MISMATCH in reasons_for(gated, "s2", "prod_1")[0]


def test_conf_bounds_are_enforced(spark):
    gated = _gate(
        spark,
        [("s1", "prod_1", 0.5, 1.4, "llm_generated")],
        [("s1", "prod_1", 40.0, 0.45, 2400.0)],
    )
    assert reasons_for(gated, "s1", "prod_1") == [[quality.REASON_CONF_BOUNDS]]


def test_boundary_values_are_accepted(spark):
    """pref/conf of exactly 0 and 1 are valid, and the ranges are inclusive."""
    gated = _gate(
        spark,
        [("s1", "prod_1", 0.0, 1.0, "llm_generated"), ("s1", "prod_2", 1.0, 0.0, "llm_generated")],
        [("s1", "prod_1", 5.0, 0.20, 1200.0), ("s1", "prod_2", 150.0, 1.00, 3000.0)],
    )
    assert reasons_for(gated, "s1", "prod_1") == [[]]
    assert reasons_for(gated, "s1", "prod_2") == [[]]


def test_nulls_do_not_trigger_range_reasons(spark):
    """A missing measurement is absent, not implausible; the gate stays quiet."""
    gated = _gate(
        spark,
        [("s1", "prod_1", 0.5, 0.9, "llm_generated")],
        [("s1", "prod_1", None, None, None)],
    )
    assert reasons_for(gated, "s1", "prod_1") == [[]]


def test_real_data_ranges_are_not_rejected(spark):
    """The published extremes must survive the gate, or it is tuned too tight.

    These are the min/max actually observed across the 149,670 alternatives.
    """
    gated = _gate(
        spark,
        [("s1", "prod_1", 0.0, 0.4, "llm_generated"), ("s1", "prod_2", 1.0, 1.0, "llm_generated")],
        [("s1", "prod_1", 8.0, 0.30, 1400.0), ("s1", "prod_2", 60.0, 0.80, 3000.0)],
    )
    assert reasons_for(gated, "s1", "prod_1") == [[]]
    assert reasons_for(gated, "s1", "prod_2") == [[]]


# --------------------------------------------------------------------------
# The gate's failing thresholds
# --------------------------------------------------------------------------

GATE_SCHEMA = "scenario_id string, source_type string, rejection_reasons array<string>"


def _rejected(spark, rows):
    return spark.createDataFrame(rows, GATE_SCHEMA)


def test_a_few_rejections_do_not_fail_the_run(spark):
    """The published data rejects 4 of 149,672 rows. That must stay green."""
    rejected = _rejected(spark, [("s1", "llm_generated", ["duplicate_scenario_product"])])
    quality.enforce_gate_limits(9_999, rejected, cfg.GateLimits())


def test_too_many_rejections_fail_the_run(spark):
    """Recording a problem is not noticing one -- a spike has to go red."""
    rejected = _rejected(
        spark, [(f"s{i}", "llm_generated", ["pref_out_of_bounds"]) for i in range(20)]
    )
    with pytest.raises(quality.GateFailure, match="above the limit"):
        quality.enforce_gate_limits(100, rejected, cfg.GateLimits())


def test_absolute_row_limit_is_enforced(spark):
    rejected = _rejected(
        spark, [(f"s{i}", "llm_generated", ["pref_out_of_bounds"]) for i in range(5)]
    )
    limits = cfg.GateLimits(max_rejected_rows=2, max_rejected_fraction=1.0)
    with pytest.raises(quality.GateFailure, match="above the limit of 2"):
        quality.enforce_gate_limits(1_000_000, rejected, limits)


def test_an_unrecognised_scenario_id_fails_the_run(spark):
    """`unknown` means the id taxonomy changed upstream and needs re-profiling."""
    rejected = _rejected(spark, [("weird_new_shape", "unknown", ["missing_label_row"])])
    with pytest.raises(quality.GateFailure, match="no longer match any known pattern"):
        quality.enforce_gate_limits(1_000_000, rejected, cfg.GateLimits())


def test_an_empty_run_is_not_a_failure(spark):
    """Nothing ingested means nothing to judge, not a division by zero."""
    quality.enforce_gate_limits(0, _rejected(spark, []), cfg.GateLimits())
