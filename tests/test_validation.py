"""The correlation sanity check.

The fixture is far too small for its own correlations to mean anything, so the
direction logic is tested against synthetic frames with a known answer, and the
fixture is used only to confirm the check runs end to end over real gold output.
"""

from __future__ import annotations

from concrete_pipeline.validation import (
    EXPECTED_DIRECTIONS,
    MIN_ROWS_FOR_DIRECTION,
    CorrelationCheck,
    correlations,
    format_summary,
    wrong_sign_checks,
)

VERDICTS = {"OK", "FLAT", "WRONG SIGN", "NO DATA", "TOO FEW"}

GOLD_SCHEMA = "source_type string, pref double, gwp double, health double, circ_orig double"


def _frame(spark, rows):
    return spark.createDataFrame(rows, GOLD_SCHEMA)


def test_detects_correctly_signed_relationships(spark):
    """pref rises as gwp falls and as health and circularity rise."""
    rows = [
        ("llm_generated", 0.1, 0.45, 1.0, 10.0),
        ("llm_generated", 0.4, 0.30, 3.0, 35.0),
        ("llm_generated", 0.7, 0.18, 4.0, 60.0),
        ("llm_generated", 0.95, 0.07, 6.0, 90.0),
    ]
    checks = correlations(_frame(spark, rows), min_rows=2)
    assert wrong_sign_checks(checks) == []
    assert {check.verdict for check in checks} == {"OK"}


def test_flags_an_inverted_relationship(spark):
    """If the pipeline mis-joined labels to features, signs flip -- catch that."""
    rows = [
        ("llm_generated", 0.95, 0.45, 1.0, 10.0),
        ("llm_generated", 0.7, 0.30, 3.0, 35.0),
        ("llm_generated", 0.4, 0.18, 4.0, 60.0),
        ("llm_generated", 0.1, 0.07, 6.0, 90.0),
    ]
    wrong = wrong_sign_checks(correlations(_frame(spark, rows), min_rows=2))
    assert {check.variable for check in wrong} == set(EXPECTED_DIRECTIONS)


def test_breaks_results_down_by_source_type_and_overall(spark):
    rows = [
        ("llm_generated", 0.2, 0.40, 2.0, 20.0),
        ("llm_generated", 0.8, 0.10, 5.0, 80.0),
        ("expert_annotated", 0.3, 0.35, 2.0, 25.0),
        ("expert_annotated", 0.9, 0.08, 6.0, 85.0),
    ]
    checks = correlations(_frame(spark, rows), min_rows=2)
    assert {check.source_type for check in checks} == {"llm_generated", "expert_annotated", "ALL"}
    assert all(check.rows > 0 for check in checks)


def test_flat_relationship_is_reported_not_failed(spark):
    """A control family that does not vary an axis is flat, not wrong."""
    rows = [
        ("control_synthetic", 0.5, 0.2, 3.0, 50.0),
        ("control_synthetic", 0.5, 0.2, 3.0, 50.0),
        ("control_synthetic", 0.5, 0.2, 3.0, 50.0),
    ]
    checks = correlations(_frame(spark, rows), min_rows=2)
    # A zero-variance column yields a null correlation in Spark.
    assert {check.verdict for check in checks} <= {"FLAT", "NO DATA"}
    assert wrong_sign_checks(checks) == []


def test_undefined_correlation_is_not_a_wrong_sign():
    check = CorrelationCheck("llm_generated", "gwp", 1, None, "negative")
    assert check.observed == "undefined"
    assert check.verdict == "NO DATA"


def test_summary_renders_every_check(spark):
    rows = [
        ("llm_generated", 0.2, 0.40, 2.0, 20.0),
        ("llm_generated", 0.8, 0.10, 5.0, 80.0),
    ]
    checks = correlations(_frame(spark, rows), min_rows=2)
    summary = format_summary(checks)
    assert "corr(pref, x)" in summary
    assert summary.count("\n") == len(checks) + 1  # header + rule + one line each


def test_small_samples_get_no_verdict(spark):
    """A strong correlation over four rows is noise, and must not fail a build.

    Without this guard the fixture's five control rows produce a confident
    `circ_orig` correlation of the wrong sign, purely from the sample size.
    """
    rows = [
        ("llm_generated", 0.95, 0.45, 1.0, 10.0),
        ("llm_generated", 0.7, 0.30, 3.0, 35.0),
        ("llm_generated", 0.4, 0.18, 4.0, 60.0),
        ("llm_generated", 0.1, 0.07, 6.0, 90.0),
    ]
    checks = correlations(_frame(spark, rows))  # default threshold
    assert {check.verdict for check in checks} == {"TOO FEW"}
    assert wrong_sign_checks(checks) == []


def test_default_threshold_is_the_documented_one():
    assert CorrelationCheck("s", "gwp", MIN_ROWS_FOR_DIRECTION - 1, -0.9, "negative").verdict == (
        "TOO FEW"
    )
    assert CorrelationCheck("s", "gwp", MIN_ROWS_FOR_DIRECTION, -0.9, "negative").verdict == "OK"


def test_runs_over_real_gold_output(gold):
    """Smoke test: the check works on the pipeline's actual output shape."""
    checks = correlations(gold)
    assert len(checks) == len(EXPECTED_DIRECTIONS) * 4  # 3 source types + ALL
    assert all(check.verdict in VERDICTS for check in checks)
