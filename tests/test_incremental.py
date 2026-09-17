"""Incremental ingestion, end to end, through all four stages.

The four stages run **once** in a module-scoped fixture and each stage's state is
captured; the test functions then assert against those snapshots. Running the
pipeline four times per assertion would be correct and unbearably slow --
`MERGE` is several Spark jobs, and there are a lot of assertions here.

The stages deliberately mirror the two arrival patterns:

    1. first load        both base files are new
    2. no-op             nothing changed, so nothing should happen
    3. new file          a batch lands alongside the originals
    4. replaced file     a file is rewritten under the same name
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field

import pytest
from pyspark.sql import functions as F

from concrete_pipeline import config as cfg
from concrete_pipeline import ingest
from concrete_pipeline.pipeline import run_pipeline

FIXTURES = cfg.REPO_ROOT / "tests" / "fixtures"


@dataclass
class Snapshot:
    """What one pipeline run produced."""

    counts: dict[str, int]
    skipped: bool
    ingested: list[str]
    changed: list[str] | None
    rejected_keys: list[tuple] = field(default_factory=list)
    labels_2647: list[tuple] = field(default_factory=list)
    pref_1068_prod1: float | None = None


def _snapshot(result) -> Snapshot:
    rejected = result.gold_rejected
    labels = result.silver[cfg.SILVER_LABELS]
    gold = result.gold_scenarios

    pref_row = (
        gold.filter(
            (F.col("scenario_id") == "control_health_1068") & (F.col("id_prod") == "prod_1")
        )
        .select("pref")
        .collect()
    )

    return Snapshot(
        counts=result.counts(),
        skipped=result.skipped,
        ingested=sorted(s.name for s in result.ingested_files),
        changed=None if result.changed_scenarios is None else sorted(result.changed_scenarios),
        rejected_keys=sorted(
            (row["scenario_id"], row["id_prod"], row["pref"])
            for row in rejected.select("scenario_id", "id_prod", "pref").collect()
        ),
        labels_2647=sorted(
            (row["label_ordinal"], row["id_prod"], row["pref"])
            for row in labels.filter(F.col("scenario_id") == "2647")
            .select("label_ordinal", "id_prod", "pref")
            .collect()
        ),
        pref_1068_prod1=pref_row[0]["pref"] if pref_row else None,
    )


@pytest.fixture(scope="module")
def stages(spark, tmp_path_factory):
    """Run the four stages once, capturing state after each."""
    root = tmp_path_factory.mktemp("incremental")
    raw = root / "raw"
    raw.mkdir()
    config = cfg.PipelineConfig(
        raw_dir=raw,
        lakehouse_dir=root / "lake",
        gate_limits=cfg.GateLimits(max_rejected_fraction=1.0),
    )

    for name in ("labelled_alternatives.json", "frozen_dataset.json"):
        shutil.copy(FIXTURES / name, raw / name)
    first = _snapshot(run_pipeline(spark, config))

    # Nothing touched between runs.
    noop = _snapshot(run_pipeline(spark, config))

    for name in ("labelled_alternatives_batch2.json", "frozen_dataset_batch2.json"):
        shutil.copy(FIXTURES / "batch2" / name, raw / name)
    added = _snapshot(run_pipeline(spark, config))

    # Same filename, corrected content.
    shutil.copy(FIXTURES / "revised" / "labelled_alternatives.json", raw)
    revised = _snapshot(run_pipeline(spark, config))

    return {"first": first, "noop": noop, "added": added, "revised": revised, "config": config}


# --------------------------------------------------------------------------
# Stage 1: first load
# --------------------------------------------------------------------------


def test_first_load_ingests_both_files(stages):
    first = stages["first"]
    assert first.ingested == ["frozen_dataset.json", "labelled_alternatives.json"]
    assert not first.skipped


def test_first_load_matches_the_non_incremental_result(stages):
    """The refactor must not have changed what the pipeline produces."""
    counts = stages["first"].counts
    assert counts[cfg.SILVER_LABELS] == 17
    assert counts[cfg.SILVER_FEATURES] == 16
    assert counts[cfg.GOLD_SCENARIOS] == 13
    assert counts[cfg.GOLD_SCENARIOS_REJECTED] == 4


# --------------------------------------------------------------------------
# Stage 2: nothing changed
# --------------------------------------------------------------------------


def test_rerun_with_no_changes_does_nothing(stages):
    """The property that makes a scheduled job safe to run on a timer."""
    noop = stages["noop"]
    assert noop.skipped
    assert noop.ingested == []


def test_noop_run_leaves_every_table_identical(stages):
    assert stages["noop"].counts == stages["first"].counts


def test_noop_run_does_not_duplicate_bronze(stages):
    """Bronze is append-only, so a double load would silently double the rows."""
    assert stages["noop"].counts[cfg.BRONZE_LABELS] == 5


# --------------------------------------------------------------------------
# Stage 3: a new file arrives
# --------------------------------------------------------------------------


def test_new_batch_is_ingested(stages):
    added = stages["added"]
    assert added.ingested == [
        "frozen_dataset_batch2.json",
        "labelled_alternatives_batch2.json",
    ]
    assert not added.skipped


def test_new_batch_only_recomputes_its_own_scenarios(stages):
    """The assertion that proves the run was incremental, not a rebuild."""
    assert stages["added"].changed == ["control_gwp_777", "expert_9"]


def test_new_batch_adds_its_rows_and_keeps_the_rest(stages):
    counts = stages["added"].counts
    assert counts[cfg.SILVER_LABELS] == 17 + 5
    assert counts[cfg.SILVER_FEATURES] == 16 + 5
    assert counts[cfg.GOLD_SCENARIOS] == 13 + 5
    assert counts[cfg.GOLD_SCENARIOS_REJECTED] == 4  # unchanged


def test_new_batch_rows_are_classified(stages, spark):
    gold = stages["config"]
    from concrete_pipeline.bronze import read_delta

    rows = (
        read_delta(spark, gold, cfg.GOLD_SCENARIOS)
        .filter(F.col("scenario_id").isin(["control_gwp_777", "expert_9"]))
        .select("scenario_id", "source_type")
        .distinct()
        .collect()
    )
    mix = {row["scenario_id"]: row["source_type"] for row in rows}
    assert mix == {"control_gwp_777": "control_synthetic", "expert_9": "expert_annotated"}


# --------------------------------------------------------------------------
# Stage 4: a file is replaced under the same name
# --------------------------------------------------------------------------


def test_replaced_file_is_detected_by_content(stages):
    """Auto Loader cannot see this; the content hash is what catches it."""
    revised = stages["revised"]
    assert revised.ingested == ["labelled_alternatives.json"]
    assert not revised.skipped


def test_replaced_file_updates_a_changed_value(stages):
    """control_health_1068/prod_1 pref went 0.393 -> 0.5."""
    assert stages["first"].pref_1068_prod1 == 0.393
    assert stages["revised"].pref_1068_prod1 == 0.5


def test_a_scenario_that_lost_a_row_has_it_deleted(stages):
    """Scenario 2647 went from 6 labels to 5.

    Without `WHEN NOT MATCHED BY SOURCE THEN DELETE` the sixth row would survive
    forever, and no row count would reveal it.
    """
    before = stages["first"].labels_2647
    after = stages["revised"].labels_2647
    assert len(before) == 6
    assert len(after) == 5
    assert max(o for o, _, _ in before) == 5
    assert max(o for o, _, _ in after) == 4


def test_resolving_the_duplicate_promotes_the_row(stages):
    """2647/prod_1 was rejected as a duplicate; the revision leaves one copy."""
    first_keys = {(s, p) for s, p, _ in stages["first"].rejected_keys}
    revised_keys = {(s, p) for s, p, _ in stages["revised"].rejected_keys}
    assert ("2647", "prod_1") in first_keys
    assert ("2647", "prod_1") not in revised_keys


def test_corrected_out_of_bounds_row_promotes(stages):
    """2647/prod_5 pref went 1.4 -> 0.97, so it stops being rejected."""
    revised_keys = {(s, p) for s, p, _ in stages["revised"].rejected_keys}
    assert ("2647", "prod_5") not in revised_keys


def test_only_the_still_broken_row_remains_rejected(stages):
    """3782/prod_4 was never revised, so its range violations stand."""
    revised = stages["revised"]
    assert len(revised.rejected_keys) == 1
    assert revised.rejected_keys[0][:2] == ("3782", "prod_4")


def test_final_counts_add_up(stages):
    counts = stages["revised"].counts
    # 16 base labels after the revision (2647 lost one) + 5 from batch2
    assert counts[cfg.SILVER_LABELS] == 21
    assert counts[cfg.SILVER_FEATURES] == 21
    assert counts[cfg.GOLD_SCENARIOS] == 20
    assert counts[cfg.GOLD_SCENARIOS_REJECTED] == 1
    assert (
        counts[cfg.GOLD_SCENARIOS] + counts[cfg.GOLD_SCENARIOS_REJECTED]
        == counts[cfg.SILVER_LABELS]
    )


# --------------------------------------------------------------------------
# The ingestion log itself
# --------------------------------------------------------------------------


def test_ingest_log_records_every_file_version(stages, spark):
    """Two versions of labelled_alternatives.json, distinguished by hash."""
    from concrete_pipeline.bronze import read_delta

    rows = read_delta(spark, stages["config"], cfg.BRONZE_INGEST_LOG).collect()
    labels = [r for r in rows if r["source_file"] == "labelled_alternatives.json"]
    assert len(labels) == 2
    assert len({r["file_hash"] for r in labels}) == 2


def test_bronze_keeps_both_versions_but_exposes_the_newest(stages, spark):
    """Append-only bronze, with the current view resolving to one version."""
    from concrete_pipeline.bronze import current_bronze, read_delta

    raw_bronze = read_delta(spark, stages["config"], cfg.BRONZE_LABELS)
    # 5 base + 2 batch2 + 5 revised = 12 rows landed in total
    assert raw_bronze.count() == 12
    # The current view is one row per scenario: 5 base + 2 from batch2.
    assert current_bronze(raw_bronze).count() == 7


def test_pending_sources_is_empty_once_everything_is_ingested(stages, spark):
    assert ingest.pending_sources(spark, stages["config"]) == []
