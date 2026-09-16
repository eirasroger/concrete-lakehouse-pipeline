"""The gold modelling table: grain, join correctness and column contract."""

from __future__ import annotations

from pyspark.sql import functions as F

from concrete_pipeline.gold import GOLD_COLUMN_ORDER
from concrete_pipeline.schemas import COST_COLUMNS, FEATURE_COLUMNS


def test_gold_has_the_agreed_columns_in_order(gold):
    assert tuple(gold.columns) == GOLD_COLUMN_ORDER


def test_gold_carries_label_and_features_together(gold):
    for column in (*FEATURE_COLUMNS, *COST_COLUMNS, "pref", "conf", "reason", "source_type"):
        assert column in gold.columns


def test_gold_grain_is_one_row_per_scenario_product(gold):
    total = gold.count()
    distinct = gold.select("scenario_id", "id_prod").distinct().count()
    assert total == distinct


def test_no_join_produced_nulls_on_either_side(gold):
    """Every promoted row has both a label and its features."""
    assert gold.filter(F.col("pref").isNull()).count() == 0
    assert gold.filter(F.col("compressive_strength").isNull()).count() == 0


def test_the_join_pairs_the_right_label_with_the_right_product(gold):
    """A join on scenario_id alone would silently cross-pair products."""
    row = gold.filter(
        (F.col("scenario_id") == "control_health_1068") & (F.col("id_prod") == "prod_2")
    ).collect()[0]
    assert row["pref"] == 1.0  # label for prod_2
    assert row["health"] == 6.0  # feature for prod_2
    assert row["density"] == 2410.0

    other = gold.filter(
        (F.col("scenario_id") == "control_health_1068") & (F.col("id_prod") == "prod_3")
    ).collect()[0]
    assert other["pref"] == 0.0
    assert other["health"] == 0.0


def test_scenario_attributes_are_attached_as_arrays(gold):
    row = gold.filter(F.col("scenario_id") == "3782").collect()[0]
    assert len(row["stakeholder_preferences"]) == 2
    assert len(row["situations"]) == 2
    assert "Thermal insulation" in row["situations"]


def test_single_valued_scenario_attributes_stay_length_one(gold):
    row = gold.filter(F.col("scenario_id") == "expert_118").collect()[0]
    assert len(row["situations"]) == 1
    assert row["situations"] == ["Standard structural application"]


def test_source_type_mix_matches_the_fixture(gold):
    mix = {
        row["source_type"]: row["count"]
        for row in gold.groupBy("source_type").count().collect()
    }
    assert mix == {"control_synthetic": 5, "llm_generated": 6, "expert_annotated": 2}


def test_rejected_table_mirrors_gold_plus_reasons(gold, rejected):
    assert tuple(rejected.columns) == (*GOLD_COLUMN_ORDER, "rejection_reasons")


def test_gold_is_a_delta_table(spark, fixture_config):
    """Readable as Delta, with a version history -- not just parquet in a folder."""
    from concrete_pipeline import config as cfg
    from concrete_pipeline.bronze import delta_table

    assert delta_table(spark, fixture_config, cfg.GOLD_SCENARIOS).history().count() >= 1


def test_time_travel_reads_the_current_version(spark, fixture_config, gold):
    """The mechanism the Delta demo relies on, exercised on the fixture."""
    from concrete_pipeline import config as cfg
    from concrete_pipeline.bronze import latest_version, read_delta_version

    version = latest_version(spark, fixture_config, cfg.GOLD_SCENARIOS)
    assert read_delta_version(spark, fixture_config, cfg.GOLD_SCENARIOS, version).count() == (
        gold.count()
    )


def test_local_config_writes_paths_not_catalog_tables(fixture_config):
    """The switch the Databricks run flips. Locally it must stay off."""
    from concrete_pipeline import config as cfg

    assert not fixture_config.uses_catalog
    assert fixture_config.table_ref(cfg.GOLD_SCENARIOS).endswith("gold_scenarios")


def test_namespace_config_addresses_unity_catalog():
    """With a namespace set, tables are named, not pathed."""
    from dataclasses import replace

    from concrete_pipeline import config as cfg

    databricks = replace(cfg.PipelineConfig(), namespace="workspace.concrete")
    assert databricks.uses_catalog
    assert databricks.table_ref(cfg.GOLD_SCENARIOS) == "workspace.concrete.gold_scenarios"
