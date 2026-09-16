"""Bronze landing and the silver explodes."""

from __future__ import annotations

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType, StringType

from concrete_pipeline import config as cfg
from concrete_pipeline.schemas import COST_COLUMNS, FEATURE_COLUMNS

FIXTURE_SCENARIOS = 5
FIXTURE_FEATURE_ROWS = 16
FIXTURE_LABEL_ROWS = 17


def test_bronze_keeps_one_row_per_scenario(result):
    assert result.bronze[cfg.BRONZE_LABELS].count() == FIXTURE_SCENARIOS
    assert result.bronze[cfg.BRONZE_FEATURES].count() == FIXTURE_SCENARIOS


def test_bronze_leaves_the_nesting_alone(result):
    """Bronze is a landing zone: the arrays must still be arrays."""
    features = result.bronze[cfg.BRONZE_FEATURES]
    assert isinstance(features.schema["alternatives"].dataType, ArrayType)
    assert isinstance(features.schema["situations"].dataType, ArrayType)
    labels = result.bronze[cfg.BRONZE_LABELS]
    assert isinstance(labels.schema["labelled_alternatives"].dataType, ArrayType)


def test_bronze_records_its_provenance(result):
    features = result.bronze[cfg.BRONZE_FEATURES]
    source_files = {row["_source_file"] for row in features.select("_source_file").collect()}
    assert source_files == {"frozen_dataset.json"}
    assert features.filter(F.col("_ingested_at").isNull()).count() == 0


def test_silver_features_row_per_alternative(result):
    assert result.silver[cfg.SILVER_FEATURES].count() == FIXTURE_FEATURE_ROWS


def test_silver_labels_row_per_label(result):
    """17, not 16 -- scenario 2647 labels prod_1 twice, and silver must not hide it."""
    assert result.silver[cfg.SILVER_LABELS].count() == FIXTURE_LABEL_ROWS


def test_silver_features_flattens_the_cost_struct(result):
    features = result.silver[cfg.SILVER_FEATURES]
    for column in COST_COLUMNS:
        assert column in features.columns
        assert isinstance(features.schema[column].dataType, DoubleType)
    assert "c" not in features.columns


def test_silver_features_carries_every_feature_column(result):
    features = result.silver[cfg.SILVER_FEATURES]
    expected = {"scenario_id", "id_prod", *FEATURE_COLUMNS, *COST_COLUMNS}
    assert set(features.columns) == expected


def test_cost_values_survive_the_flattening(result):
    """Spot-check one row against the fixture, so a mis-aliased column is caught."""
    row = (
        result.silver[cfg.SILVER_FEATURES]
        .filter((F.col("scenario_id") == "control_health_1068") & (F.col("id_prod") == "prod_1"))
        .collect()[0]
    )
    assert row["c_p"] == 0.066
    assert row["c_w"] == 0.012
    assert row["c_m"] == 0.014
    assert row["compressive_strength"] == 52.0
    assert row["density"] == 2463.0


def test_scenario_stakeholder_explodes_multi_value_field(result):
    table = result.silver[cfg.SILVER_SCENARIO_STAKEHOLDER]
    # Four scenarios have one stakeholder, 3782 has two.
    assert table.count() == 6
    assert table.filter(F.col("scenario_id") == "3782").count() == 2
    assert isinstance(table.schema["stakeholder_preference"].dataType, StringType)


def test_scenario_situation_explodes_multi_value_field(result):
    table = result.silver[cfg.SILVER_SCENARIO_SITUATION]
    assert table.count() == 6
    assert table.filter(F.col("scenario_id") == "3782").count() == 2


def test_scenario_tables_do_not_multiply_the_fact_grain(result):
    """The reason these live in their own tables rather than on the label rows."""
    labels = result.silver[cfg.SILVER_LABELS]
    assert labels.filter(F.col("scenario_id") == "3782").count() == 4
