"""Shared pytest fixtures.

One Spark session is built for the whole session (JVM startup dominates the
runtime), and the pipeline runs once against the synthetic fixture into a
temporary lakehouse directory that every test reads.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Spark launches its workers by re-invoking this interpreter; without these it
# picks whatever `python` is on PATH, which may not be the one running pytest.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from concrete_pipeline import config as cfg  # noqa: E402
from concrete_pipeline.pipeline import run_pipeline  # noqa: E402
from concrete_pipeline.session import get_spark  # noqa: E402


# On fixture-sized data every one of these costs more than it saves: adaptive
# query execution re-plans each stage, the broadcast threshold triggers a
# separate job per join, and Delta's default log machinery writes more than it
# reads back. Stripping them cuts the suite's runtime several-fold.
FAST_LOCAL_CONF = {
    "spark.sql.adaptive.enabled": "false",
    "spark.sql.autoBroadcastJoinThreshold": "-1",
    "spark.sql.shuffle.partitions": "1",
    "spark.default.parallelism": "1",
    "spark.databricks.delta.snapshotPartitions": "1",
    "spark.databricks.delta.stats.collect": "false",
    "spark.databricks.delta.stalenessLimit": "3600000",
    "spark.sql.sources.parallelPartitionDiscovery.parallelism": "1",
    "spark.rdd.compress": "false",
}


@pytest.fixture(scope="session")
def spark():
    session = get_spark(
        app_name="concrete-pipeline-tests",
        shuffle_partitions=1,
        extra_conf=FAST_LOCAL_CONF,
    )
    yield session
    session.stop()


@pytest.fixture(scope="session")
def lakehouse_dir():
    path = Path(tempfile.mkdtemp(prefix="concrete-lakehouse-"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="session")
def fixture_config(lakehouse_dir) -> cfg.PipelineConfig:
    return cfg.PipelineConfig.for_fixtures(lakehouse_dir)


@pytest.fixture(scope="session")
def result(spark, fixture_config):
    """The full pipeline, run once over the synthetic fixture.

    Every table is cached: the suite reads each of them repeatedly, and without
    this each assertion re-reads the Delta log and re-scans the parquet.
    """
    outcome = run_pipeline(spark, fixture_config)
    for layer in (outcome.bronze, outcome.silver, outcome.gold):
        for df in layer.values():
            df.cache()
    return outcome


@pytest.fixture(scope="session")
def gold(result):
    return result.gold_scenarios


@pytest.fixture(scope="session")
def rejected(result):
    return result.gold_rejected
