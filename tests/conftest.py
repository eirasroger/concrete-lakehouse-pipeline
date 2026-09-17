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

# Heap size is fixed when the JVM launches, so it has to be set here rather than
# through SparkConf. Spark's 1 GB default is not enough for this suite: one
# session serves every test, MERGE holds both sides in memory, and the tables
# stay cached throughout. Running out of heap does not surface as an
# OutOfMemoryError -- the driver stalls in GC, misses its heartbeats, and the
# run dies with "NullPointerException ... idWithoutTopologyInfo is null", which
# says nothing about memory at all.
#
# Asked for rather than assumed: a hosted CI runner has far less RAM than a
# laptop, and a JVM that cannot reserve its heap fails immediately with
# "Could not reserve enough space for object heap". Override with
# CONCRETE_TEST_DRIVER_MEMORY.
TEST_DRIVER_MEMORY = os.environ.get("CONCRETE_TEST_DRIVER_MEMORY", "2g")
os.environ.setdefault(
    "PYSPARK_SUBMIT_ARGS", f"--driver-memory {TEST_DRIVER_MEMORY} pyspark-shell"
)

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
    # Give the driver room to pause for GC without the heartbeat declaring it
    # dead. On a single JVM serving the whole suite, a long collection is normal
    # and is not a reason to tear the session down.
    "spark.executor.heartbeatInterval": "60s",
    "spark.network.timeout": "300s",
}


#: Any test that requests one of these needs a JVM, so it gets marked `spark`
#: automatically. Deriving the mark from the fixtures beats decorating every
#: test, which is the sort of bookkeeping that goes stale the moment it is added.
SPARK_FIXTURES = frozenset({"spark", "result", "gold", "rejected", "stages"})


def pytest_collection_modifyitems(config, items):
    for item in items:
        if SPARK_FIXTURES & set(getattr(item, "fixturenames", ())):
            item.add_marker(pytest.mark.spark)


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
    # Cache silver and gold, which the assertions read repeatedly. Bronze is
    # deliberately left uncached: it is a windowed view over an append-only
    # table, so caching it pins the nested arrays in heap for the whole session
    # to save a scan of five rows.
    for layer in (outcome.silver, outcome.gold):
        for df in layer.values():
            df.cache()
    return outcome


@pytest.fixture(scope="session")
def gold(result):
    return result.gold_scenarios


@pytest.fixture(scope="session")
def rejected(result):
    return result.gold_rejected
