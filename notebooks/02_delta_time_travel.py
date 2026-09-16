# Databricks notebook source
# MAGIC %md
# MAGIC # Delta time travel: fixing a bug without losing the old answer
# MAGIC
# MAGIC The scenario IDs come in three shapes, and the pipeline tags each row with
# MAGIC which kind of data it is:
# MAGIC
# MAGIC | ID looks like | source_type |
# MAGIC |---|---|
# MAGIC | `control_health_1068` | `control_synthetic` |
# MAGIC | `3782` | `llm_generated` |
# MAGIC | `expert_118` | `expert_annotated` |
# MAGIC
# MAGIC Suppose you got that rule wrong the first time. This notebook writes the
# MAGIC gold table with a buggy version, overwrites it with the fixed version, then
# MAGIC uses Delta history to show exactly which rows changed — and proves the old
# MAGIC version is still readable.
# MAGIC
# MAGIC That last part is the whole argument for Delta over CSV files.
# MAGIC
# MAGIC Run `01_build_and_validate` first.

# COMMAND ----------

import sys
from pathlib import Path


def find_repo_root() -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "src" / "concrete_pipeline").is_dir():
            return candidate
    raise RuntimeError(f"Could not find src/concrete_pipeline above {Path.cwd()}")


REPO_ROOT = find_repo_root()
sys.path.insert(0, str(REPO_ROOT / "src"))

# COMMAND ----------

from concrete_pipeline import config as cfg

CATALOG = "workspace"
SCHEMA = "concrete_versioning_demo"  # its own schema, so notebook 01's tables are untouched
RAW_VOLUME = f"/Volumes/{CATALOG}/concrete/raw"

spark_sql = f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}"

config = cfg.PipelineConfig(
    raw_dir=Path(RAW_VOLUME),
    namespace=f"{CATALOG}.{SCHEMA}",
)

# COMMAND ----------

from concrete_pipeline.session import get_spark

spark = get_spark()
spark.sql(spark_sql)
print("writing to:", config.table_ref(cfg.GOLD_SCENARIOS))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Version 0 — the buggy rule
# MAGIC
# MAGIC Two mistakes, both easy to make if you guess the ID patterns from a few
# MAGIC examples instead of checking the files:
# MAGIC
# MAGIC 1. No rule for `expert_`, so all 272 expert scenarios get dumped into
# MAGIC    `llm_generated`.
# MAGIC 2. A fixed list of control axes that forgets `archfinish_slump`, so 3,000
# MAGIC    more scenarios end up in `llm_generated` too.

# COMMAND ----------

from concrete_pipeline.bronze import latest_version
from concrete_pipeline.pipeline import run_pipeline
from concrete_pipeline.source_type import naive_source_type_column, source_type_column

first = run_pipeline(spark, config, classifier=naive_source_type_column)
print(f"v0 rows: {first.gold_scenarios.count():,}")
display(first.gold_scenarios.groupBy("source_type").count())

version_before = latest_version(spark, config, cfg.GOLD_SCENARIOS)
print("v0 =", version_before)

# COMMAND ----------

# MAGIC %md
# MAGIC Note `expert_annotated` is missing entirely, and `control_synthetic` is
# MAGIC 10,665 rows short.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Version 1 — the fixed rule

# COMMAND ----------

second = run_pipeline(spark, config, classifier=source_type_column)
print(f"v1 rows: {second.gold_scenarios.count():,}")
display(second.gold_scenarios.groupBy("source_type").count())

version_after = latest_version(spark, config, cfg.GOLD_SCENARIOS)
print("v1 =", version_after)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Both versions are in the history

# COMMAND ----------

display(
    spark.sql(f"DESCRIBE HISTORY {config.table_ref(cfg.GOLD_SCENARIOS)}").select(
        "version", "timestamp", "operation", "operationMetrics"
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## The diff
# MAGIC
# MAGIC Read both versions with `VERSION AS OF` and compare them row by row.
# MAGIC Same row count both times — nothing was added or lost, rows only moved
# MAGIC between categories.

# COMMAND ----------

from pyspark.sql import functions as F

from concrete_pipeline.bronze import read_delta_version

KEYS = ["scenario_id", "id_prod"]

before = read_delta_version(spark, config, cfg.GOLD_SCENARIOS, version_before).select(
    *KEYS, F.col("source_type").alias("before")
)
after = read_delta_version(spark, config, cfg.GOLD_SCENARIOS, version_after).select(
    *KEYS, F.col("source_type").alias("after")
)

changed = before.join(after, on=KEYS, how="fullouter").filter(
    F.col("before") != F.col("after")
)
print(f"{changed.count():,} rows reclassified")

# COMMAND ----------

display(
    changed.groupBy("before", "after")
    .agg(
        F.count(F.lit(1)).alias("rows"),
        F.countDistinct("scenario_id").alias("scenarios"),
    )
    .orderBy(F.col("rows").desc())
)

# COMMAND ----------

# MAGIC %md
# MAGIC 3,000 scenarios and 272 scenarios — exactly the `archfinish_slump` family
# MAGIC and the expert family. The diff reads like a bug report rather than a wall
# MAGIC of changes, which is the point.

# COMMAND ----------

display(changed.select("scenario_id", "before", "after").distinct().orderBy("scenario_id").limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## The old version is still there
# MAGIC
# MAGIC Fixing the bug did not erase what the table used to say. You can query the
# MAGIC old version directly in SQL.

# COMMAND ----------

display(
    spark.sql(
        f"SELECT source_type, count(*) AS rows "
        f"FROM {config.table_ref(cfg.GOLD_SCENARIOS)} VERSION AS OF {version_before} "
        f"GROUP BY source_type ORDER BY rows DESC"
    )
)
