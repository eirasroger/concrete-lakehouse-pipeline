# Databricks notebook source
# MAGIC %md
# MAGIC # Build the tables, then check them
# MAGIC
# MAGIC Turns the two raw JSON files into three layers of Delta tables in Unity
# MAGIC Catalog, then checks the result.
# MAGIC
# MAGIC Before running, do this once:
# MAGIC
# MAGIC 1. In **Catalog**, create a schema `concrete` in the `workspace` catalog.
# MAGIC 2. In that schema, create a **volume** called `raw`.
# MAGIC 3. Upload `labelled_alternatives.json` and `frozen_dataset.json` into it.
# MAGIC 4. Run this notebook top to bottom on serverless compute.
# MAGIC
# MAGIC All the logic lives in `src/concrete_pipeline/` in this repo. The notebook
# MAGIC only points it at the right paths, so the laptop run and this run execute
# MAGIC the same code.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make the repo's code importable

# COMMAND ----------

import sys
from pathlib import Path


def find_repo_root() -> Path:
    """Walk up from the notebook until we find the folder containing src/."""
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "src" / "concrete_pipeline").is_dir():
            return candidate
    raise RuntimeError(f"Could not find src/concrete_pipeline above {Path.cwd()}")


REPO_ROOT = find_repo_root()
sys.path.insert(0, str(REPO_ROOT / "src"))
print("repo root:", REPO_ROOT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Where to read from and write to
# MAGIC
# MAGIC `namespace` sends the tables to Unity Catalog instead of a local folder.
# MAGIC That is the only difference between this run and a laptop run.

# COMMAND ----------

from concrete_pipeline import config as cfg

CATALOG = "workspace"
SCHEMA = "concrete"
RAW_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/raw"

config = cfg.PipelineConfig(
    raw_dir=Path(RAW_VOLUME),
    namespace=f"{CATALOG}.{SCHEMA}",
)

print("reading from :", config.raw_dir)
print("writing to   :", config.table_ref("gold_scenarios"))

# COMMAND ----------

# MAGIC %md
# MAGIC Create the schema if it isn't there yet, and check the raw files are
# MAGIC visible. If this cell fails, the volume upload (step 2-3 above) is missing.

# COMMAND ----------

from concrete_pipeline.session import get_spark

spark = get_spark()
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

print(config.resolve_label_file())
print(config.resolve_feature_file())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build bronze, silver and gold
# MAGIC
# MAGIC - **bronze** — the raw files landed unchanged
# MAGIC - **silver** — the nested product lists unpacked to one row per product
# MAGIC - **gold** — the two files joined into the table you would model on
# MAGIC
# MAGIC `get_spark()` returns the notebook's existing session, so none of the
# MAGIC local Spark configuration applies here.

# COMMAND ----------

from concrete_pipeline.pipeline import print_run_report, run_pipeline

result = run_pipeline(spark, config)
print_run_report(result)

# COMMAND ----------

# MAGIC %md
# MAGIC The tables are now in Unity Catalog and queryable with plain SQL.

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES IN workspace.concrete

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT source_type, count(*) AS rows, count(DISTINCT scenario_id) AS scenarios
# MAGIC FROM workspace.concrete.gold_scenarios
# MAGIC GROUP BY source_type
# MAGIC ORDER BY rows DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## What the quality gate rejected
# MAGIC
# MAGIC Rows that failed a check are kept in their own table, with the reason
# MAGIC attached, instead of being dropped. On the real files this catches 4 rows:
# MAGIC scenarios `2647` and `13122` each label `prod_1` twice, with contradictory
# MAGIC preference scores.

# COMMAND ----------

display(
    result.gold_rejected.select(
        "scenario_id", "id_prod", "source_type", "pref", "conf", "rejection_reasons"
    ).orderBy("scenario_id", "id_prod")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Does the table make sense?
# MAGIC
# MAGIC Not a model. If the join matched the wrong label to the wrong product, then
# MAGIC preference would not track the sustainability variables. So we check only
# MAGIC the direction: preferred products should have **lower** carbon footprint
# MAGIC (`gwp`) and **better** health and circularity scores.

# COMMAND ----------

from concrete_pipeline.validation import correlations, format_summary, wrong_sign_checks

checks = correlations(result.gold_scenarios)
print(format_summary(checks))

wrong = wrong_sign_checks(checks)
assert not wrong, f"correlations point the wrong way: {wrong}"
print("\nOK: every direction is as expected.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### The sharper version of the same check
# MAGIC
# MAGIC The `control_*` scenarios are synthetic probes: each family varies one
# MAGIC variable and holds the rest steady. So each family should light up on its
# MAGIC own variable and sit near zero on the others. It does — `gwp` scores
# MAGIC -0.9994 on `gwp`, `health` scores +0.9152 on `health`. A join that paired
# MAGIC the wrong rows could not produce that pattern.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   regexp_extract(scenario_id, '^control_([a-z][a-z_]*[a-z])_[0-9]+$', 1) AS control_axis,
# MAGIC   count(*) AS rows,
# MAGIC   round(corr(pref, gwp), 4)       AS corr_gwp,
# MAGIC   round(corr(pref, health), 4)    AS corr_health,
# MAGIC   round(corr(pref, circ_orig), 4) AS corr_circ_orig
# MAGIC FROM workspace.concrete.gold_scenarios
# MAGIC WHERE source_type = 'control_synthetic'
# MAGIC GROUP BY control_axis
# MAGIC ORDER BY control_axis

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table history
# MAGIC
# MAGIC Every run adds a version. `02_delta_time_travel` uses this to compare a
# MAGIC fixed classifier against the buggy one it replaced.

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY workspace.concrete.gold_scenarios
