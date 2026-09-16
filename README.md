# concrete-lakehouse-pipeline

Two messy JSON files turned into one clean, validated table, using PySpark and
Delta Lake on Databricks.

The data is a published research dataset of concrete product recommendations
([DOI 10.34810/DATA3164](https://doi.org/10.34810/DATA3164)): 42,874 decision
scenarios, each with a handful of candidate concrete mixes. It arrives as two
separate files that have to be matched up to be useful — one holds the
preference scores, the other holds the product properties. Neither is much use
alone.

## What it does

```mermaid
flowchart LR
    R1["labelled_alternatives.json<br/>preference scores"] --> B
    R2["frozen_dataset.json<br/>product properties"] --> B
    B["<b>BRONZE</b><br/>raw files, unchanged"] --> S
    S["<b>SILVER</b><br/>nested lists unpacked<br/>one row per product"] --> Q
    Q{"<b>quality gate</b>"} -->|passes| G["<b>GOLD</b><br/>149,668 rows<br/>ready to model on"]
    Q -->|fails| X["<b>rejected</b><br/>4 rows + reasons"]

    style B fill:#f0e0cc,stroke:#c99b5e
    style S fill:#e8e8ec,stroke:#9a9aa8
    style G fill:#f7edc8,stroke:#c9a227
    style X fill:#f5dcdc,stroke:#c98080
    style Q fill:#e6eef5,stroke:#6b93b8
```

**Bronze** copies the raw files into Delta tables and changes nothing, so
there's always an untouched original to go back to.

**Silver** unpacks the nesting. In the raw files a scenario contains a list of
2–5 products; silver flattens that to one row per product, and splits the
scenario-level fields (stakeholder, situation) into their own tables.

**Gold** joins the two sides on `(scenario, product)`. This is the table you'd
actually use.

Splitting it this way means a bug in the unpacking is fixed by rebuilding silver
and gold — no need to touch the 140 MB of raw input again.

## The tables

| Table | One row per | Rows |
|---|---|---|
| `bronze_labels` / `bronze_features` | scenario, still nested | 42,874 each |
| `silver_labels` | scenario + product, with preference score | 149,672 |
| `silver_features` | scenario + product, with properties | 149,670 |
| `silver_scenario_stakeholder` | scenario + stakeholder | 45,146 |
| `silver_scenario_situation` | scenario + situation | 43,755 |
| **`gold_scenarios`** | **scenario + product, everything joined** | **149,668** |
| `gold_scenarios_rejected` | rejected row, with reasons | 4 |

## The quality gate

Before anything reaches gold it gets checked. A row that fails is written to
`gold_scenarios_rejected` with the reason attached, rather than dropped
silently. Promoted + rejected always adds up to the joined total.

| Check | Failures on the real data |
|---|---|
| `pref` and `conf` between 0 and 1 | 0 |
| No duplicate `(scenario, product)` | **4** |
| Both files agree on which products a scenario has | 0 |
| Every label has matching properties, and vice versa | 0 |
| `compressive_strength` 5–150 MPa | 0 |
| `water_to_cement_ratio` 0.20–1.00 | 0 |
| `density` 1200–3000 kg/m³ | 0 |

Those 4 failures are a real error in the source data: scenarios `2647` and
`13122` each label product `prod_1` **twice**, with contradictory scores (0.62
at confidence 0.75 against 0.05 at confidence 0.95). Both copies are rejected —
the source gives no basis for picking a winner. The other products in those
scenarios promote normally.

## Where the data came from (`source_type`)

The scenario IDs mix three different kinds of data in one file. Every gold row
is tagged so you can filter or compare them:

| ID looks like | `source_type` | Scenarios |
|---|---|---|
| `control_health_1068` | `control_synthetic` — synthetic probes, 8 axes | 24,000 |
| `3782` | `llm_generated` | 18,602 |
| `expert_118` | `expert_annotated` | 272 |

Two traps found by profiling the files (`scripts/explore_raw.py`), both of which
would corrupt the tagging silently:

- `expert_1..272` and the plain IDs `1..272` both count from 1 but are **completely
  different scenarios** — none of the 272 pairs share the same products.
- `control_archfinish` and `control_archfinish_slump` are two separate probes,
  not one with a suffix.

## Does the output make sense?

A direction check, not a model. If the join paired the wrong label with the
wrong product, preference wouldn't track anything — and row counts would never
reveal it, because a mis-keyed join gives you exactly the right number of
exactly wrong rows.

Each `control_*` family varies one variable and holds the rest steady, so each
should light up on its own variable and sit near zero elsewhere:

```
control_axis              rows          gwp       health    circ_orig
gwp                     10,459      -0.9994      +0.0067      +0.0165
health                  10,504      -0.6695      +0.9152      +0.6293
archfinish              10,512      -0.0043      -0.0017      -0.6038
archfinish_slump        10,665      +0.0163      -0.0145      +0.0055
cost                    10,312      +0.0067      -0.0047      -0.0153
density                 10,440      +0.0165      +0.0009      -0.0148
fwu                     10,501      -0.0207      +0.0099      +0.0077
wdp                     10,447      +0.0058      +0.0033      +0.0075
```

That diagonal is the result. `archfinish` is negative on `circ_orig` (−0.60) on
purpose: in that family more recycled content means a worse surface finish.

## Delta time travel

`gold_scenarios` is written with a deliberately buggy `source_type` rule, then
overwritten with the correct one. Delta keeps both versions, so the fix can be
diffed:

```
11,553 rows reclassified, 0 added, 0 removed

before               -> after                      rows   scenarios
llm_generated        -> control_synthetic        10,665       3,000
llm_generated        -> expert_annotated            888         272
```

3,000 and 272 — precisely the `archfinish_slump` family and the expert family.
And the old version is still queryable:

```sql
SELECT * FROM workspace.concrete.gold_scenarios VERSION AS OF 0
```

That's the thing you can't do with CSVs.

---

## Running it on Databricks

Run on **Databricks Free Edition**, serverless compute (Databricks Runtime
19.6.x, Photon). The row counts and correlations in this README were produced
there and reproduced byte-for-byte on a local Spark run. One-time setup:

1. **Catalog** → in the `workspace` catalog, create a schema `concrete`, and
   inside it a **volume** named `raw`.
2. Upload `labelled_alternatives.json` and `frozen_dataset.json` to that volume.
3. **Workspace → Repos** → clone this repo (already done if you're reading it there).
4. Run `notebooks/01_build_and_validate.py`, then `notebooks/02_delta_time_travel.py`.

Both notebooks run on serverless. The first cell prints where it's reading from
and writing to, so a wrong path fails immediately instead of halfway through.

The tables land in Unity Catalog, so they show up in Catalog Explorer and can be
queried with plain SQL.

The notebooks contain no logic — they set two paths and call into
`src/concrete_pipeline/`. The only difference from a laptop run is one config
line:

```python
config = PipelineConfig(
    raw_dir=Path("/Volumes/workspace/concrete/raw"),
    namespace="workspace.concrete",          # -> Unity Catalog tables
)
```

Without `namespace`, the same code writes Delta tables to a local folder.

## Deploying it as a scheduled job

`databricks.yml` defines the pipeline as a **Databricks Asset Bundle** — a
scheduled Job, deployed from the command line:

```bash
databricks bundle validate -t dev
databricks bundle deploy   -t dev
databricks bundle run concrete_pipeline -t dev   # trigger it now
```

Two tasks on serverless compute, the second depending on the first. Both ran
green on Free Edition in ~100s end to end, producing the same row counts and
correlations as the notebook and the local run:

| Task | Runs | Fails the job when |
|---|---|---|
| `build_tables` | `scripts/run_pipeline.py --namespace workspace.concrete` | ingestion or the gate errors |
| `validate_gold` | `scripts/validate_gold.py --namespace workspace.concrete` | a correlation points the wrong way |

The job runs **the same entry points a laptop runs**, with `--namespace`
redirecting the output to Unity Catalog. There is no Databricks-specific copy of
the logic anywhere.

`bundle deploy` builds `src/` into a wheel and installs it into the serverless
environment, so `import concrete_pipeline` resolves in the job the same way it
does in a venv. That is not decoration: Databricks executes a `python_file` via
`exec()`, where `__file__` does not exist, so the usual
`sys.path.insert(Path(__file__)...)` trick fails outright. Shipping a package is
the fix.

Targets: `dev` deploys under your user folder, prefixes the job name with
`[dev <you>]` and forces the schedule **paused**, so a dev deploy can never
start firing on its own. `prod` deploys to `/Workspace/Shared` and unpauses the
daily 06:00 schedule. Catalog, schema and volume path are bundle variables, so
prod could point at a different catalog without touching code.

A note on the schedule: this dataset is static, so a daily cron is really just
demonstrating the shape. A real incremental source would use a file-arrival
trigger with Auto Loader, and bronze would `MERGE` rather than `overwrite`.

## Running it locally

Needs Python 3.10–3.13 (PySpark doesn't support 3.14 yet) and a JDK 17.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The raw files are **not** in this repo — they're CC BY-NC 4.0 and ~140 MB. Get
them from [the DOI](https://doi.org/10.34810/DATA3164) and put them in
`data/raw/`. Or skip them: every script takes `--fixtures` and runs against a
5-scenario synthetic stand-in in `tests/fixtures/`.

```bash
python scripts/explore_raw.py            # profile the raw files
python scripts/run_pipeline.py           # build bronze -> silver -> gold
python scripts/validate_gold.py          # the direction check
python scripts/delta_versioning_demo.py  # the time-travel diff
pytest -q                                # 80 tests, ~3 min, fixture only
```

For a real run, give the driver some room first — both files are one big JSON
array, so Spark has to read each in a single task:

```bash
export PYSPARK_SUBMIT_ARGS="--driver-memory 6g pyspark-shell"
```

Windows also needs `winutils.exe` and `hadoop.dll` — see
[docs/windows-setup.md](docs/windows-setup.md).

## Layout

```
src/concrete_pipeline/    all the logic (bronze, silver, gold, quality, validation)
scripts/                  command-line entry points — also what the Job runs
notebooks/                the Databricks exploratory view
databricks.yml            the bundle: pipeline as a scheduled Job
tests/                    pytest suite + synthetic fixture
docs/windows-setup.md     local Windows Spark setup
```

`src/` holds plain functions over DataFrames and imports nothing from
Databricks. Everything else is a thin caller: the scripts parse arguments, the
notebooks set two paths, the bundle schedules the scripts. That is what lets the
same code run on a laptop and on serverless and produce identical output.

Tests run against the fixture only, so they need neither the raw files nor
Databricks. CI runs them on Python 3.11 and 3.12 on every push.

## Citation

Dataset: CORA.RDR, [https://doi.org/10.34810/DATA3164](https://doi.org/10.34810/DATA3164),
CC BY-NC 4.0 — cite using the string on the DOI landing page.

Paper: *Context-adaptive deep learning for sustainable product recommendation:
Application to concrete*, Sustainable Production and Consumption, 2026.

This repo is an engineering re-implementation of the dataset's preparation, not
a redistribution of the data.
