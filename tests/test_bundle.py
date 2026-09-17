"""Structural checks on the Databricks Asset Bundle.

`databricks bundle validate` is the real check, but it resolves the current user
through the workspace API, so it needs credentials and cannot run in CI. These
assertions cover the parts that break silently without one: a task pointing at a
script that has been renamed, an entry point dropped from the job, a target
removed.

Kept as tests rather than a CI shell step so they run locally in the `not spark`
subset, and so CI gets them without a bespoke step and its own dependencies.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = REPO_ROOT / "databricks.yml"

EXPECTED_TASKS = {"build_tables", "validate_gold", "maintain_tables"}


@pytest.fixture(scope="module")
def bundle() -> dict:
    return yaml.safe_load(BUNDLE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def job(bundle) -> dict:
    return bundle["resources"]["jobs"]["concrete_pipeline"]


def test_bundle_defines_the_expected_tasks(job):
    assert {t["task_key"] for t in job["tasks"]} == EXPECTED_TASKS


def test_every_task_entry_point_exists(job):
    """A renamed script would otherwise fail only at deploy time."""
    for task in job["tasks"]:
        script = REPO_ROOT / task["spark_python_task"]["python_file"]
        assert script.is_file(), f"{task['task_key']} -> missing {script}"


def test_tasks_run_in_order(job):
    """Validation gates maintenance: never compact a batch that failed its checks."""
    depends = {
        t["task_key"]: {d["task_key"] for d in t.get("depends_on", [])} for t in job["tasks"]
    }
    assert depends["build_tables"] == set()
    assert depends["validate_gold"] == {"build_tables"}
    assert depends["maintain_tables"] == {"validate_gold"}


def test_the_pipeline_ships_as_a_wheel(bundle):
    """Databricks runs a python_file through exec(), where __file__ is undefined,
    so the package has to be installed rather than imported by path."""
    assert bundle["artifacts"]["concrete_pipeline"]["type"] == "whl"
    for environment in job_environments(bundle):
        assert any(dep.endswith(".whl") for dep in environment["spec"].get("dependencies", []))


def job_environments(bundle) -> list[dict]:
    return bundle["resources"]["jobs"]["concrete_pipeline"]["environments"]


def test_serverless_only(bundle):
    """No job_clusters: Free Edition has none, and there is no cluster config to drift."""
    assert "job_clusters" not in bundle["resources"]["jobs"]["concrete_pipeline"]
    assert job_environments(bundle)


def test_dev_and_prod_targets_exist(bundle):
    assert set(bundle["targets"]) == {"dev", "prod"}
    assert bundle["targets"]["dev"]["mode"] == "development"
    assert bundle["targets"]["prod"]["mode"] == "production"


def test_dev_schedule_is_paused(job):
    """A dev deploy must never start firing on its own."""
    assert job["schedule"]["pause_status"] == "PAUSED"


def test_prod_unpauses_the_schedule(bundle):
    prod_job = bundle["targets"]["prod"]["resources"]["jobs"]["concrete_pipeline"]
    assert prod_job["schedule"]["pause_status"] == "UNPAUSED"


def test_catalog_and_schema_are_variables(bundle):
    """So prod can point somewhere else without touching code."""
    assert {"catalog", "schema", "raw_dir"} <= set(bundle["variables"])
