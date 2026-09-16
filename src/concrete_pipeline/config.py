"""Paths, table names and quality-gate thresholds.

Every tunable the pipeline reads lives here, so a run can be pointed at the
synthetic fixture instead of the real 100 MB+ files by swapping one object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# The labels file has been distributed under both names; the loader accepts either.
LABEL_FILE_CANDIDATES = ("labelled_alternatives.json", "labelled_dataset.json")
FEATURE_FILE_CANDIDATES = ("frozen_dataset.json",)

BRONZE_LABELS = "bronze_labels"
BRONZE_FEATURES = "bronze_features"
SILVER_LABELS = "silver_labels"
SILVER_FEATURES = "silver_features"
SILVER_SCENARIO_STAKEHOLDER = "silver_scenario_stakeholder"
SILVER_SCENARIO_SITUATION = "silver_scenario_situation"
GOLD_SCENARIOS = "gold_scenarios"
GOLD_SCENARIOS_REJECTED = "gold_scenarios_rejected"


@dataclass(frozen=True)
class RangeRule:
    """An inclusive plausibility range for one numeric column."""

    column: str
    minimum: float
    maximum: float

    @property
    def rejection_reason(self) -> str:
        return f"{self.column}_out_of_range"


@dataclass(frozen=True)
class QualityThresholds:
    """Bounds enforced by the silver -> gold quality gate.

    The physical ranges are deliberately wider than the ranges observed in the
    published dataset (compressive_strength 8-60 MPa, w/c 0.30-0.80, density
    1400-3000 kg/m3). They are meant to catch genuine corruption -- negatives,
    zeros, order-of-magnitude slips -- without rejecting legitimate lightweight
    or ultra-high-performance mixes that a future release might add.
    """

    pref_min: float = 0.0
    pref_max: float = 1.0
    conf_min: float = 0.0
    conf_max: float = 1.0
    ranges: tuple[RangeRule, ...] = (
        RangeRule("compressive_strength", 5.0, 150.0),
        RangeRule("water_to_cement_ratio", 0.20, 1.00),
        RangeRule("density", 1200.0, 3000.0),
    )


@dataclass(frozen=True)
class PipelineConfig:
    """Where a single pipeline run reads from and writes to.

    Two write targets are supported, and they are the two places this pipeline
    actually runs:

    * `lakehouse_dir` -- a local folder of Delta tables. Used on a laptop and in
      CI.
    * `namespace` -- a Unity Catalog `catalog.schema`, e.g. `workspace.concrete`.
      Used on Databricks, where tables belong in the catalog so they show up in
      Catalog Explorer and can be queried with plain SQL. When set, it wins.
    """

    raw_dir: Path = REPO_ROOT / "data" / "raw"
    lakehouse_dir: Path = REPO_ROOT / "data" / "lakehouse"
    namespace: str | None = None
    thresholds: QualityThresholds = field(default_factory=QualityThresholds)

    @property
    def uses_catalog(self) -> bool:
        """True when tables are written to Unity Catalog rather than to a folder."""
        return bool(self.namespace)

    def table_ref(self, table: str) -> str:
        """How to address a table: a catalog name, or a filesystem path."""
        if self.uses_catalog:
            return f"{self.namespace}.{table}"
        return str((self.lakehouse_dir / table).resolve())

    # Kept as the explicit name for the path-only case (the Delta demo prints it).
    def table_path(self, table: str) -> str:
        return str((self.lakehouse_dir / table).resolve())

    def resolve_label_file(self) -> Path:
        return _resolve(self.raw_dir, LABEL_FILE_CANDIDATES, "labels")

    def resolve_feature_file(self) -> Path:
        return _resolve(self.raw_dir, FEATURE_FILE_CANDIDATES, "features")

    @classmethod
    def for_fixtures(cls, lakehouse_dir: Path) -> "PipelineConfig":
        """Config pointing at the synthetic fixture, for tests and local demos."""
        return cls(raw_dir=REPO_ROOT / "tests" / "fixtures", lakehouse_dir=lakehouse_dir)


def _resolve(raw_dir: Path, candidates: tuple[str, ...], kind: str) -> Path:
    for name in candidates:
        path = raw_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No {kind} file found in {raw_dir}. Expected one of {candidates}.\n"
        "The raw files are not committed (CC BY-NC 4.0). Download them from\n"
        "https://doi.org/10.34810/DATA3164 and place them in data/raw/."
    )
