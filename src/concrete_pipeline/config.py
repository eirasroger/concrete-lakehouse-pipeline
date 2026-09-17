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
BRONZE_INGEST_LOG = "bronze_ingest_log"
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


#: Glob patterns matching each raw source. Both arrival patterns are covered by
#: globbing rather than naming exact files: a new batch dropped alongside the
#: original (`labelled_alternatives_batch2.json`) is picked up automatically.
LABEL_FILE_GLOBS = ("labelled_alternatives*.json", "labelled_dataset*.json")
FEATURE_FILE_GLOBS = ("frozen_dataset*.json",)


@dataclass(frozen=True)
class IngestLimits:
    """Bounds on how incremental processing behaves.

    `max_scenarios_for_merge` is the point at which a targeted `MERGE` stops
    paying off. A merge scoped to a set of scenario ids has to name them in its
    delete predicate, so a very large change set is both slower and uglier than
    simply rebuilding the table. Crossing this threshold is normal on a first
    load or a full re-release, not an error.
    """

    max_scenarios_for_merge: int = 5_000


@dataclass(frozen=True)
class GateLimits:
    """When a run should fail rather than just record its rejections.

    The gate always routes bad rows to `gold_scenarios_rejected`. These bounds
    decide when the *volume* of rejections means the upstream data is broken and
    a scheduled run should go red instead of quietly succeeding.

    The published data rejects 4 of 149,672 rows (0.003%), so 1% is roughly a
    300x margin -- loose enough not to be noise, tight enough that a structural
    break in a future release trips it.
    """

    max_rejected_rows: int | None = None
    max_rejected_fraction: float = 0.01
    fail_on_unknown_source_type: bool = True


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
    gate_limits: GateLimits = field(default_factory=GateLimits)
    ingest_limits: IngestLimits = field(default_factory=IngestLimits)
    #: Use Databricks Auto Loader for file discovery instead of the portable
    #: hash-log reader. Only affects *new* files -- see `ingest.py`.
    use_autoloader: bool = False
    #: Checkpoint location required by Auto Loader. A volume path on Databricks.
    checkpoint_dir: Path | None = None

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
        """The single newest labels file. Kept for the one-shot exploration path."""
        return _resolve_one(self.raw_dir, LABEL_FILE_CANDIDATES, "labels")

    def resolve_feature_file(self) -> Path:
        return _resolve_one(self.raw_dir, FEATURE_FILE_CANDIDATES, "features")

    def label_files(self) -> list[Path]:
        """Every labels file present, including later batches."""
        return _glob_sources(self.raw_dir, LABEL_FILE_GLOBS, "labels")

    def feature_files(self) -> list[Path]:
        """Every features file present, including later batches."""
        return _glob_sources(self.raw_dir, FEATURE_FILE_GLOBS, "features")

    @classmethod
    def for_fixtures(cls, lakehouse_dir: Path) -> "PipelineConfig":
        """Config pointing at the synthetic fixture, for tests and local demos.

        The gate's volume limits are relaxed here on purpose. The fixture plants
        4 defects in 17 rows so that every rejection reason is exercised -- 23%,
        against a production limit of 1%. Keeping the strict limit would make the
        fixture fail every run, and lowering the *production* limit to accommodate
        a deliberately broken fixture would be exactly the wrong trade.
        """
        return cls(
            raw_dir=REPO_ROOT / "tests" / "fixtures",
            lakehouse_dir=lakehouse_dir,
            gate_limits=GateLimits(max_rejected_fraction=1.0),
        )


def _missing(raw_dir: Path, patterns: tuple[str, ...], kind: str) -> FileNotFoundError:
    return FileNotFoundError(
        f"No {kind} file found in {raw_dir}. Expected something matching {patterns}.\n"
        "The raw files are not committed (CC BY-NC 4.0). Download them from\n"
        "https://doi.org/10.34810/DATA3164 and place them in data/raw/."
    )


def _resolve_one(raw_dir: Path, candidates: tuple[str, ...], kind: str) -> Path:
    for name in candidates:
        path = raw_dir / name
        if path.exists():
            return path
    raise _missing(raw_dir, candidates, kind)


def _glob_sources(raw_dir: Path, patterns: tuple[str, ...], kind: str) -> list[Path]:
    found: list[Path] = []
    for pattern in patterns:
        found.extend(sorted(raw_dir.glob(pattern)))
    unique = sorted({p.resolve() for p in found if p.is_file()})
    if not unique:
        raise _missing(raw_dir, patterns, kind)
    return unique
