"""A bronze/silver/gold lakehouse for the concrete recommendation dataset.

Layer modules (`bronze`, `silver`, `gold`) hold the transformations, `quality`
holds the gate, `source_type` holds the provenance classifier, and `pipeline`
wires them together. Nothing in here depends on Databricks -- the same code runs
against a local Delta directory or a Unity Catalog volume.
"""

from __future__ import annotations

from .config import PipelineConfig, QualityThresholds
from .pipeline import PipelineResult, print_run_report, run_pipeline
from .source_type import (
    CONTROL_SYNTHETIC,
    EXPERT_ANNOTATED,
    LLM_GENERATED,
    UNKNOWN,
    classify_source_type,
)

__all__ = [
    "PipelineConfig",
    "QualityThresholds",
    "PipelineResult",
    "run_pipeline",
    "print_run_report",
    "classify_source_type",
    "CONTROL_SYNTHETIC",
    "LLM_GENERATED",
    "EXPERT_ANNOTATED",
    "UNKNOWN",
]

__version__ = "0.1.0"
