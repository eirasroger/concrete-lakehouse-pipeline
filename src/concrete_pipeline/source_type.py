"""Classification of a scenario id into its provenance (`source_type`).

The taxonomy below was not assumed -- it was derived by running
`scripts/explore_raw.py` over the published files. What it found, across all
42,874 scenarios:

    prefix        count   shape                      evidence
    control_     24,000   control_<axis>_<n>         8 axes x exactly 3,000
    <numeric>    18,602   <n>, contiguous 1..18602   free-text reasons, conf 0.40-1.00
    expert_         272   expert_<n>, contiguous     9 canned reasons, conf 0.50-0.95

The eight control axes are health, gwp, wdp, fwu, density, cost, archfinish and
archfinish_slump. `control_archfinish` (driven by circ_orig) and
`control_archfinish_slump` (driven by slump) are *separate* probes, not one axis
with a suffix -- their reason templates differ.

One trap is worth stating explicitly: `expert_1..272` and the bare ids `1..272`
number from 1 independently and are entirely different scenarios. Across all
272 collisions, zero share the same `alternatives`, and stakeholder/situation
differ for most. Any rule that strips or ignores the prefix silently merges two
unrelated populations.
"""

from __future__ import annotations

import re
from typing import Callable

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

#: Provenance values emitted by the current classifier.
CONTROL_SYNTHETIC = "control_synthetic"
LLM_GENERATED = "llm_generated"
EXPERT_ANNOTATED = "expert_annotated"
UNKNOWN = "unknown"

SOURCE_TYPES: tuple[str, ...] = (CONTROL_SYNTHETIC, LLM_GENERATED, EXPERT_ANNOTATED, UNKNOWN)

#: The eight control axes observed in the published data. Kept for validation
#: and documentation; the classifier itself does not require an axis allow-list.
CONTROL_AXES: tuple[str, ...] = (
    "archfinish",
    "archfinish_slump",
    "cost",
    "density",
    "fwu",
    "gwp",
    "health",
    "wdp",
)

# Any number of underscore-separated axis tokens, then a trailing integer.
_CONTROL_RE = re.compile(r"^control_(?P<axis>[a-z][a-z_]*[a-z])_(?P<n>\d+)$")
_EXPERT_RE = re.compile(r"^expert_(?P<n>\d+)$")
_NUMERIC_RE = re.compile(r"^\d+$")

# Spark-side equivalents of the patterns above.
_CONTROL_SQL = r"^control_[a-z][a-z_]*[a-z]_\d+$"
_EXPERT_SQL = r"^expert_\d+$"
_NUMERIC_SQL = r"^\d+$"


def classify_source_type(scenario_id: str | None) -> str:
    """Classify a single scenario id. The reference implementation.

    >>> classify_source_type("control_archfinish_slump_299")
    'control_synthetic'
    >>> classify_source_type("expert_118")
    'expert_annotated'
    >>> classify_source_type("3782")
    'llm_generated'
    >>> classify_source_type("something_else")
    'unknown'
    """
    if scenario_id is None:
        return UNKNOWN
    candidate = scenario_id.strip()
    if _CONTROL_RE.match(candidate):
        return CONTROL_SYNTHETIC
    if _EXPERT_RE.match(candidate):
        return EXPERT_ANNOTATED
    if _NUMERIC_RE.match(candidate):
        return LLM_GENERATED
    return UNKNOWN


def control_axis(scenario_id: str | None) -> str | None:
    """Return the probed axis for a control id, else None.

    Not part of the gold schema -- exposed for profiling and for the tests that
    assert `archfinish_slump` is not truncated to `archfinish`.
    """
    if scenario_id is None:
        return None
    match = _CONTROL_RE.match(scenario_id.strip())
    return match.group("axis") if match else None


def source_type_column(id_column: str = "scenario_id") -> Column:
    """The classifier as a native Spark expression (no Python UDF round-trip)."""
    trimmed = F.trim(F.col(id_column))
    return (
        F.when(trimmed.rlike(_CONTROL_SQL), F.lit(CONTROL_SYNTHETIC))
        .when(trimmed.rlike(_EXPERT_SQL), F.lit(EXPERT_ANNOTATED))
        .when(trimmed.rlike(_NUMERIC_SQL), F.lit(LLM_GENERATED))
        .otherwise(F.lit(UNKNOWN))
    )


def naive_source_type_column(id_column: str = "scenario_id") -> Column:
    """The *deliberately defective* first-pass classifier, kept for the Delta demo.

    It carries the two mistakes that are easiest to make when the taxonomy is
    guessed from a handful of sample ids rather than profiled:

    1. **No `expert_` rule.** Everything that is not a control id falls through
       to `llm_generated`, so all 272 expert scenarios are mislabelled.
    2. **A closed axis allow-list that omits `archfinish_slump`.** Written by
       eyeballing ids and splitting on ``_``, it recognises only the seven
       single-token axes. `control_archfinish_slump_299` therefore fails the
       control pattern too and also lands in the `llm_generated` fallback.

    `scripts/delta_versioning_demo.py` writes gold with this rule, then
    overwrites with :func:`source_type_column` and diffs the two Delta versions.
    """
    single_token_axes = "|".join(axis for axis in CONTROL_AXES if "_" not in axis)
    trimmed = F.trim(F.col(id_column))
    return F.when(
        trimmed.rlike(rf"^control_({single_token_axes})_\d+$"), F.lit(CONTROL_SYNTHETIC)
    ).otherwise(F.lit(LLM_GENERATED))


#: A classifier is any callable taking the id column name and returning a Column.
Classifier = Callable[[str], Column]

CLASSIFIERS: dict[str, Classifier] = {
    "naive": naive_source_type_column,
    "corrected": source_type_column,
}


def with_source_type(
    df: DataFrame,
    id_column: str = "scenario_id",
    classifier: Classifier = source_type_column,
) -> DataFrame:
    """Attach a `source_type` column derived from `id_column`."""
    return df.withColumn("source_type", classifier(id_column))
