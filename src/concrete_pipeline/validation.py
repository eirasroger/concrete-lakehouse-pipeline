"""A sanity check on gold_scenarios -- not a model.

If the pipeline wired the labels to the wrong features, the correlation between
`pref` and the headline sustainability variables would be noise or inverted.
So we assert only the *sign* of each correlation, per source_type:

    gwp        negative   lower global warming potential should be preferred
    health     positive   a better health score should be preferred
    circ_orig  positive   more circular origin should be preferred

A failure here means the join or the explode is wrong, not that the science is.
Control scenarios are the sharpest test: each `control_<axis>` family varies one
variable by construction, so its own axis should correlate strongly while the
others may sit near zero.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

#: Variable -> expected sign of its correlation with `pref`.
EXPECTED_DIRECTIONS: dict[str, str] = {
    "gwp": "negative",
    "health": "positive",
    "circ_orig": "positive",
}

#: Correlations weaker than this are reported but not treated as a direction.
NEGLIGIBLE = 0.05

#: Below this many rows a correlation is noise and gets no verdict either way.
#: The five-scenario fixture sits well under it, so running this check against
#: the fixture is a smoke test of the plumbing rather than a claim about signs.
MIN_ROWS_FOR_DIRECTION = 30


@dataclass(frozen=True)
class CorrelationCheck:
    """One (source_type, variable) correlation and its verdict."""

    source_type: str
    variable: str
    rows: int
    correlation: float | None
    expected: str
    min_rows: int = MIN_ROWS_FOR_DIRECTION

    @property
    def observed(self) -> str:
        if self.correlation is None:
            return "undefined"
        if abs(self.correlation) < NEGLIGIBLE:
            return "negligible"
        return "positive" if self.correlation > 0 else "negative"

    @property
    def verdict(self) -> str:
        if self.observed == "undefined":
            return "NO DATA"
        if self.rows < self.min_rows:
            return "TOO FEW"
        if self.observed == "negligible":
            return "FLAT"
        return "OK" if self.observed == self.expected else "WRONG SIGN"


def safe_corr(left: str, right: str):
    """Pearson correlation that yields NULL instead of raising on zero variance.

    Spark 4 enables ANSI SQL mode by default, where the division inside `corr`
    throws DIVIDE_BY_ZERO if either column is constant. That is not a pipeline
    defect -- a control family that holds a variable fixed *should* have no
    correlation with it -- so the aggregate is rebuilt from its definition with
    `try_divide`, which returns NULL for the degenerate case.
    """
    covariance = F.covar_samp(left, right)
    spread = F.stddev_samp(left) * F.stddev_samp(right)
    return F.try_divide(covariance, spread)


def correlations(
    gold: DataFrame, min_rows: int = MIN_ROWS_FOR_DIRECTION
) -> list[CorrelationCheck]:
    """Correlate `pref` against each expected variable, overall and per source_type."""
    aggregations = [F.count(F.lit(1)).alias("rows")] + [
        safe_corr("pref", variable).alias(f"corr_{variable}")
        for variable in EXPECTED_DIRECTIONS
    ]

    by_source = gold.groupBy("source_type").agg(*aggregations).orderBy("source_type").collect()
    overall = gold.agg(*aggregations).collect()

    checks: list[CorrelationCheck] = []
    for row in [*by_source, *overall]:
        source_type = row["source_type"] if "source_type" in row.asDict() else "ALL"
        for variable, expected in EXPECTED_DIRECTIONS.items():
            value = row[f"corr_{variable}"]
            checks.append(
                CorrelationCheck(
                    source_type=source_type,
                    variable=variable,
                    rows=row["rows"],
                    correlation=None if value is None else float(value),
                    expected=expected,
                    min_rows=min_rows,
                )
            )
    return checks


def format_summary(checks: list[CorrelationCheck]) -> str:
    """Render the checks as a fixed-width table."""
    header = (
        f"{'source_type':<20} {'rows':>9} {'variable':<11} "
        f"{'corr(pref, x)':>14} {'expected':<10} {'verdict':<11}"
    )
    lines = [header, "-" * len(header)]
    for check in checks:
        correlation = "n/a" if check.correlation is None else f"{check.correlation:+.4f}"
        lines.append(
            f"{check.source_type:<20} {check.rows:>9,} {check.variable:<11} "
            f"{correlation:>14} {check.expected:<10} {check.verdict:<11}"
        )
    return "\n".join(lines)


def wrong_sign_checks(checks: list[CorrelationCheck]) -> list[CorrelationCheck]:
    """The checks whose correlation points the wrong way -- the ones that matter."""
    return [check for check in checks if check.verdict == "WRONG SIGN"]
