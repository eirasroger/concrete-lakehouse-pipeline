"""Guards against APIs that work locally but fail on Databricks serverless.

Databricks serverless compute runs on Spark Connect, which does not implement
the whole PySpark surface. The failures are not caught at import or at call
time -- `cache()` is lazy, `.rdd` fails only when evaluated -- so they surface
minutes into a job run, in a stack trace that names a thread pool rather than
the offending line.

These are source-level checks, done by parsing the AST rather than grepping, so
that documenting a pitfall in a docstring does not trip the guard against it.
They need no Spark session, so they run in the `not spark` subset in
milliseconds, and they encode two failures that actually happened:

    PERSIST TABLE is not supported on serverless compute
    RDD is not supported on serverless compute
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Code that can run as a Databricks job task or in a notebook. The test suite
#: itself is exempt -- it only ever runs on local or CI Spark, where caching is
#: both supported and worth doing.
PRODUCTION_DIRS = ("src", "scripts", "notebooks")

#: Attribute name -> why it cannot be used.
FORBIDDEN_ATTRIBUTES = {
    "cache": "cache() -- serverless rejects PERSIST TABLE, and it fails lazily",
    "persist": "persist() -- serverless rejects PERSIST TABLE",
    "unpersist": "unpersist() -- pointless without cache, and equally unsupported",
    "rdd": "the RDD API is not available on Spark Connect",
}


def production_sources() -> list[Path]:
    files: list[Path] = []
    for directory in PRODUCTION_DIRS:
        files.extend(sorted((REPO_ROOT / directory).rglob("*.py")))
    return [f for f in files if "__pycache__" not in f.parts]


def attribute_uses(path: Path, name: str) -> list[str]:
    """Lines where `name` is accessed as an attribute, ignoring strings."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        f"{path.relative_to(REPO_ROOT)}:{node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == name
    ]


def test_there_are_sources_to_check():
    """A guard on the guard: a bad glob would make every check below vacuous."""
    assert len(production_sources()) >= 10


def test_every_production_source_parses():
    """Also catches a notebook whose cell markers have been mangled."""
    for path in production_sources():
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@pytest.mark.parametrize("name,reason", sorted(FORBIDDEN_ATTRIBUTES.items()))
def test_no_serverless_incompatible_api(name, reason):
    offenders = [use for path in production_sources() for use in attribute_uses(path, name)]
    assert not offenders, f"{reason}\n" + "\n".join(offenders)


def test_sparkcontext_is_only_touched_behind_the_local_branch():
    """`spark.sparkContext` exists locally but not on Spark Connect.

    `session.py` uses it once, to quiet the local session's logging, and only
    after the Databricks branch has already returned.
    """
    session = REPO_ROOT / "src" / "concrete_pipeline" / "session.py"
    uses = attribute_uses(session, "sparkContext")
    assert len(uses) == 1, uses

    source = session.read_text(encoding="utf-8")
    line = int(uses[0].rsplit(":", 1)[1])
    branch_line = source[: source.index("if on_databricks():")].count("\n") + 1
    assert line > branch_line, "sparkContext is reached before the Databricks branch returns"
