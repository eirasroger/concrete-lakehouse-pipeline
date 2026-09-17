"""Compact the Delta tables and expire the files incremental writes orphaned.

A separate entry point from the pipeline on purpose. `OPTIMIZE` is expensive and
pointless after one small batch, and `VACUUM` permanently removes the old
versions that time travel depends on -- neither belongs on the ingest path.

    python scripts/maintain_tables.py --fixtures
    python scripts/maintain_tables.py --namespace workspace.concrete
    python scripts/maintain_tables.py --no-vacuum          # compact only
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

# Running from a clone: put src/ on the path and pin the interpreter Spark hands
# to its workers. Databricks runs a job's python_file through exec(), where
# __file__ does not exist -- there the concrete_pipeline wheel is installed into
# the serverless environment by the bundle, so no path fixing is needed.
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
except NameError:
    pass

from concrete_pipeline import config as cfg
from concrete_pipeline.maintenance import (
    DEFAULT_RETAIN_HOURS,
    print_maintenance_report,
    run_maintenance,
)
from concrete_pipeline.session import get_spark, stop_spark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lakehouse-dir", type=Path, default=None)
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--fixtures", action="store_true")
    parser.add_argument(
        "--retain-hours",
        type=int,
        default=DEFAULT_RETAIN_HOURS,
        help=f"time-travel window to preserve (default {DEFAULT_RETAIN_HOURS}h)",
    )
    parser.add_argument(
        "--no-vacuum",
        action="store_true",
        help="compact only; keep every historical file",
    )
    args = parser.parse_args(argv)

    lakehouse = args.lakehouse_dir or cfg.REPO_ROOT / "data" / "lakehouse"
    config = (
        cfg.PipelineConfig.for_fixtures(lakehouse)
        if args.fixtures
        else cfg.PipelineConfig(lakehouse_dir=lakehouse)
    )
    if args.namespace:
        config = replace(config, namespace=args.namespace)

    spark = get_spark(app_name="concrete-maintenance")
    try:
        results = run_maintenance(
            spark, config, retain_hours=args.retain_hours, vacuum=not args.no_vacuum
        )
        print_maintenance_report(results)
        if not args.no_vacuum:
            print(f"\nTime travel preserved for the last {args.retain_hours}h.")
    finally:
        stop_spark(spark)
    return 0


if __name__ == "__main__":
    # Exit non-zero only on failure. `sys.exit(0)` would raise SystemExit, and
    # Databricks runs a job's python_file through exec() -- there any SystemExit
    # escaping the script is reported as a task failure, even for status 0.
    _status = main()
    if _status:
        sys.exit(_status)
