"""Run the whole pipeline: raw JSON -> bronze -> silver -> gold.

    python scripts/run_pipeline.py                  # real files in data/raw
    python scripts/run_pipeline.py --fixtures       # the 5-scenario fixture
    python scripts/run_pipeline.py --lakehouse-dir /tmp/lake
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401  (adds src/ to sys.path)

from concrete_pipeline import config as cfg
from concrete_pipeline.pipeline import print_run_report, run_pipeline
from concrete_pipeline.session import get_spark


def build_config(args: argparse.Namespace) -> cfg.PipelineConfig:
    lakehouse = args.lakehouse_dir or cfg.REPO_ROOT / "data" / "lakehouse"
    if args.fixtures:
        config = cfg.PipelineConfig.for_fixtures(lakehouse)
    else:
        config = cfg.PipelineConfig(raw_dir=args.raw_dir, lakehouse_dir=lakehouse)
    if args.namespace:
        config = replace(config, namespace=args.namespace)
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=cfg.REPO_ROOT / "data" / "raw")
    parser.add_argument("--lakehouse-dir", type=Path, default=None)
    parser.add_argument(
        "--namespace",
        default=None,
        help="Unity Catalog catalog.schema to write to instead of a local folder",
    )
    parser.add_argument(
        "--fixtures",
        action="store_true",
        help="run against tests/fixtures instead of the real files",
    )
    parser.add_argument(
        "--infer-schema",
        action="store_true",
        help="let Spark infer the JSON schema instead of using the pinned one",
    )
    args = parser.parse_args(argv)

    config = build_config(args)
    spark = get_spark()
    try:
        result = run_pipeline(spark, config, infer_schema=args.infer_schema)
        print_run_report(result)
        target = config.namespace if config.uses_catalog else config.lakehouse_dir
        print(f"Tables written to {target}")
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
