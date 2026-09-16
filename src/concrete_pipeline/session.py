"""Spark session construction for local Delta Lake runs.

On Databricks the session already exists and Delta is the default format, so
`get_spark()` returns the active session untouched. Locally it configures the
Delta extensions and a small single-machine shuffle profile.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

from pyspark.sql import SparkSession

DELTA_EXTENSION = "io.delta.sql.DeltaSparkSessionExtension"
DELTA_CATALOG = "org.apache.spark.sql.delta.catalog.DeltaCatalog"


# Raw: the Windows paths below contain \b, which is a backspace in a normal string.
WINDOWS_NATIVE_HELP = r"""Hadoop's native library was not found.

On Windows, Delta's table reads go through Hadoop's JNI file-access checks, and
without hadoop.dll they fail with:

    UnsatisfiedLinkError: NativeIO$Windows.access0

To fix, download winutils.exe and hadoop.dll (e.g. from
https://github.com/cdarlint/winutils, hadoop-3.3.6/bin), put both in
<dir>\bin, then set:

    HADOOP_HOME=<dir>
    PATH=%HADOOP_HOME%\bin;%PATH%

Both are required: HADOOP_HOME is what Hadoop checks, and it is PATH that ends
up on the JVM's java.library.path. Nothing is needed on Linux or macOS."""


def _warn_if_windows_native_missing() -> None:
    """Warn early and specifically when the Windows native setup is incomplete.

    Without this the first symptom is an UnsatisfiedLinkError thrown from inside
    Delta, several layers below anything the caller wrote.
    """
    if os.name != "nt":
        return

    hadoop_home = os.environ.get("HADOOP_HOME")
    on_library_path = any(
        (Path(entry) / "hadoop.dll").exists()
        for entry in os.environ.get("PATH", "").split(os.pathsep)
        if entry
    )
    if hadoop_home and on_library_path:
        return

    warnings.warn(WINDOWS_NATIVE_HELP, RuntimeWarning, stacklevel=3)


def get_spark(
    app_name: str = "concrete-data-pipeline",
    shuffle_partitions: int = 8,
    extra_conf: dict[str, str] | None = None,
) -> SparkSession:
    """Return an active Spark session with Delta Lake enabled.

    Reuses the ambient session when one exists (Databricks, notebooks, pytest
    fixtures) rather than fighting it for configuration. `extra_conf` is applied
    last, so callers can override anything set here -- the test suite uses it to
    strip the machinery that only pays off on data larger than a fixture.
    """
    active = SparkSession.getActiveSession()
    if active is not None:
        return active

    _warn_if_windows_native_missing()

    builder = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.extensions", DELTA_EXTENSION)
        .config("spark.sql.catalog.spark_catalog", DELTA_CATALOG)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.databricks.delta.snapshotPartitions", "2")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.session.timeZone", "UTC")
        # Spark builds internal URLs as spark://Component@<hostname>:<port>. A
        # hostname containing an underscore -- legal in Windows machine names,
        # illegal in a URI authority -- makes SparkContext fail to start with
        # "Invalid Spark URL". Pinning the driver host sidesteps the machine
        # name entirely, which local mode never needs anyway.
        .config("spark.driver.host", "localhost")
        .config("spark.driver.bindAddress", "127.0.0.1")
    )

    for key, value in (extra_conf or {}).items():
        builder = builder.config(key, value)

    try:  # delta-spark ships a helper that pins the matching delta-core jars
        from delta import configure_spark_with_delta_pip

        builder = configure_spark_with_delta_pip(builder)
    except ImportError:  # pragma: no cover - only when delta-spark is absent
        pass

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark
