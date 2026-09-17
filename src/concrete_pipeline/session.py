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

#: Set when `get_spark` builds a local session itself, so `stop_spark` knows
#: whether the session is ours to shut down.
_OWNS_SESSION = False


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


def on_databricks() -> bool:
    """True when running on Databricks compute (a notebook or a job task)."""
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def get_spark(
    app_name: str = "concrete-data-pipeline",
    shuffle_partitions: int = 8,
    extra_conf: dict[str, str] | None = None,
    local_threads: int = 2,
) -> SparkSession:
    """Return an active Spark session with Delta Lake enabled.

    Reuses the ambient session when one exists (Databricks, notebooks, pytest
    fixtures) rather than fighting it for configuration. `extra_conf` is applied
    last, so callers can override anything set here -- the test suite uses it to
    strip the machinery that only pays off on data larger than a fixture.
    """
    global _OWNS_SESSION

    active = SparkSession.getActiveSession()
    if active is not None:
        return active

    if on_databricks():
        # A serverless job task has no *active* session until something asks for
        # one, but it must come from the platform. Falling through to the local
        # builder below would try to start a local[*] Spark inside the job and
        # write Delta to the driver's disk instead of Unity Catalog.
        return SparkSession.builder.getOrCreate()

    _warn_if_windows_native_missing()

    # Set before the JVM starts: SPARK_LOCAL_IP is the lowest-level knob for the
    # address Spark binds and advertises, and it keeps the driver host, the bind
    # address and the block manager's own registration consistent. Setting only
    # the spark.driver.* configs leaves the BlockManager free to register under a
    # different name, which surfaces as a heartbeat failing forever with
    # "NullPointerException ... idWithoutTopologyInfo is null".
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

    builder = (
        SparkSession.builder.appName(app_name)
        # Not local[*]. Every executor thread registers its own BlockManager, and
        # concurrent registrations race in Spark's BlockManagerId cache -- on a
        # many-core machine that intermittently wedges the driver's heartbeat.
        # Two threads is plenty for data this size and makes runs reproducible.
        .master(f"local[{local_threads}]")
        .config("spark.sql.extensions", DELTA_EXTENSION)
        .config("spark.sql.catalog.spark_catalog", DELTA_CATALOG)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.databricks.delta.snapshotPartitions", "2")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.session.timeZone", "UTC")
        # Both of these are a literal 127.0.0.1, deliberately, and must match.
        #
        # Spark builds internal URLs as spark://Component@<host>:<port>. A
        # machine name containing an underscore is legal on Windows but illegal
        # in a URI authority, so the default makes SparkContext fail outright
        # with "Invalid Spark URL". Pinning the driver host avoids the name.
        #
        # It has to be the IP and not "localhost": if the host resolves to IPv6
        # ::1 while the driver binds to IPv4, the executor cannot reach the
        # driver and block manager registration dies with a confusing
        # "NullPointerException ... idWithoutTopologyInfo is null". Using the
        # same literal address for both removes the resolution step entirely.
        .config("spark.driver.host", "127.0.0.1")
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
    _OWNS_SESSION = True
    return spark


def stop_spark(spark: SparkSession) -> None:
    """Stop the session only if this process was the one that started it.

    A script run as a Databricks job task shares the platform's session. Calling
    `spark.stop()` on that would tear down the runtime's own session, so the
    entry points call this instead of stopping unconditionally.
    """
    if _OWNS_SESSION:
        spark.stop()
