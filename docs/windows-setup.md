# Running locally on Windows

None of this applies on Databricks, Linux or macOS. It's only needed to run the
pipeline on a Windows laptop, and it's all handled in
`src/concrete_pipeline/session.py` except the first item.

## Hadoop native library

Spark on Windows needs Hadoop's native library, or Delta's table reads fail with:

```
UnsatisfiedLinkError: NativeIO$Windows.access0
```

Download `winutils.exe` **and** `hadoop.dll` (e.g. from
[cdarlint/winutils](https://github.com/cdarlint/winutils), `hadoop-3.3.6/bin`),
put both in a `bin\` folder, then set:

```powershell
$env:HADOOP_HOME = "C:\path\to\hadoop"
$env:PATH = "$env:HADOOP_HOME\bin;$env:PATH"
```

Both variables matter. `HADOOP_HOME` is what Hadoop itself checks, and `PATH` is
what ends up on the JVM's `java.library.path`. Setting only one leaves you with
the same error. `get_spark()` prints these instructions if either is missing.

Run from **PowerShell or cmd, not Git Bash** — Git Bash rewrites `PATH` on the
way to the JVM and mangles the entry, so the library silently isn't found.

## Hostname with an underscore

Spark builds internal URLs like `spark://HeartbeatReceiver@<hostname>:<port>`.
Underscores are legal in a Windows machine name but illegal in a URL, so a
machine called e.g. `Some_Laptop` fails at startup with:

```
SparkException: Invalid Spark URL: spark://HeartbeatReceiver@Some_Laptop:54261
```

`session.py` pins `spark.driver.host` to `localhost`, which sidesteps the machine
name entirely. Local mode never needs it.

## ANSI SQL and correlations

Spark 4 runs in ANSI SQL mode by default, where `corr()` on a column with no
variance raises `DIVIDE_BY_ZERO` instead of returning null. That happens
legitimately here: a control family that holds a variable fixed genuinely has no
correlation with it.

`validation.py` rebuilds the correlation from `covar_samp` and `stddev_samp`
using `try_divide`, which returns null for that case instead of throwing.

## "idWithoutTopologyInfo is null"

A run stalls, then the log fills with:

```
ERROR Inbox: Ignoring error
java.lang.NullPointerException: Cannot invoke
  "org.apache.spark.storage.BlockManagerId.executorId()"
  because "idWithoutTopologyInfo" is null
WARN Executor: Issue communicating with driver in heartbeater
```

This is the driver's heartbeat failing, and the message points nowhere near the
cause. Three things contribute, and the pipeline addresses all three:

1. **Host/bind mismatch.** `spark.driver.host` and `spark.driver.bindAddress`
   must agree. Setting one to `localhost` and the other to `127.0.0.1` can
   resolve to IPv6 and IPv4 respectively, so nothing can reach the driver.
   `session.py` pins both to the literal `127.0.0.1` and sets `SPARK_LOCAL_IP`.
2. **Too many local threads.** `local[*]` registers one BlockManager per core,
   and concurrent registrations race in Spark's `BlockManagerId` cache. The
   default is now `local[2]`, overridable via `get_spark(local_threads=...)`.
3. **Heap exhaustion.** This is the one that actually bites under load. The
   driver does not raise `OutOfMemoryError` — it stalls in garbage collection,
   misses heartbeats, and dies with the NPE above. Spark's default is 1 GB,
   which is not enough once `MERGE` and cached tables are involved.

If you hit it anyway, raise the heap before the JVM starts:

```powershell
$env:PYSPARK_SUBMIT_ARGS = "--driver-memory 4g pyspark-shell"
```

The test suite sets this itself in `tests/conftest.py`.

## Driver memory

Both raw files are a single pretty-printed JSON array, so Spark must read each
with `multiLine`, and one task holds a whole file. Before a real run:

```powershell
$env:PYSPARK_SUBMIT_ARGS = "--driver-memory 6g pyspark-shell"
```

The 5-scenario fixture needs none of this.
