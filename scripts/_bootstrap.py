"""Make `src/` importable and pin the interpreter Spark hands to its workers.

Imported for its side effects by the scripts in this directory, so a fresh
clone runs with `python scripts/<name>.py` and no `pip install -e .` first.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Without these Spark launches workers with whatever `python` is on PATH, which
# is often not the virtualenv running the driver.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
