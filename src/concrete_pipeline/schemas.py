"""Explicit Spark schemas for the two raw JSON files.

Schema inference works on these files, but it costs a full extra pass over
100 MB+ of JSON and is not stable across inputs: `cement_content` and `health`
are whole numbers in the published data and infer as `bigint`, yet nothing in
the source format guarantees a future release will not emit `374.0`. Pinning
the schema keeps bronze byte-for-byte reproducible. Inference remains available
via `bronze.load_*(infer_schema=True)` for exploration.
"""

from __future__ import annotations

from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    StringType,
    StructField,
    StructType,
)

#: Cost sub-struct: production, waste and maintenance cost components.
COST_STRUCT = StructType(
    [
        StructField("c_p", DoubleType(), True),
        StructField("c_w", DoubleType(), True),
        StructField("c_m", DoubleType(), True),
    ]
)

#: One product alternative in `frozen_dataset.json`.
ALTERNATIVE_STRUCT = StructType(
    [
        StructField("id_prod", StringType(), True),
        StructField("circ_orig", DoubleType(), True),
        StructField("fu_recyc", DoubleType(), True),
        StructField("fu_incin", DoubleType(), True),
        StructField("fu_inert", DoubleType(), True),
        StructField("fu_haz", DoubleType(), True),
        StructField("health", DoubleType(), True),
        StructField("gwp", DoubleType(), True),
        StructField("wdp", DoubleType(), True),
        StructField("fwu", DoubleType(), True),
        StructField("b", DoubleType(), True),
        StructField("c", COST_STRUCT, True),
        StructField("compressive_strength", DoubleType(), True),
        StructField("slump", DoubleType(), True),
        StructField("water_to_cement_ratio", DoubleType(), True),
        StructField("cement_content", DoubleType(), True),
        StructField("SCM_content", DoubleType(), True),
        StructField("density", DoubleType(), True),
        StructField("d_max", DoubleType(), True),
    ]
)

#: One scenario in `frozen_dataset.json`.
FEATURES_SCHEMA = StructType(
    [
        StructField("id", StringType(), True),
        StructField("stakeholder_preference", ArrayType(StringType()), True),
        StructField("situations", ArrayType(StringType()), True),
        StructField("alternatives", ArrayType(ALTERNATIVE_STRUCT), True),
    ]
)

#: One human/model label in `labelled_alternatives.json`.
LABELLED_ALTERNATIVE_STRUCT = StructType(
    [
        StructField("id_prod", StringType(), True),
        StructField("pref", DoubleType(), True),
        StructField("conf", DoubleType(), True),
        StructField("reason", StringType(), True),
    ]
)

#: One scenario in `labelled_alternatives.json`.
LABELS_SCHEMA = StructType(
    [
        StructField("id", StringType(), True),
        StructField("labelled_alternatives", ArrayType(LABELLED_ALTERNATIVE_STRUCT), True),
    ]
)

#: Feature columns carried from the nested `alternatives` struct into silver,
#: in source order. `c` is excluded -- it is flattened into c_p/c_w/c_m.
FEATURE_COLUMNS: tuple[str, ...] = tuple(
    f.name for f in ALTERNATIVE_STRUCT.fields if f.name not in {"id_prod", "c"}
)

#: The flattened cost columns, in source order.
COST_COLUMNS: tuple[str, ...] = tuple(f.name for f in COST_STRUCT.fields)
