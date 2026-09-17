# Synthetic fixture

Five scenarios that mirror the schema of the real CORA.RDR files exactly, so the
test suite never touches the 140 MB of raw data. **No row here is real data** —
values are invented, but every field, type and nesting level matches the source.

The fixture covers all three id families and plants the defects the quality gate
is meant to catch.

| scenario | family | products | labels | planted defect |
|---|---|---|---|---|
| `control_health_1068` | control_synthetic | 3 | 3 | — |
| `control_archfinish_slump_299` | control_synthetic | 2 | 2 | — (trips the naive classifier's axis allow-list) |
| `3782` | llm_generated | 4 | 4 | `prod_4` has `density` 950 and `water_to_cement_ratio` 1.35 → two range reasons on one row |
| `expert_118` | expert_annotated | 2 | 2 | — (trips the naive classifier's missing `expert_` rule) |
| `2647` | llm_generated | 5 | 6 | `prod_1` labelled twice with conflicting prefs (0.62/0.05), mirroring the real scenarios `2647` and `13122`; `prod_5` has `pref` 1.4 |

Totals: 16 feature rows, 17 label rows, 17 joined rows.
The gate rejects 4 and promotes 13.

Two scenarios also exercise the multi-value scenario attributes: `3782` carries
two `stakeholder_preference` entries and two `situations`, matching the real
data where those arrays reach length 3 and 2 respectively.

## The incremental fixtures

The pipeline has to handle both ways new data can arrive, so there is a fixture
for each. Tests copy the base files into a temporary directory, run, then add or
replace files and run again.

### `batch2/` — a new file arrives alongside the old

`labelled_alternatives_batch2.json` and `frozen_dataset_batch2.json` add two
scenarios that are not in the base fixture:

| scenario | family | products | labels |
|---|---|---|---|
| `control_gwp_777` | control_synthetic | 2 | 2 |
| `expert_9` | expert_annotated | 3 | 3 |

Both are clean, so the second run should promote 5 more rows and leave the
original 13 untouched. That "untouched" part is the assertion that matters — it
proves the run was incremental rather than a silent full rebuild.

### `revised/` — the same file replaced with corrected content

`labelled_alternatives.json`, **same filename** as the base fixture but different
content, so only the content hash distinguishes it. Three changes, each testing
a different merge clause:

| scenario | change | tests |
|---|---|---|
| `control_health_1068` | `prod_1` pref 0.393 → 0.5 | `WHEN MATCHED THEN UPDATE` |
| `2647` | duplicate `prod_1` label removed, 6 labels → 5 | `WHEN NOT MATCHED BY SOURCE THEN DELETE` |
| `2647` | `prod_5` pref 1.4 → 0.97 | a previously rejected row now promotes |

The `2647` case is the important one. Its label count shrinks, so a pipeline
without the delete clause would keep the orphaned sixth row forever — and
because the duplicate is gone, `2647/prod_1` should move *out* of
`gold_scenarios_rejected` and into `gold_scenarios`. After the revision the
fixture should reject only 1 row instead of 4.
