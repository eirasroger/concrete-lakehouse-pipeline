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
