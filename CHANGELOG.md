# Changelog

## 0.1.0 (2026-10-06)

First release.

- `evaluate()` over labeled cases with a Pydantic schema: validity rate, overall and per-field accuracy, wildcard paths for lists.
- Comparators: `exact`, `numeric`, `fuzzy`, `contains`, `one_of`.
- `Report.save/load/compare` and a `schemascore run` CLI that fails CI on regressions or a minimum-accuracy floor.
