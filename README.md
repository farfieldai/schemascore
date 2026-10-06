# schemascore

[![CI](https://github.com/farfieldai/schemascore/actions/workflows/ci.yml/badge.svg)](https://github.com/farfieldai/schemascore/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/schemascore)](https://pypi.org/project/schemascore/)

Field-level evals for structured LLM output.

"The extraction looks fine" isn't a metric. When you change a prompt or swap a
model, you need to know which fields got worse. `schemascore` runs your
extraction over labeled cases, validates every output against a Pydantic
schema, and reports validity plus accuracy per field. Save the report, compare
the next run against it, and fail CI when something regresses.

```bash
pip install schemascore
```

Python 3.10+. One dependency: Pydantic 2.

## Quickstart

```python
from pydantic import BaseModel
from schemascore import evaluate, fuzzy, numeric, Report

class Invoice(BaseModel):
    vendor: str
    total: float
    currency: str

cases = [
    {"id": "acme", "input": "INVOICE ... Acme Corp ... USD 1,250.00",
     "expected": {"vendor": "Acme Corp", "total": 1250.0, "currency": "USD"}},
    # ...
]

report = evaluate(
    Invoice,
    cases,
    extract,  # your function: input -> JSON string, dict, or Invoice
    comparators={"total": numeric(abs_tol=0.01), "vendor": fuzzy(0.85)},
    concurrency=8,
)
print(report.summary())
report.save("baseline.json")

# Later, after changing the prompt or model:
diff = report.compare(Report.load("baseline.json"), max_drop=0.02)
diff.ok           # False if any metric dropped more than 2 points
diff.regressions  # {"total": -0.25, "<overall>": -0.08}
```

`extract` can return a JSON string (```` ```json ```` fences are fine), a
dict, or a model instance, and can be sync or async. Sync targets run on
threads and async ones as asyncio tasks, `concurrency` at a time.

## How scoring works

- Expected and actual outputs are flattened to paths like `items[0].price`.
- **Only fields present in `expected` are scored**, so partial labels work.
- Accuracy is reported per field, with list indexes folded together:
  `items[*].price`.
- A comparator is chosen by exact path, then wildcard path, then the default,
  `exact()`.
- If the output isn't valid JSON, fails the schema, or the target raises, the
  case is **invalid** and every labeled field counts as a miss. The eval never
  crashes because of the thing it's evaluating.
- **overall** = correct fields ÷ labeled fields, across all cases.
  **validity** = valid cases ÷ cases.

## Comparators

A comparator is any `(expected, actual) -> bool`.

| Comparator | Matches when |
| --- | --- |
| `exact(normalize=True)` | equal; strings ignore case and extra whitespace, floats ignore rounding noise |
| `numeric(abs_tol=0, rel_tol=0)` | numbers within tolerance; numeric strings like `"1,250.00"` are converted |
| `fuzzy(threshold=0.85)` | text similarity (difflib ratio) is at least `threshold` |
| `contains()` | the expected text appears in the actual text |
| `one_of([...])` | both values are in the same group of synonyms, e.g. `["USD", "US$", "$"]` |

## In CI

```bash
schemascore run \
  --schema examples.invoice:Invoice \
  --target examples.invoice:extract \
  --cases examples/cases.jsonl \
  --comparators examples.invoice:COMPARATORS \
  --baseline baseline.json --max-drop 0.02 --min-overall 0.9 \
  --out latest.json
```

Exit code 1 if any metric (validity, overall, or a field) dropped more than
`--max-drop` against the baseline, or overall is below `--min-overall`.
Commit a baseline, and update it deliberately when a change is an improvement.

The bundled example scores a fake extractor with realistic mistakes over four
invoices, one of which comes back as truncated JSON:

```
cases     4
valid      75.0%
overall    70.8%
```

## Limits

- **List items are matched by position**, not by content. If a model returns
  the right items in a different order, they score as misses.
- **One call per case.** Model output varies between runs; before trusting a
  small difference, run more than once or use more cases.

## License

MIT
