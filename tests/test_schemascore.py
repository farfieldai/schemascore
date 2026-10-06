import asyncio
import json
import threading
from pathlib import Path

import pytest
from pydantic import BaseModel

from schemascore import (
    Case,
    Report,
    contains,
    evaluate,
    exact,
    fuzzy,
    load_cases,
    numeric,
    one_of,
)
from schemascore.cli import main

ROOT = Path(__file__).resolve().parents[1]


class Item(BaseModel):
    name: str
    price: float


class Order(BaseModel):
    customer: str
    total: float
    status: str
    items: list[Item] = []


GOOD = {"customer": "Ada", "total": 30.0, "status": "paid",
        "items": [{"name": "pen", "price": 10.0}, {"name": "pad", "price": 20.0}]}


def echo(x):
    return x


def test_perfect_score():
    report = evaluate(Order, [Case("a", GOOD, GOOD)], echo)
    assert report.validity == 1.0
    assert report.overall == 1.0
    assert report.fields["items[*].price"] == 1.0


def test_field_level_misses():
    out = {**GOOD, "status": "pending", "total": 31.0}
    report = evaluate(Order, [Case("a", out, GOOD)], echo)
    result = report.cases[0]
    assert result.valid
    assert result.fields["status"] is False and result.fields["total"] is False
    assert result.fields["customer"] is True
    assert report.overall == pytest.approx(5 / 7)


def test_invalid_json_scores_every_labeled_field_as_a_miss():
    report = evaluate(Order, [Case("a", '{"customer": "Ada", "tot', GOOD)], echo)
    result = report.cases[0]
    assert not result.valid
    assert "invalid JSON" in result.error
    assert result.fields and not any(result.fields.values())
    assert report.overall == 0.0


def test_schema_violation_is_invalid():
    report = evaluate(Order, [Case("a", {"customer": "Ada", "status": "paid"}, GOOD)], echo)
    assert not report.cases[0].valid
    assert "schema violation at total" in report.cases[0].error


def test_code_fences_and_model_instances_are_accepted():
    fenced = "```json\n" + json.dumps(GOOD) + "\n```"
    report = evaluate(Order, [Case("fenced", fenced, GOOD), Case("model", Order(**GOOD), GOOD)], echo)
    assert report.validity == 1.0 and report.overall == 1.0


def test_partial_labels_only_score_labeled_fields():
    out = {**GOOD, "status": "WRONG"}
    report = evaluate(Order, [Case("a", out, {"customer": "Ada"})], echo)
    assert report.cases[0].fields == {"customer": True}
    assert report.overall == 1.0


def test_missing_list_items_are_misses():
    out = {**GOOD, "items": GOOD["items"][:1]}
    result = evaluate(Order, [Case("a", out, GOOD)], echo).cases[0]
    assert result.fields["items[0].price"] is True
    assert result.fields["items[1].name"] is False and result.fields["items[1].price"] is False


def test_comparators_by_exact_and_wildcard_path():
    out = {**GOOD, "total": 30.004, "items": [{"name": "pen", "price": 10.004}, {"name": "pad", "price": 20.3}]}
    comps = {"total": numeric(abs_tol=0.01), "items[*].price": numeric(abs_tol=0.01)}
    result = evaluate(Order, [Case("a", out, GOOD)], echo, comps).cases[0]
    assert result.fields["total"] is True
    assert result.fields["items[0].price"] is True
    assert result.fields["items[1].price"] is False
    # Without comparators the default is exact().
    assert evaluate(Order, [Case("a", out, GOOD)], echo).cases[0].fields["total"] is False


def test_comparator_units():
    assert exact()("  Hello   World", "hello world")
    assert not exact(normalize=False)("Hello", "hello")
    assert exact()(0.1 + 0.2, 0.3)
    assert numeric(abs_tol=0.01)("1,250.00", 1250.004)
    assert not numeric()("abc", 1.0)
    assert fuzzy(0.85)("Globex Corporation", "Globex Corporaton")
    assert not fuzzy(0.85)("Globex", "Initech")
    assert contains()("acme", "ACME Corp, Inc.")
    assert one_of(["USD", "US$", "$"])("USD", "us$")
    assert not one_of(["USD", "US$"])("USD", "EUR")


def test_target_exceptions_are_caught():
    def boom(_):
        raise RuntimeError("model timeout")

    report = evaluate(Order, [Case("a", None, GOOD), Case("b", GOOD, GOOD)], lambda x: boom(x) if x is None else x)
    assert not report.cases[0].valid
    assert "RuntimeError: model timeout" in report.cases[0].error
    assert report.cases[1].valid
    assert report.validity == 0.5


def test_threaded_and_async_targets():
    # Four threads must reach the barrier together, or it times out and the
    # cases fail: this passes only if cases really run four at a time.
    barrier = threading.Barrier(4, timeout=5)

    def sync_target(x):
        barrier.wait()
        return x

    in_flight, peak = 0, 0

    async def async_target(x):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return x

    cases = [Case(str(i), GOOD, GOOD) for i in range(8)]
    assert evaluate(Order, cases, sync_target, concurrency=4).overall == 1.0
    assert evaluate(Order, cases, async_target, concurrency=4).overall == 1.0
    assert peak == 4  # concurrency caps async tasks too


def test_dict_cases_get_auto_ids():
    report = evaluate(Order, [{"input": GOOD, "expected": GOOD}, {"id": "named", "input": GOOD, "expected": GOOD}], echo)
    assert [c.id for c in report.cases] == ["case-1", "named"]


def test_save_load_and_compare(tmp_path):
    baseline = evaluate(Order, [Case("a", GOOD, GOOD)], echo)
    baseline.save(tmp_path / "baseline.json")
    loaded = Report.load(tmp_path / "baseline.json")
    assert loaded.metrics() == baseline.metrics()

    worse = evaluate(Order, [Case("a", {**GOOD, "status": "void"}, GOOD)], echo)
    diff = worse.compare(loaded, max_drop=0.02)
    assert not diff.ok
    assert diff.regressions["status"] == -1.0
    assert diff.regressions["<overall>"] == pytest.approx(-1 / 7, abs=1e-6)
    assert worse.compare(loaded, max_drop=1.0).ok


def test_load_cases_jsonl_and_json(tmp_path):
    rows = [{"id": "x", "input": "a", "expected": {"customer": "Ada"}}, {"input": "b", "expected": {"total": 1}}]
    (tmp_path / "c.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    (tmp_path / "c.json").write_text(json.dumps(rows))
    for name in ("c.jsonl", "c.json"):
        cases = load_cases(tmp_path / name)
        assert [c.id for c in cases] == ["x", "case-2"]
        assert cases[1].expected == {"total": 1}


def test_cli_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    common = ["run", "--schema", "examples.invoice:Invoice", "--target", "examples.invoice:extract",
              "--cases", "examples/cases.jsonl", "--comparators", "examples.invoice:COMPARATORS"]
    out = tmp_path / "baseline.json"

    assert main([*common, "--out", str(out)]) == 0
    assert "valid      75.0%" in capsys.readouterr().out

    # Same run vs itself: no regressions.
    assert main([*common, "--baseline", str(out), "--max-drop", "0.02"]) == 0

    # Gate 1: a baseline that was better than today.
    better = json.loads(out.read_text())
    for case in better["cases"]:
        case["valid"], case["fields"] = True, dict.fromkeys(case["fields"], True)
    (tmp_path / "better.json").write_text(json.dumps(better))
    assert main([*common, "--baseline", str(tmp_path / "better.json"), "--max-drop", "0.02"]) == 1
    assert "regressions vs" in capsys.readouterr().out

    # Gate 2: a minimum overall accuracy we don't reach.
    assert main([*common, "--min-overall", "0.99"]) == 1
    assert "below the minimum" in capsys.readouterr().out
