"""The evaluation engine: run a target over labeled cases, validate each output
against a Pydantic schema, and score every labeled field."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from .compare import Comparator, exact

OVERALL = "<overall>"
VALIDITY = "<validity>"

_INDEX = re.compile(r"\[\d+\]")
_FENCE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class Case:
    """One labeled example: what goes in, and the fields you expect out."""

    id: str
    input: Any
    expected: dict[str, Any]


@dataclass
class CaseResult:
    id: str
    valid: bool
    """The output parsed and passed the schema."""
    error: str | None
    """Why the output was invalid, or what the target raised."""
    fields: dict[str, bool]
    """Every labeled field path (``items[0].price``) and whether it matched."""


@dataclass
class Diff:
    """How a report compares with a baseline."""

    ok: bool
    """No metric dropped by more than ``max_drop``."""
    regressions: dict[str, float]
    """Metrics that dropped too far, with their change (negative)."""
    deltas: dict[str, float]
    """Every metric present in both reports, with its change."""


@dataclass
class Report:
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def validity(self) -> float:
        """Share of cases whose output was valid."""
        return sum(c.valid for c in self.cases) / len(self.cases) if self.cases else 0.0

    @property
    def overall(self) -> float:
        """Correct fields over labeled fields, across all cases."""
        scored = [ok for c in self.cases for ok in c.fields.values()]
        return sum(scored) / len(scored) if scored else 0.0

    @property
    def fields(self) -> dict[str, float]:
        """Accuracy per field, with list indexes folded: ``items[*].price``."""
        tally: dict[str, list[int]] = {}
        for c in self.cases:
            for path, ok in c.fields.items():
                t = tally.setdefault(_INDEX.sub("[*]", path), [0, 0])
                t[0] += ok
                t[1] += 1
        return {path: hits / total for path, (hits, total) in sorted(tally.items())}

    def metrics(self) -> dict[str, float]:
        return {VALIDITY: self.validity, OVERALL: self.overall, **self.fields}

    def summary(self) -> str:
        lines = [
            f"cases     {len(self.cases)}",
            f"valid     {self.validity:6.1%}",
            f"overall   {self.overall:6.1%}",
            "",
        ]
        fields = self.fields
        width = max((len(p) for p in fields), default=5)
        lines += [f"{path:<{width}}  {acc:6.1%}" for path, acc in fields.items()]
        failed = [c for c in self.cases if not c.valid]
        if failed:
            lines += ["", "invalid:"] + [f"  {c.id}: {c.error}" for c in failed]
        return "\n".join(lines)

    def compare(self, baseline: Report, max_drop: float = 0.0) -> Diff:
        """Compare with ``baseline``. A metric regresses if it dropped by more
        than ``max_drop`` (0.02 = two points)."""
        now, before = self.metrics(), baseline.metrics()
        deltas = {k: round(now[k] - before[k], 6) for k in before if k in now}
        regressions = {k: d for k, d in deltas.items() if d < -max_drop}
        return Diff(ok=not regressions, regressions=regressions, deltas=deltas)

    def to_dict(self) -> dict[str, Any]:
        return {"metrics": self.metrics(), "cases": [asdict(c) for c in self.cases]}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> Report:
        data = json.loads(Path(path).read_text())
        return cls([CaseResult(**c) for c in data["cases"]])


def load_cases(path: str | Path) -> list[Case]:
    """Read cases from ``.jsonl`` (one per line) or ``.json`` (a list). Each
    needs ``input`` and ``expected``; ``id`` is optional."""
    text = Path(path).read_text()
    if str(path).endswith(".jsonl"):
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        rows = json.loads(text)
    return _as_cases(rows)


def _as_cases(cases: Iterable[Case | Mapping[str, Any]]) -> list[Case]:
    out = []
    for i, c in enumerate(cases, start=1):
        if isinstance(c, Case):
            out.append(c)
        else:
            out.append(Case(id=str(c.get("id", f"case-{i}")), input=c["input"], expected=dict(c["expected"])))
    return out


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """``{"items": [{"price": 1}]}`` -> ``{"items[0].price": 1}``."""
    if isinstance(value, Mapping) and value:
        out: dict[str, Any] = {}
        for k, v in value.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    if isinstance(value, list) and value:
        out = {}
        for i, v in enumerate(value):
            out.update(_flatten(v, f"{prefix}[{i}]"))
        return out
    return {prefix: value}


def _pick(path: str, comparators: Mapping[str, Comparator], default: Comparator) -> Comparator:
    if path in comparators:
        return comparators[path]
    return comparators.get(_INDEX.sub("[*]", path), default)


def _parse(schema: type[BaseModel], raw: Any) -> dict[str, Any]:
    """Target output -> validated, JSON-shaped dict. Raises on anything invalid."""
    if isinstance(raw, BaseModel):
        model = schema.model_validate(raw.model_dump())
    elif isinstance(raw, str):
        fenced = _FENCE.match(raw)
        model = schema.model_validate_json(fenced.group(1) if fenced else raw)
    else:
        model = schema.model_validate(raw)
    return model.model_dump(mode="json")


def _score(
    schema: type[BaseModel],
    case: Case,
    raw: Any,
    raised: BaseException | None,
    comparators: Mapping[str, Comparator],
    default: Comparator,
) -> CaseResult:
    labeled = _flatten(case.expected)
    if raised is not None:
        return CaseResult(case.id, False, f"target raised {type(raised).__name__}: {raised}", dict.fromkeys(labeled, False))
    try:
        actual = _flatten(_parse(schema, raw))
    except ValidationError as exc:
        kind = "invalid JSON" if any(e["type"] == "json_invalid" for e in exc.errors()) else "schema violation"
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"]) or "<root>"
        return CaseResult(case.id, False, f"{kind} at {where}: {first['msg']}", dict.fromkeys(labeled, False))
    except Exception as exc:  # e.g. a model instance that won't dump; still just invalid output
        return CaseResult(case.id, False, f"invalid output: {type(exc).__name__}: {exc}", dict.fromkeys(labeled, False))

    fields = {}
    for path, expected in labeled.items():
        fields[path] = path in actual and bool(_pick(path, comparators, default)(expected, actual[path]))
    return CaseResult(case.id, True, None, fields)


def evaluate(
    schema: type[BaseModel],
    cases: Iterable[Case | Mapping[str, Any]],
    target: Callable[[Any], Any],
    comparators: Mapping[str, Comparator] | None = None,
    *,
    default: Comparator | None = None,
    concurrency: int = 1,
) -> Report:
    """Run ``target`` on every case's input and score its output.

    ``target`` may return a JSON string (```json fences are fine), a dict, or
    a model instance, and may be sync or async. Only fields present in a
    case's ``expected`` are scored. ``comparators`` maps a field path
    (``total``) or wildcard path (``items[*].price``) to a comparator;
    everything else uses ``default`` (``exact()``). A target that raises or
    returns invalid output marks the case invalid; it never stops the run.
    """
    case_list = _as_cases(cases)
    comps = dict(comparators or {})
    fallback = default or exact()
    workers = max(1, concurrency)

    def call(case: Case) -> tuple[Any, BaseException | None]:
        try:
            return target(case.input), None
        except Exception as exc:  # the eval reports a broken target, never crashes on it
            return None, exc

    if inspect.iscoroutinefunction(target):
        async def run_all() -> list[tuple[Any, BaseException | None]]:
            gate = asyncio.Semaphore(workers)

            async def one(case: Case) -> tuple[Any, BaseException | None]:
                async with gate:
                    try:
                        return await target(case.input), None
                    except Exception as exc:
                        return None, exc

            return await asyncio.gather(*(one(c) for c in case_list))

        outputs = asyncio.run(run_all())
    elif workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            outputs = list(pool.map(call, case_list))
    else:
        outputs = [call(c) for c in case_list]

    return Report([_score(schema, c, raw, err, comps, fallback) for c, (raw, err) in zip(case_list, outputs)])
