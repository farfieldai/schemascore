"""``schemascore run``: evaluate from the command line and gate CI on the result.

Exit codes: 0 pass, 1 a quality gate failed, 2 a usage error.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Any

from .core import Report, evaluate, load_cases


def _load(ref: str) -> Any:
    """``package.module:attribute`` -> the attribute."""
    module, sep, attr = ref.partition(":")
    if not sep or not module or not attr:
        raise SystemExit(f"schemascore: expected module:attribute, got {ref!r}")
    # Let `examples.invoice:Invoice` resolve from the directory you run in.
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    obj: Any = importlib.import_module(module)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="schemascore", description="Field-level evals for structured LLM output.")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="evaluate a target over labeled cases")
    r.add_argument("--schema", required=True, help="Pydantic model, as module:Class")
    r.add_argument("--target", required=True, help="extraction function, as module:function")
    r.add_argument("--cases", required=True, type=Path, help=".jsonl or .json file of cases")
    r.add_argument("--comparators", help="dict of field path -> comparator, as module:NAME")
    r.add_argument("--concurrency", type=int, default=1, help="cases to run at once (default 1)")
    r.add_argument("--out", type=Path, help="write this run's report here")
    r.add_argument("--baseline", type=Path, help="report to compare against")
    r.add_argument("--max-drop", type=float, default=0.0,
                   help="fail if any metric drops more than this vs the baseline (0.02 = 2 points)")
    r.add_argument("--min-overall", type=float, help="fail if overall accuracy is below this (0-1)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.baseline and not args.baseline.exists():
        print(f"schemascore: baseline {args.baseline} not found (create one with --out)", file=sys.stderr)
        return 2

    report = evaluate(
        _load(args.schema),
        load_cases(args.cases),
        _load(args.target),
        _load(args.comparators) if args.comparators else None,
        concurrency=args.concurrency,
    )
    print(report.summary())
    if args.out:
        report.save(args.out)

    failed = False
    if args.baseline:
        diff = report.compare(Report.load(args.baseline), max_drop=args.max_drop)
        if diff.ok:
            print(f"\nno regressions vs {args.baseline} (max drop {args.max_drop:.1%})")
        else:
            failed = True
            print(f"\nregressions vs {args.baseline} (max drop {args.max_drop:.1%}):")
            for metric, delta in diff.regressions.items():
                print(f"  {metric}  {delta:+.1%}")
    if args.min_overall is not None and report.overall < args.min_overall:
        failed = True
        print(f"\noverall {report.overall:.1%} is below the minimum {args.min_overall:.1%}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
