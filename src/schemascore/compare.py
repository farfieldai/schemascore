"""Comparators: how a field's expected value is checked against the actual one.

A comparator is any ``(expected, actual) -> bool``. These cover the common
cases; write your own for anything else.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from difflib import SequenceMatcher
from typing import Any

Comparator = Callable[[Any, Any], bool]


def _norm(value: Any) -> str:
    """Collapse whitespace and casefold, for text comparisons."""
    return " ".join(str(value).split()).casefold()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _to_number(value: Any) -> float | None:
    if _is_number(value):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip().replace(",", ""))
        except ValueError:
            return None
    return None


def _named(fn: Comparator, name: str) -> Comparator:
    fn.__name__ = fn.__qualname__ = name
    return fn


def exact(normalize: bool = True) -> Comparator:
    """Equal values. With ``normalize``, strings match after collapsing
    whitespace and ignoring case. Numbers match within float rounding."""

    def compare(expected: Any, actual: Any) -> bool:
        if _is_number(expected) and _is_number(actual):
            return math.isclose(expected, actual, rel_tol=1e-9, abs_tol=1e-12)
        if normalize and isinstance(expected, str) and isinstance(actual, str):
            return _norm(expected) == _norm(actual)
        return bool(expected == actual)

    return _named(compare, f"exact(normalize={normalize})")


def numeric(abs_tol: float = 0.0, rel_tol: float = 0.0) -> Comparator:
    """Numbers within a tolerance. Numeric strings like ``"1,250.00"`` are
    converted; anything that isn't a number is a miss."""

    def compare(expected: Any, actual: Any) -> bool:
        e, a = _to_number(expected), _to_number(actual)
        if e is None or a is None or math.isnan(e) or math.isnan(a):
            return False
        return math.isclose(e, a, rel_tol=rel_tol, abs_tol=abs_tol)

    return _named(compare, f"numeric(abs_tol={abs_tol}, rel_tol={rel_tol})")


def fuzzy(threshold: float = 0.85) -> Comparator:
    """Text similar enough: difflib's ratio, after normalizing, of at least
    ``threshold`` (0–1). Good for names with small spelling differences."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    def compare(expected: Any, actual: Any) -> bool:
        if actual is None:
            return expected is None
        return SequenceMatcher(None, _norm(expected), _norm(actual)).ratio() >= threshold

    return _named(compare, f"fuzzy({threshold})")


def contains() -> Comparator:
    """The expected text appears inside the actual text, after normalizing."""

    def compare(expected: Any, actual: Any) -> bool:
        return actual is not None and _norm(expected) in _norm(actual)

    return _named(compare, "contains()")


def one_of(synonyms: Iterable[Any]) -> Comparator:
    """Values that mean the same thing, e.g. ``one_of(["USD", "US$", "$"])``.
    A match is the exact (normalized) value, or any two values in the group."""
    group = {_norm(s) for s in synonyms}

    def compare(expected: Any, actual: Any) -> bool:
        if actual is None:
            return expected is None
        e, a = _norm(expected), _norm(actual)
        return e == a or (e in group and a in group)

    return _named(compare, f"one_of({sorted(group)})")
