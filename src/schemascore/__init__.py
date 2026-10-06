"""Field-level evals for structured LLM output.

Validate outputs against a Pydantic schema, report validity and per-field
accuracy, and fail CI when a prompt or model change makes fields worse.
"""

from .compare import Comparator, contains, exact, fuzzy, numeric, one_of
from .core import Case, CaseResult, Diff, Report, evaluate, load_cases

__all__ = [
    "Case",
    "CaseResult",
    "Comparator",
    "Diff",
    "Report",
    "contains",
    "evaluate",
    "exact",
    "fuzzy",
    "load_cases",
    "numeric",
    "one_of",
]
__version__ = "0.1.0"
