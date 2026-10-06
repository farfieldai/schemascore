"""Example: scoring an invoice extractor.

`extract` stands in for an LLM call. It returns canned model output for each
example invoice, including the kinds of mistakes real models make: a vendor
name slightly off, a misread price, a wrong currency, and truncated JSON.

    schemascore run --schema examples.invoice:Invoice --target examples.invoice:extract \\
        --cases examples/cases.jsonl --comparators examples.invoice:COMPARATORS
"""

from pydantic import BaseModel

from schemascore import fuzzy, numeric, one_of


class LineItem(BaseModel):
    description: str
    price: float


class Invoice(BaseModel):
    vendor: str
    total: float
    currency: str
    items: list[LineItem] = []


COMPARATORS = {
    "vendor": fuzzy(0.85),
    "total": numeric(abs_tol=0.01),
    "currency": one_of(["USD", "US$", "$"]),
    "items[*].price": numeric(abs_tol=0.01),
}

# What a model said for each invoice, keyed by the invoice's first line.
_RESPONSES = {
    "INVOICE #1001": """```json
{"vendor": "Acme Corp", "total": 1250.00, "currency": "USD",
 "items": [{"description": "Widgets", "price": 1000.00}, {"description": "Shipping", "price": 250.00}]}
```""",
    "INVOICE #1002": '{"vendor": "Globex Corporaton", "total": "89.99", "currency": "US$",'
    ' "items": [{"description": "Annual licence", "price": 89.99}]}',
    "INVOICE #1003": '{"vendor": "Initech", "total": 430.0, "currency": "EUR",'
    ' "items": [{"description": "TPS reports", "price": 304.0}, {"description": "Staplers", "price": 90.0}]}',
    "INVOICE #1004": '{"vendor": "Umbrella", "total": 75.5, "currency": "USD", "items": [{"descr',
}


def extract(text: str) -> str:
    return _RESPONSES[text.splitlines()[0].strip()]
