"""Build a test set automatically from the official XBRL numbers in each filing.

Each question has a correct answer straight from SEC data, and 'gold pages': pdf pages whose text
contains that number as printed (for example 391,035 when the table is in millions).
"""
import random
import re
from datetime import date

# (label used in the question, XBRL concepts to try in order, kind)
METRICS = [
    ("total revenue", ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"], "flow"),
    ("net income", ["NetIncomeLoss"], "flow"),
    ("operating income", ["OperatingIncomeLoss"], "flow"),
    ("gross profit", ["GrossProfit"], "flow"),
    ("research and development expense", ["ResearchAndDevelopmentExpense"], "flow"),
    ("income tax expense", ["IncomeTaxExpenseBenefit"], "flow"),
    ("net cash provided by operating activities", ["NetCashProvidedByUsedInOperatingActivities"], "flow"),
    ("diluted earnings per share", ["EarningsPerShareDiluted"], "flow"),
    ("total assets", ["Assets"], "instant"),
    ("total liabilities", ["Liabilities"], "instant"),
    ("total shareholders' equity", ["StockholdersEquity"], "instant"),
    ("cash and cash equivalents", ["CashAndCashEquivalentsAtCarryingValue"], "instant"),
]

FLOW_TEMPLATES = [
    "What was {co}'s {label} for the fiscal year ended {date}?",
    "How much {label} did {co} report for fiscal {year}?",
    "In its annual report, what {label} did {co} record for the year ending {date}?",
]
INSTANT_TEMPLATES = [
    "What were {co}'s {label} as of {date}?",
    "How much {label} did {co} have on its balance sheet at {date}?",
]


def _d(s: str) -> date:
    return date.fromisoformat(s)


def _nice_date(s: str) -> str:
    d = _d(s)
    return f"{d.strftime('%B')} {d.day}, {d.year}"


def facts_in_filing(facts: dict, accession: str) -> list:
    """Values reported in this exact filing: current year plus the comparison years shown beside it."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    out = []
    for label, concepts, kind in METRICS:
        for concept in concepts:
            units = gaap.get(concept, {}).get("units", {})
            unit = "USD/shares" if "USD/shares" in units else "USD"
            rows = [e for e in units.get(unit, []) if e.get("accn") == accession and e.get("form") == "10-K"]
            if kind == "flow":
                rows = [e for e in rows if e.get("start") and 340 <= (_d(e["end"]) - _d(e["start"])).days <= 380]
            else:
                rows = [e for e in rows if not e.get("start")]
            by_end = {}
            for e in rows:
                by_end.setdefault(e["end"], e["val"])
            if not by_end:
                continue
            keep = 3 if kind == "flow" else 2
            for end in sorted(by_end, reverse=True)[:keep]:
                out.append({"metric": label, "concept": concept, "kind": kind, "unit": unit,
                            "end": end, "value": by_end[end]})
            break  # first concept with data wins
    return out


def printed_forms(value: float, unit: str) -> list:
    """How the number is likely printed in the filing's tables."""
    v = abs(value)
    if unit == "USD/shares":
        return [f"{v:.2f}"]
    forms = []
    for scale in (1e6, 1e3):
        x = v / scale
        if x >= 1000 and abs(x - round(x)) < 1e-6:
            forms.append(f"{int(round(x)):,}")
    if v >= 1e9:
        forms.append(f"{v / 1e9:.1f} billion")
    return forms


def gold_pages(value: float, unit: str, page_texts: list) -> list:
    pats = [re.compile(r"(?<![\d,.])" + re.escape(f) + r"(?![\d,])") for f in printed_forms(value, unit)]
    if not pats:
        return []
    return [i + 1 for i, t in enumerate(page_texts) if any(p.search(t) for p in pats)]


def make_questions(ticker: str, company: str, facts: dict, accession: str, page_texts: list,
                   seed: int = 0, max_gold: int = 8) -> list:
    rng = random.Random(f"{seed}:{ticker}")
    qs = []
    for f in facts_in_filing(facts, accession):
        tpl = rng.choice(FLOW_TEMPLATES if f["kind"] == "flow" else INSTANT_TEMPLATES)
        q = tpl.format(co=company, label=f["metric"], date=_nice_date(f["end"]), year=_d(f["end"]).year)
        gp = gold_pages(f["value"], f["unit"], page_texts)
        qs.append({
            "qid": f"{ticker}:{f['concept']}:{f['end']}",
            "ticker": ticker, "question": q, "metric": f["metric"], "concept": f["concept"],
            "period_end": f["end"], "unit": f["unit"], "value": f["value"],
            "gold_pages": " ".join(map(str, gp)) if 0 < len(gp) <= max_gold else "",
            "type": "numeric",
        })
    return qs
